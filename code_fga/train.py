#%%
import os
import sys
import pickle as pkl
from argparse import ArgumentParser
from copy import deepcopy
from os.path import join as oj
import numpy as np
import pickle
import random
import configparser
import torch
from torch import nn
import torch.utils.data
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau

sys.path.insert(0, "../code_fga")
import utils
import models
import data_fns
import my_eval

cuda = torch.cuda.is_available()
if not cuda:
    sys.exit()
device = torch.device("cuda")

random.seed(0)
np.random.seed()

config = configparser.ConfigParser()
config.read("../config.ini")
torch.backends.cudnn.deterministic = True


def get_args():
    parser = ArgumentParser(description="Functional group analysis")

    parser.add_argument("--exp_name", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_epochs", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--num_conv", type=int, default=16)
    parser.add_argument("--patience", type=int, default=5)
    ret_args = parser.parse_args()
    return ret_args


args = get_args()

data_path = config["PATHS"]["data_path"]
save_path = config["PATHS"]["model_path"]
if not os.path.exists(save_path):
    os.makedirs(save_path)

# load data
PATH_IR_INDEX = oj(data_path, "ir_spectra_index.pkl")
PATH_IR_DATA = oj(data_path, "ir_spectra_data.pkl")

with open(PATH_IR_DATA, "rb") as pickle_file:
    irdata = pickle.load(pickle_file)

with open(PATH_IR_INDEX, "rb") as pickle_file:
    irindex = pickle.load(pickle_file)

filters = (("state", "gas"), ("yunits", "ABSORBANCE"), ("xunits", "1/CM"))
data, labels = utils.fixed_domain_filter(
    irdata, irindex, filters
)  # Filter points from the dataset


data /= data.max(axis=1)[:, None]  # Normalisation

data = data[:, None]


#%%
num_classes = labels.shape[1]
train_idxs, val_idxs, test_idxs = data_fns.get_split(len(data), seed=42)

# scale data
weights = labels[train_idxs].mean(axis=0)
anti_weights = 1 - weights

mult_weights = weights * anti_weights
weights /= mult_weights
anti_weights /= mult_weights

weights = torch.Tensor(weights).to(device)
anti_weights = torch.Tensor(anti_weights).to(device)

#%%
# create datasets in torch
torch.manual_seed(args.seed)
train_dataset = TensorDataset(
    *[torch.Tensor(input) for input in [data[train_idxs], labels[train_idxs]]],
)
val_dataset = TensorDataset(
    *[torch.Tensor(input) for input in [data[val_idxs], labels[val_idxs]]],
)

train_loader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
print(num_classes)
sys.exit()
# model creation
model = models.FGANet(
    num_input=data.shape[2],
    num_output=num_classes,
    conv_channels=args.num_conv,
    num_in_channels=data.shape[1],
    stride=1,
).to(device)


# training
def train(max_patience=20, num_epochs=1000, batch_size=128):

    optimizer = optim.Adam(model.parameters(), )
    # scheduler = optim.lr_scheduler.

    training_loss = []
    validation_loss = []
    validation_accs = []

    best_val_loss = 500000

    cur_patience = 0
    patience_delta = 0
    best_weights = None

    for epoch in range(num_epochs):

        model.train()

        tr_loss = 0

        for _, (
            data_cur,
            labels_cur,
        ) in enumerate(train_loader):
            data_cur = data_cur.to(device)

            labels_cur = labels_cur.to(device)
            optimizer.zero_grad()
            y_pred = model(data_cur)

            # cur_loss = loss_function(model(data_cur), labels_cur)
            cur_loss_all = nn.functional.binary_cross_entropy(
                y_pred, labels_cur, reduction="none"
            )
            cur_loss = (
                cur_loss_all
                * (
                    anti_weights[None, :] * labels_cur
                    + weights[None, :] * (1 - labels_cur)
                )
            ).mean()

            l2_lambda = 0.001
            l2_norm = sum(p.pow(2.0).sum() for p in model.features[-5].parameters())
            (cur_loss+l2_norm).backward()
            tr_loss += cur_loss.item()
            optimizer.step()

        tr_loss /= len(train_loader.dataset)
        training_loss.append(tr_loss)

        model.eval()
        val_loss = 0
        val_acc = 0

        # validation
        with torch.no_grad():
            for _, (
                data_cur,
                labels_cur,
            ) in enumerate(val_loader):
                data_cur = data_cur.to(device)
                labels_cur = labels_cur.to(device)

                y_pred = model(data_cur)

                cur_loss_all = nn.functional.binary_cross_entropy(
                    y_pred, labels_cur, reduction="none"
                )
                cur_loss = (
                    cur_loss_all
                    * (
                        anti_weights[None, :] * labels_cur
                        + weights[None, :] * (1 - labels_cur)
                    )
                ).mean()
                val_loss += cur_loss.item()

                val_corr = ((y_pred > 0.5) == labels_cur).sum()

                val_acc += val_corr.item()

        val_acc /= len(val_loader.dataset) * num_classes
        val_loss /= len(val_loader.dataset)

        validation_loss.append(val_loss)
        validation_accs.append(val_acc)

        print(
            "Epoch: %d, TrLoss: %1.5f, ValLoss: %1.5f, ValAcc: %1.2f "
            % (epoch + 1, tr_loss, val_loss, val_acc)
        )

        if val_loss + patience_delta < best_val_loss:
            best_weights = deepcopy(model.state_dict())
            cur_patience = 0
            best_val_loss = val_loss
        else:
            cur_patience += 1
        if cur_patience > max_patience:
            break

    print("Training finished")

    model.load_state_dict(best_weights)
    np.random.seed()

    file_name = "".join([str(np.random.choice(10)) for x in range(10)])

    results = {}
    results["filename"] = file_name
    for arg in vars(args):
        if arg != "save_path":
            results[str(arg)] = getattr(args, arg)
    results["train_losses"] = training_loss
    results["val_losses"] = validation_loss
    results["best_val_loss"] = best_val_loss
    results["val_auc"] = my_eval.calculate_roc_auc_score(
        model, device, data[val_idxs], labels[val_idxs], batch_size
    )
    print("Validation AUC: ", results["val_auc"])

    pkl.dump(results, open(os.path.join(save_path, file_name + ".pkl"), "wb"))
    torch.save(model.state_dict(), oj(save_path, file_name + ".pt"))
    print("Saved model and results")

    return results


if __name__ == "__main__":
    train(
        num_epochs=args.num_epochs,
        max_patience=args.patience,
        batch_size=args.batch_size,
    )
