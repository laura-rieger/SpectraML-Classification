import torch
from sklearn import metrics
def calculate_roc_auc_score(model, device, x, y, batch_size):
    pred_y = model.forward(torch.Tensor(x).to(device)).detach().cpu().numpy()
    return metrics.roc_auc_score(y, pred_y)

