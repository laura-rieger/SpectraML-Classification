import torch
import torch.utils.data
from torch.nn import functional as F
from torch import nn
import numpy as np

class FGANet(nn.Module):

    def __init__(self,
                 num_input=4000,
                 conv_channels=4,
                 num_output=10,
                 num_in_channels=1,
                 kernel_size=3,
                 stride=1):
        super(FGANet, self).__init__()

        self.features = nn.Sequential(
            nn.Conv1d(num_in_channels, conv_channels, kernel_size, stride),
            nn.ReLU(),  #3, 1 stride
            nn.BatchNorm1d(conv_channels),
            nn.Conv1d(conv_channels, conv_channels, kernel_size,
                      stride),  #5, 1 stride
            nn.ReLU(),
            nn.BatchNorm1d(conv_channels),
            nn.MaxPool1d(2),  # 6, 2 stride
            nn.Conv1d(conv_channels, conv_channels * 2, kernel_size,
                      stride),  # 10, 2 stride
            nn.ReLU(),
            nn.BatchNorm1d(conv_channels * 2),
            nn.Conv1d(
                conv_channels * 2,
                conv_channels * 2,
                kernel_size,  #14, 2 stride
                stride),
            nn.ReLU(),
            nn.MaxPool1d(2),  #16, 4 stride
            nn.BatchNorm1d(conv_channels * 2, affine=False),
            nn.Flatten())
        self.num_dense_input = self.features.forward(
            torch.Tensor(np.zeros((2, num_in_channels, num_input)))).shape[1]
        # print('Hidden representation size: ', self.num_dense_input)
   
        self.classifier = nn.Sequential(
            nn.Linear(self.num_dense_input, num_output), )

        # self.classifier = nn.Sequential(nn.Linear(self.num_dense_input, 128),
        #                                 nn.ReLU(), nn.Linear(128, num_output))

    def forward_repr(self, x):
        x = self.features(x)
        x = self.classifier[0](x)
        return x

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        x = torch.sigmoid(x)
        return x


if __name__ == "__main__":
    my_model = FGANet(num_input=850, stride=1)
    test_input = torch.Tensor(np.zeros((42, 2, 850)))
    # print(test_input.shape)
    print("Out size",
          my_model.features.forward(torch.Tensor(test_input)).shape)
    print(my_model.forward(torch.Tensor(test_input)).shape)