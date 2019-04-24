import torch
import torch.nn as nn

class FC_1(nn.Module):
    def __init__(self, D_in, D_h, D_out):

        super(FC_1, self.__init__())
        self.relu = nn.ReLU()
        self.linear1 = nn.Linear(D_in,D_h)
        self.linear2 = nn.Linear(D_h, D_out)

    def forward(self, x):
        h1 = self.relu(self.linear1(x))
        y_pred = self.relu(self.linear2(h1))