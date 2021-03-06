import torch
import torch.nn as nn

class FC_1(nn.Module):
    def __init__(self, D_in, D_h, D_out):
        super(FC_1, self).__init__()
        self.relu = nn.ReLU()
        self.linear1 = nn.Linear(D_in,D_h)
        self.linear2 = nn.Linear(D_h, D_out)

    def forward(self, x):
        h1 = self.relu(self.linear1(x))
        y_pred = self.relu(self.linear2(h1))
        return y_pred

class FC_2(nn.Module):
    def __init__(self, D_in, D_h, D_out):
        super(FC_2, self).__init__()
        self.relu = nn.ReLU()
        self.linear1 = nn.Linear(D_in,D_h)
        self.linear2 = nn.Linear(D_h, D_h)
        self.linear3 = nn.Linear(D_h, D_out)

    def forward(self, x):
        h1 = self.relu(self.linear1(x))
        h2 = self.relu(self.linear2(h1))
        y_pred = self.relu(self.linear3(h2))
        return y_pred

class FC_4(nn.Module):
    def __init__(self, D_in, D_h, D_out):
        super(FC_4, self).__init__()
        self.relu = nn.ReLU()
        self.linear1 = nn.Linear(D_in,D_h)
        self.linear2 = nn.Linear(D_h, D_h)
        self.linear3 = nn.Linear(D_h, D_h)
        self.linear4 = nn.Linear(D_h, D_h)
        self.linear5 = nn.Linear(D_h, D_out)

    def forward(self, x):
        h1 = self.relu(self.linear1(x))
        h2 = self.relu(self.linear2(h1))
        h3 = self.relu(self.linear3(h2))
        h4 = self.relu(self.linear4(h3))
        y_pred = self.relu(self.linear5(h4))
        return y_pred

class FC_8(nn.Module):
    def __init__(self, D_in, D_h, D_out):
        super(FC_8,self).__init__()
        self.relu = nn.ReLU()
        self.linear1 = nn.Linear(D_in, D_h)
        self.linear2 = nn.Linear(D_h, D_h)
        self.linear3 = nn.Linear(D_h, D_h)
        self.linear4 = nn.Linear(D_h, D_h)
        self.linear5 = nn.Linear(D_h, D_h)
        self.linear6 = nn.Linear(D_h, D_h)
        self.linear7 = nn.Linear(D_h, D_h)
        self.linear8 = nn.Linear(D_h, D_h)
        self.linear9 = nn.Linear(D_h, D_out)

    def forward(self, x):
        h1 = self.relu(self.linear1(x))
        h2 = self.relu(self.linear2(h1))
        h3 = self.relu(self.linear3(h2))
        h4 = self.relu(self.linear4(h3))
        h5 = self.relu(self.linear5(h4))
        h6 = self.relu(self.linear6(h5))
        h7 = self.relu(self.linear7(h6))
        h8 = self.relu(self.linear8(h7))
        y_pred = self.relu(self.linear9(h8))
        return y_pred
