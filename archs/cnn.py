import torch
import torch.nn as nn

class CNN_7(nn.Module):
    def __init__(self):
        super(CNN_7, self).__init__()

        n_channels = 16
        ks = 3
        padding = int((ks-1)/2)

        self.conv1 = nn.Sequential(
            nn.Conv2d(3,n_channels, kernel_size=ks, stride=1, padding=padding),
            nn.BatchNorm2d(n_channels),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(n_channels, n_channels, kernel_size=ks, stride=1, padding=padding),
            nn.BatchNorm2d(n_channels),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(n_channels,3, kernel_size=ks, stride=1, padding=padding),
            nn.BatchNorm2d(3),
            nn.ReLU()
        )


        self.fc1 = nn.Sequential(
            nn.Linear(1536, 1536),
            nn.BatchNorm1d(1536),
            nn.ReLU())
        self.fc2 = nn.Sequential(
            nn.Linear(1536, 1536),
            nn.ReLU())

    def forward(self,x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.fc1(out.view(-1,1536))
        out = self.fc2(out)

        return out.view(-1, 3, 16, 32)

class CNN_8(nn.Module):
    def __init__(self):
        super(CNN_8, self).__init__()

        n_channels = 32
        ks = 3
        padding = int((ks - 1) / 2)

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, n_channels, kernel_size=ks, stride=1, padding=padding),
            nn.BatchNorm2d(n_channels),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(n_channels, n_channels, kernel_size=ks, stride=1, padding=padding),
            nn.BatchNorm2d(n_channels),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(n_channels, n_channels, kernel_size=ks, stride=1, padding=padding),
            nn.BatchNorm2d(n_channels),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(n_channels, n_channels, kernel_size=ks, stride=1, padding=padding),
            nn.BatchNorm2d(n_channels),
            nn.ReLU()
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(n_channels, n_channels, kernel_size=ks, stride=1, padding=padding),
            nn.BatchNorm2d(n_channels),
            nn.ReLU()
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(n_channels, 3, kernel_size=ks, stride=1, padding=padding),
            nn.BatchNorm2d(3),
            nn.ReLU()
        )

        self.fc1 = nn.Sequential(
            nn.Linear(1536, 1536),
            nn.BatchNorm1d(1536),
            nn.ReLU())
        self.fc2 = nn.Sequential(
            nn.Linear(1536, 1536),
            nn.ReLU())

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.conv6(out)
        out = self.fc1(out.view(-1, 1536))
        out = self.fc2(out)

        return out.view(-1, 3, 16, 32)