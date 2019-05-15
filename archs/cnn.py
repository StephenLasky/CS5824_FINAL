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

class CNN_9(nn.Module):
    def __init__(self):
        super(CNN_9, self).__init__()

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
            nn.Linear(1536, 512),
            nn.BatchNorm1d(512),
            nn.ReLU())
        self.fc2 = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU())
        self.fc3 = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU())
        self.fc4 = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU())
        self.fc5 = nn.Sequential(
            nn.Linear(512, 1536),
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
        out = self.fc3(out)
        out = self.fc4(out)
        out = self.fc5(out)

        return out.view(-1, 3, 16, 32)

# 6 CNN layers, 1 linear
class CNN_10(nn.Module):
    def __init__(self):
        super(CNN_10, self).__init__()

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
            nn.ReLU())

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.conv6(out)
        out = self.fc1(out.view(-1, 1536))

        return out.view(-1, 3, 16, 32)

# new idea: step up CNN channels... 16->32->64->128. one single FC at the end
class CNN_11(nn.Module):
    def __init__(self):
        super(CNN_11, self).__init__()
        ks = 3
        padding = 1

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=ks, stride=1, padding=padding),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=ks, stride=1, padding=padding),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=ks, stride=1, padding=padding),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=ks, stride=1, padding=padding),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 3, kernel_size=ks, stride=1, padding=padding),
            nn.BatchNorm2d(3),
            nn.ReLU()
        )
        self.fc1 = nn.Sequential(
            nn.Linear(1536, 1536))

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.fc1(out.view(-1, 1536))

        return out.view(-1, 3, 16, 32)

# new idea: 3 convolutions followed by 1 linear, and then repeated
class CNN_12(nn.Module):
    def __init__(self):
        super(CNN_12, self).__init__()

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
            nn.Conv2d(n_channels, 3, kernel_size=ks, stride=1, padding=padding),
            nn.BatchNorm2d(3),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(3, n_channels, kernel_size=ks, stride=1, padding=padding),
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
            nn.Linear(1536, 1536))

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.fc1(out.view(-1, 1536))
        out = self.conv4(out.view(-1, 3, 16, 32))
        out = self.conv5(out)
        out = self.conv6(out)
        out = self.fc2(out.view(-1, 1536))

        return out.view(-1, 3, 16, 32)

# lightweight network ?
class CNN_13(nn.Module):
    def __init__(self):
        super(CNN_13, self).__init__()
        ks = 3
        padding = 1
        channels = 16

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, channels, kernel_size=ks, stride=1, padding=padding),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=ks, stride=1, padding=padding),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=ks, stride=1, padding=padding),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=ks, stride=1, padding=padding),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=ks, stride=1, padding=padding),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=ks, stride=1, padding=padding),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=ks, stride=1, padding=padding),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )
        self.conv8 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=ks, stride=1, padding=padding),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )
        self.conv9 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=ks, stride=1, padding=padding),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )
        self.conv10 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=ks, stride=1, padding=padding),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )
        self.conv11 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=ks, stride=1, padding=padding),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )
        self.conv12 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=ks, stride=1, padding=padding),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )
        self.conv13 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=ks, stride=1, padding=padding),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )
        self.conv14 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=ks, stride=1, padding=padding),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )
        self.conv15 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=ks, stride=1, padding=padding),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )
        self.conv16 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=ks, stride=1, padding=padding),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )
        self.conv17 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=ks, stride=1, padding=padding),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )
        self.conv18 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=ks, stride=1, padding=padding),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )
        self.conv19 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=ks, stride=1, padding=padding),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )
        self.conv20 = nn.Sequential(
            nn.Conv2d(channels, 3, kernel_size=ks, stride=1, padding=padding),
            nn.BatchNorm2d(3),
            nn.ReLU()
        )

        self.fc1 = nn.Sequential(
            nn.Linear(1536, 1536))

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.conv6(out)
        out = self.conv7(out)
        out = self.conv8(out)
        out = self.conv9(out)
        out = self.conv10(out)
        out = self.conv11(out)
        out = self.conv12(out)
        out = self.conv13(out)
        out = self.conv14(out)
        out = self.conv15(out)
        out = self.conv16(out)
        out = self.conv17(out)
        out = self.conv18(out)
        out = self.conv19(out)
        out = self.conv20(out)

        out = self.fc1(out.view(-1, 1536))

        return out.view(-1, 3, 16, 32)

# LOTS OF conv layers

# filter step-down
class CNN_10(nn.Module):
    def __init__(self):
        super(CNN_10, self).__init__()

        n_channels = 32
        ks = 3
        padding = int((ks - 1) / 2)

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, n_channels, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(n_channels),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(n_channels, n_channels, kernel_size=5, stride=1, padding=2),
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
            nn.ReLU())

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.conv6(out)
        out = self.fc1(out.view(-1, 1536))

        return out.view(-1, 3, 16, 32)