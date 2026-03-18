import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, image_size, n_classes):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(image_size[0] * image_size[1], 128),
            nn.ReLU(),
        )
        self.fc = nn.Linear(128, n_classes)

    def forward(self, x):
        h = self.mlp(x)
        y = self.fc(h)
        return h, y


class CNN(nn.Module):
    def __init__(self, n_classes=2, dropout=0.5):
        super().__init__()
        self.conv_net = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(64, n_classes)

    def forward(self, x):
        h = self.conv_net(x)
        h_drop = self.dropout(h)
        y = self.fc(h_drop)
        return h, y
