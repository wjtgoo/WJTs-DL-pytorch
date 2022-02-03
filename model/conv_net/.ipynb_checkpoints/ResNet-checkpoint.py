import torch
from torch import nn, optim
import torch.nn.functional as F
from .utils import FlattenLayer, GlobalAvgPool2d
# Residual layer
class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, use_1x1conv=False, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None
    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        return F.relu(Y+X)

# resnet block
def resnet_block(in_channels, out_channels, num_residuals, first_block=False):
    # first block need the same input channels and output channels
    if first_block:
        assert in_channels == out_channels 
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(in_channels, out_channels, use_1x1conv=True, stride=2))
        else:
            blk.append(Residual(out_channels, out_channels))
    return nn.Sequential(*blk)

# ResNet
class ResNet(nn.Module):
    '''
    params:
        img_channels: nums of image channels
        class_num: all classes which need to be classfied
    '''
    def __init__(self, img_channels, class_num):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(img_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.resnet_blocks = nn.Sequential(
            resnet_block(64, 64, 2, first_block=True),
            resnet_block(64, 128, 2),
            resnet_block(128, 256, 2),
            resnet_block(256, 512, 2)
        )
        self.global_avg_pool = GlobalAvgPool2d()
        self.fc = nn.Sequential(
            FlattenLayer(),
            nn.Linear(512, class_num)
        )
    def forward(self, X):
        Y = self.head(X)
        Y = self.resnet_blocks(Y)
        Y = self.global_avg_pool(Y)
        Y = self.fc(Y)
        return Y
    