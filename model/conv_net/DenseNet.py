import torch
from torch import nn, optim
import torch.nn.functional as F
from .utils import FlattenLayer, GlobalAvgPool2d
'''
DenseNet
    composed of `dense block` and `transition layer`<br>
'''
'''
1. dense block
    it's so similar to ResNet that the only difference is junction which resnet use plus to connect back and front layer while densenet use concatenation
    it's consist of simple unit named conv_block whose structrue is BN,ReLU&conv2d 
'''
# convolution block
def conv_block(in_channels, out_channels):
    blk = nn.Sequential(nn.BatchNorm2d(in_channels), 
                        nn.ReLU(),
                        nn.Conv2d(in_channels,out_channels, kernel_size=3, padding=1)
                       )
    return blk
# dense block
class DenseBlock(nn.Module):
    '''
    params:
        num_convs: the num of conv_block
        in_channels: input channels
        growth_rate: output channels of conv_block
    tips: the real nums of output channel = in_channels+num_convs*growth_rate
    '''
    def __init__(self, num_convs, in_channels, growth_rate):
        super().__init__()
        net = []
        for i in range(num_convs):
            in_c = in_channels + i * growth_rate
            net.append(conv_block(in_c, growth_rate))
        self.net = nn.ModuleList(net)
        self.out_channels = in_channels + num_convs * growth_rate
    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            X = torch.cat((X, Y), dim=1)
        return X
'''
2. transition layer
    reduce the number of dense block's output channel and half image size
'''
def transition_block(in_channels, out_channels):
    blk = nn.Sequential(
            nn.BatchNorm2d(in_channels), 
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.AvgPool2d(kernel_size=2, stride=2))
    return blk

# DenseNet
class DenseNet(nn.Module):
    def __init__(self, img_channels, class_num):
        super().__init__()
        # net's head is conv2d, bn, relu&mp
        self.net = nn.Sequential(
            nn.Conv2d(img_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.num_channels = 64 # current channel num
        self.growth_rate = 32 # Denseblock's out_channels
        # dense block part
        # here I use 4 dense block, each block have 4 convlution block
        self.num_convs_in_dense_blocks = [4, 4, 4, 4]
        for i, num_convs in enumerate(self.num_convs_in_dense_blocks):
            DB = DenseBlock(num_convs, self.num_channels, self.growth_rate)
            self.net.add_module('DenseBlock_%d' % i, DB)
            # update last output channels
            self.num_channels = DB.out_channels
            # add transition layer between each dense block
            if i != len(self.num_convs_in_dense_blocks) - 1:
                self.net.add_module('transition_block_%d'%i, 
                            transition_block(self.num_channels,self.num_channels//2))
                self.num_channels = self.num_channels // 2
        # BN, ReLU, global avg_pooling, fully connetion
        self.net.add_module("BN", nn.BatchNorm2d(self.num_channels))
        self.net.add_module("relu", nn.ReLU())
        self.net.add_module("global_avg_pool", GlobalAvgPool2d()) # (Batch, num_channels, 1, 1)
        self.net.add_module("fc", nn.Sequential(FlattenLayer(), nn.Linear(self.num_channels, 10)))
    def forward(self, X):
        Y = self.net(X)
        return Y
