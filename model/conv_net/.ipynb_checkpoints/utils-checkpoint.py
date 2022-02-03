import torch
from torch import nn
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# use cnn model to predict image data
def use_model(img_arr, net):
    '''
    input: 
            img_arr: need to be transformed from dataloader.trans_data function
                     or it can be torch's offical data transformed by ToTensor
                     !! it can be a batch of images or one image
                     
            net: the model you trained
    '''
    net.eval()# 测试模式，此时会固定住dropout，BN的值
    device = next(net.parameters()).device
    img_arr.to(device)
    # add one dimension at axis 0
    if len(img_arr.shape)==3:
        img_arr = torch.unsqueeze(img_arr, dim=0)
    with torch.no_grad():
        out = net(img_arr)
        result = torch.argmax(out,dim=-1)
    return result

# global average pool : kernel size:x.shape
class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()
    def forward(self, x):
        return F.avg_pool2d(x, kernel_size=x.size()[2:])

# flatten layer
class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()
    def forward(self, x): # x shape: (batch, *, *, ...)
        return x.view(x.shape[0], -1)