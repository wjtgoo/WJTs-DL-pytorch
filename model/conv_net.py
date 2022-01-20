import torch
from torch import nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Lenet
class LeNet(nn.Module):
    def __init__(self,img_size=(28,28)):
        super().__init__()
        # conv layers
        self.conv = nn.Sequential(
            # first conv layer
            nn.Conv2d(1, 6, 5), # in_channels, out_channels, kernels
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2), # kernel_size, stride
            # second conv layer
            nn.Conv2d(6, 16, 5), # in_channels, out_channels, kernels
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2), # kernel_size, stride
        )
        # full_connected  layer
        self.fc = nn.Sequential(
            # img_size=28*28
            # 16*4*4
            nn.Linear(int(16*(((img_size[0]-4)/2)-4)/2*(((img_size[1]-4)/2)-4)/2), 120),
            nn.Sigmoid(),
            nn.Linear(120, 84),
            nn.Sigmoid(),
            nn.Linear(84, 10)
        )
        
    def forward(self, img):
        feature = self.conv(img)
        output = self.fc(feature.view(img.shape[0], -1))# batch_size, flatten_img_size
        return output
