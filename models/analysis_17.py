#!/Library/Frameworks/Python.framework/Versions/3.5/bin/python3.5
import torch.nn as nn
import torch
from .GDN import GDN
import math
from ELICUtilis.layers import (
    AttentionBlock,
    conv3x3,
    CheckboardMaskedConv2d,
)
from torch import Tensor

def conv1x1(in_ch: int, out_ch: int, stride: int = 1) -> nn.Module:
    """1x1 convolution."""
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride)

class ResidualBottleneckBlock(nn.Module):
    """Simple residual block with two 3x3 convolutions.

    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels
    """

    def __init__(self, in_ch: int):
        super().__init__()
        self.conv1 = conv1x1(in_ch, in_ch//2)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(in_ch//2, in_ch//2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = conv1x1(in_ch//2, in_ch)

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.conv3(out)

        out = out + identity
        return out

class Analysis_net_17(nn.Module):
    '''
    Analysis net
    '''
    def __init__(self, out_channel_N=192):

        super(Analysis_net_17, self).__init__()

        self.conv1 = nn.Conv2d(3, out_channel_N, 9, stride=4, padding=4)
        torch.nn.init.xavier_normal_(self.conv1.weight.data, (math.sqrt(2 * (3 + out_channel_N) / (6)))) # 가중치 초기화
        torch.nn.init.constant_(self.conv1.bias.data, 0.01) # bias 초기화
        # self.gdn1 = GDN(out_channel_N)
        self.res1_1 = ResidualBottleneckBlock(out_channel_N)
        self.res1_2 = ResidualBottleneckBlock(out_channel_N)
        self.res1_3 = ResidualBottleneckBlock(out_channel_N)

        self.conv2 = nn.Conv2d(out_channel_N, out_channel_N, 5, stride=2, padding=2)
        torch.nn.init.xavier_normal_(self.conv2.weight.data, math.sqrt(2))
        torch.nn.init.constant_(self.conv2.bias.data, 0.01)
        self.res2_1 = ResidualBottleneckBlock(out_channel_N)
        self.res2_2 = ResidualBottleneckBlock(out_channel_N)
        self.res2_3 = ResidualBottleneckBlock(out_channel_N)
        # self.gdn2 = GDN(out_channel_N)
        
        self.conv3 = nn.Conv2d(out_channel_N, out_channel_N, 5, stride=2, padding=2, bias=False)
        torch.nn.init.xavier_normal_(self.conv3.weight.data, math.sqrt(2))

        # torch.nn.init.constant_(self.conv3.bias.data, 0.01)
        # self.gdn3 = GDN(out_channel_N)
        # self.conv4 = nn.Conv2d(out_channel_N, out_channel_M, 5, stride=2, padding=2)
        # torch.nn.init.xavier_normal_(self.conv4.weight.data, (math.sqrt(2 * (out_channel_M + out_channel_N) / (out_channel_N + out_channel_N))))
        # torch.nn.init.constant_(self.conv4.bias.data, 0.01)

    def forward(self, x):
        # x = self.gdn1(self.conv1(x))
        # x = self.gdn2(self.conv2(x))
        x = self.conv1(x)
        x = self.res1_1(x)
        x = self.res1_2(x)
        x = self.res1_3(x)
        
        # Apply conv2 and the second set of residual blocks
        x = self.conv2(x)
        x = self.res2_1(x)
        x = self.res2_2(x)
        x = self.res2_3(x)
        
        # Apply the final conv3 layer
        x = self.conv3(x)
        return x


def build_model():
        input_image = torch.zeros([4, 3, 256, 256])

        analysis_net = Analysis_net_17()
        feature = analysis_net(input_image)

        print(feature.size())


if __name__ == '__main__':
    build_model()
