from .analysis_17 import Analysis_net_17
import torch.nn as nn
from .GDN import GDN
import torch
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

class Synthesis_net_17(nn.Module):
    '''
    Decode synthesis
    '''

    def __init__(self, out_channel_N=192):
        super(Synthesis_net_17, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(out_channel_N, out_channel_N, 5, stride=2, padding=2, output_padding=1)
        torch.nn.init.xavier_normal_(self.deconv1.weight.data, (math.sqrt(2 * 1 )))
        torch.nn.init.constant_(self.deconv1.bias.data, 0.01)
        # self.igdn1 = GDN(out_channel_N, inverse=True)

        self.res1_1 = ResidualBottleneckBlock(out_channel_N)
        self.res1_2 = ResidualBottleneckBlock(out_channel_N)
        self.res1_3 = ResidualBottleneckBlock(out_channel_N)

        self.deconv2 = nn.ConvTranspose2d(out_channel_N, out_channel_N, 5, stride=2, padding=2, output_padding=1)
        torch.nn.init.xavier_normal_(self.deconv2.weight.data, math.sqrt(2 * 1))
        torch.nn.init.constant_(self.deconv2.bias.data, 0.01)
        #self.igdn2 = GDN(out_channel_N, inverse=True)
        self.res2_1 = ResidualBottleneckBlock(out_channel_N)
        self.res2_2 = ResidualBottleneckBlock(out_channel_N)
        self.res2_3 = ResidualBottleneckBlock(out_channel_N)

        self.deconv3 = nn.ConvTranspose2d(out_channel_N, 3, 9, stride=4, padding=4, output_padding=3)
        torch.nn.init.xavier_normal_(self.deconv3.weight.data, math.sqrt(2 * 1))
        torch.nn.init.constant_(self.deconv3.bias.data, 0.01)

    def forward(self, x):
        # x = self.igdn1(self.deconv1(x))
        # x = self.igdn2(self.deconv2(x))
        # x = self.res1_1(self.deconv1(x))
        # x = self.res1_2(self.deconv1(x))
        # x = self.res1_3(self.deconv1(x))
        # x = self.res2_1(self.deconv2(x))
        # x = self.res2_2(self.deconv2(x))
        # x = self.res2_3(self.deconv2(x))
        # x = self.deconv3(x)

        x = self.deconv1(x)
        x = self.res1_1(x)
        x = self.res1_2(x)
        x = self.res1_3(x)
        
        # Apply conv2 and the second set of residual blocks
        x = self.deconv2(x)
        x = self.res2_1(x)
        x = self.res2_2(x)
        x = self.res2_3(x)
        
        # Apply the final conv3 layer
        x = self.deconv3(x)
        return x

# synthesis_one_pass = tf.make_template('synthesis_one_pass', synthesis_net)

def build_model():
    input_image = torch.zeros([7,3,256,256])
    analysis_net = Analysis_net_17()
    synthesis_net = Synthesis_net_17()
    feature = analysis_net(input_image)
    recon_image = synthesis_net(feature)

    print("input_image : ", input_image.size())
    print("feature : ", feature.size())
    print("recon_image : ", recon_image.size())

# def main(_):
#   build_model()


if __name__ == '__main__':
    build_model()
