import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConvBlock(nn.Module):

    def __init__(self, input_channel, output_channel, layer_number):
        super().__init__()
        self.convs = nn.Sequential()
        self.convs.append(nn.Conv2d(in_channels=input_channel, out_channels=output_channel, kernel_size=(3, 3), stride=(1, 1), padding=1))
        for i in range(1, layer_number):
            self.convs.append(nn.Conv2d(in_channels=output_channel, out_channels=output_channel, kernel_size=(3, 3), stride=(1, 1), padding=1))
        self.projection = nn.Conv2d(in_channels=output_channel, out_channels=1, kernel_size=(1, 1), stride=(1, 1), padding=0)

    def __call__(self, x, down_sampling=False):
        h = x
        if down_sampling:
            h = F.max_pool2d(h, kernel_size=(2, 2), stride=(2, 2))
        for conv in self.convs:
            h = conv(h)
            h = F.relu(h)
        return h, self.projection(h)


class ControlNetHED(nn.Module):

    def __init__(self):
        super().__init__()
        self.norm = nn.Parameter(torch.zeros(size=(1, 3, 1, 1)))
        self.block1 = DoubleConvBlock(input_channel=3, output_channel=64, layer_number=2)
        self.block2 = DoubleConvBlock(input_channel=64, output_channel=128, layer_number=2)
        self.block3 = DoubleConvBlock(input_channel=128, output_channel=256, layer_number=3)
        self.block4 = DoubleConvBlock(input_channel=256, output_channel=512, layer_number=3)
        self.block5 = DoubleConvBlock(input_channel=512, output_channel=512, layer_number=3)

    def __call__(self, x):
        h = x - self.norm
        h, projection1 = self.block1(h)
        h, projection2 = self.block2(h, down_sampling=True)
        h, projection3 = self.block3(h, down_sampling=True)
        h, projection4 = self.block4(h, down_sampling=True)
        h, projection5 = self.block5(h, down_sampling=True)
        return projection1, projection2, projection3, projection4, projection5


