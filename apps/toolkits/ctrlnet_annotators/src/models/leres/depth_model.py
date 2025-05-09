import torch
import torch.nn as nn

from . import network
from .utils import get_func


class RelDepthModel(nn.Module):

    def __init__(self, backbone: str = 'resnet50'):
        super(RelDepthModel, self).__init__()
        if backbone == 'resnet50':
            encoder = 'resnet50_stride32'
        elif backbone == 'resnext101':
            encoder = 'resnext101_stride32x8d'
        self.depth_model = DepthModel(encoder)

    def inference(self, rgb):
        with torch.no_grad():
            input = rgb.to(self.depth_model.device)
            depth = self.depth_model(input)
            return depth 


class DepthModel(nn.Module):

    def __init__(self, encoder):
        super(DepthModel, self).__init__()
        backbone = network.__name__.split('.')[-1] + '.' + encoder
        self.encoder_modules = get_func(backbone)()
        self.decoder_modules = network.Decoder()

    def forward(self, x):
        out_encoded = self.encoder_modules(x)
        out_decoded = self.decoder_modules(out_encoded)
        return out_decoded


