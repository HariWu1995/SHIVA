from types import SimpleNamespace

NormalConfig = SimpleNamespace()
NormalConfig.mode = 'client'
NormalConfig.architecture = 'BN'
NormalConfig.pretrained = 'scannet'
NormalConfig.sampling_ratio = 0.4
NormalConfig.importance_ratio = 0.7

from .NNET import NNET as NormalNet
