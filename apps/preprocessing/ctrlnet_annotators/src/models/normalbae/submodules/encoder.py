import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()

        basemodel_name = 'tf_efficientnet_b5_ap'
        print(f'Loading base model ({basemodel_name}) ...')
        repo_path = os.path.join(os.path.dirname(__file__), 'efficientnet')
        basemodel = torch.hub.load(repo_path, basemodel_name, pretrained=False, source='local')

        # Remove last layer
        print('Removing last 2 layers (global_pool & classifier) ...')
        basemodel.global_pool = nn.Identity()
        basemodel.classifier = nn.Identity()

        self.original_model = basemodel

    def forward(self, x):
        features = [x]
        for k, v in self.original_model._modules.items():
            if (k == 'blocks'):
                for ki, vi in v._modules.items():
                    features.append(vi(features[-1]))
            else:
                features.append(v(features[-1]))
        return features


