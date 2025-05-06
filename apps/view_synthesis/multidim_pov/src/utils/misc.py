import random
import torch
import numpy as np


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_EXTENSIONS = ('.safetensors','.ckpt','.pt','.pth','.bin')


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


