from .utils import sam_models, det_models

all_models = list(sam_models.keys())
default_model = 'vit_b'

det_models = list(det_models.keys())
default_det = 'florence'
