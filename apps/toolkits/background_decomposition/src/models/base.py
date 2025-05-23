import os
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image
from PIL.Image import Image as ImageClass

import numpy as np
import onnxruntime as ort


CHECKPOINT_ROOT = os.environ.get('SHIVA_CKPT_ROOT', None)
if CHECKPOINT_ROOT is not None:
    MODEL_DIR = Path(CHECKPOINT_ROOT) / 'rembg'
else:
    MODEL_DIR = Path(__file__).parents[5] / 'checkpoints/rembg'
MODEL_DIR = str(MODEL_DIR)


class BaseModel:
    """
    This is a base class for managing a session with a machine learning model.
    """
    def __init__(
        self,
        model_name: str,
        sess_opts: ort.SessionOptions,
        providers=None,
        *args,
        **kwargs
    ):
        """
        Initialize an instance of the BaseModel class.
        """
        self.model_name = model_name
        self.providers = []

        _providers = ort.get_available_providers()
        if providers:
            for provider in providers:
                if provider in _providers:
                    self.providers.append(provider)
        else:
            self.providers.extend(_providers)

        self.inner_session = ort.InferenceSession(
            str(self.__class__.download_models(*args, **kwargs)),
            providers=self.providers,
            sess_options=sess_opts,
        )

    def normalize(
        self,
        img: ImageClass,
        mean: Tuple[float, float, float],
        std: Tuple[float, float, float],
        size: Tuple[int, int],
        *args,
        **kwargs
    ) -> Dict[str, np.ndarray]:

        im = img.convert("RGB").resize(size, Image.Resampling.LANCZOS)

        im_ary = np.array(im)
        im_ary = im_ary / np.max(im_ary)

        tmpImg = np.zeros((im_ary.shape[0], im_ary.shape[1], 3))
        tmpImg[:, :, 0] = (im_ary[:, :, 0] - mean[0]) / std[0]
        tmpImg[:, :, 1] = (im_ary[:, :, 1] - mean[1]) / std[1]
        tmpImg[:, :, 2] = (im_ary[:, :, 2] - mean[2]) / std[2]

        tmpImg = tmpImg.transpose((2, 0, 1))

        return {
            self.inner_session.get_inputs()[0].name: np.expand_dims(tmpImg, 0).astype(np.float32)
        }

    def predict(self, img: ImageClass, *args, **kwargs) -> List[ImageClass]:
        raise NotImplementedError

    @classmethod
    def checksum_disabled(cls, *args, **kwargs):
        return os.getenv("MODEL_CHECKSUM_DISABLED", None) is not None

    @classmethod
    def ckpt_dir(cls, *args, **kwargs):
        try:
            from src.config.paths import rembg_dir as ckpt_diroot
        except:
            ckpt_diroot = MODEL_DIR
            # ckpt_diroot = os.path.expanduser(
            #     os.getenv("U2NET_HOME", os.path.join(os.getenv("XDG_DATA_HOME", "~"), ".u2net"))
            # )
        if not os.path.isdir(ckpt_diroot):
            os.makedirs(ckpt_diroot)
        return ckpt_diroot

    @classmethod
    def download_models(cls, *args, **kwargs):
        raise NotImplementedError

    @classmethod
    def name(cls, *args, **kwargs):
        raise NotImplementedError
