import yaml
import argparse

import torch
from torch.utils.data import DataLoader

torch.set_float32_matmul_precision('medium')

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from .src.dataset import MP3Ddataset, Scannetdataset
from .src.wrappers import DepthGenerator, PanoGenerator, PanoOutpaintor


def parse_args():
    # init a costum parser which will be added into pl.Trainer parser
    # check documentation: https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html#trainer-flags
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--cfg_path', type=str, help='config path')
    parser.add_argument('--ckpt_path', type=str, default=None, help='pretrained checkpoint path, helpful for using a pre-trained coarse-only LoFTR')
    parser.add_argument('--exp_name', type=str, default='default_exp_name')
    parser.add_argument('--mode', type=str, default='val')

    parser.add_argument('--batch_size', type=int, default=4, help='batch_size per gpu')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--eval_on_train', action='store_true')

    parser = pl.Trainer.add_argparse_args(parser)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    config = yaml.load(open(args.cfg_path, 'rb'), Loader=yaml.SafeLoader)
    config['train']['max_epochs'] = args.max_epochs

    mode = 'train' if args.eval_on_train else 'val'

    if config['dataset']['name'] == 'mp3d':
        dataset = MP3Ddataset(config['dataset'], mode=mode)
    elif config['dataset']['name'] == 'scannet':
        dataset = Scannetdataset(config['dataset'], mode=mode)

    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, drop_last=False)

    if config['model']['model_type'] == 'pano_generation':
        model = PanoGenerator(config)
    elif config['model']['model_type'] == 'pano_outpainting':
        model = PanoOutpaintor(config)
    elif config['model']['model_type'] == 'depth':
        model = DepthGenerator(config)

    if args.ckpt_path is not None:
        model.load_state_dict(
            torch.load(args.ckpt_path, map_location='cpu')['state_dict'], strict=True)

    logger = TensorBoardLogger(save_dir='logs/tb_logs', name=args.exp_name, default_hp_metric=False)
    trainer = pl.Trainer.from_argparse_args(args, logger=logger)

    if args.mode == 'test':
        trainer.test(model, data_loader)
    else:
        trainer.validate(model, data_loader)
