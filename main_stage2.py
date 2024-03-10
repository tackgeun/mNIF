import os
import argparse

from torch.utils.data.dataloader import DataLoader
import torchvision

import pytorch_lightning as pl
from torch.distributed.algorithms.ddp_comm_hooks import default_hooks
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from src.datasets import CelebAHQ, ShapeNet, SRNDatasets, INRWeightWrapper
from src.utils.logger import LatentDDPMLogger
from src.models import build_model_stage2
from src.utils.config2 import build_config
from src.utils.utils import logging_model_size


parser = argparse.ArgumentParser()

parser.add_argument('-c', '--config-path', type=str, default=None, required=True)
parser.add_argument('-r', '--result-path', type=str, default=None, required=True)
parser.add_argument('--checkpoint_path', type=str, default='')
parser.add_argument('--stage1_epoch', type=int, default=-1)
parser.add_argument('--eval', default=False)

parser.add_argument('--world_size', default=-1, type=int, help='number of nodes for distributed training')
parser.add_argument('--local_rank', default=-1, type=int, help='local rank for distributed training')
parser.add_argument('--node_rank', default=-1, type=int, help='node rank for distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')
parser.add_argument('--n-nodes', type=int, default=1)
parser.add_argument('--n-gpus', type=int, default=1)
parser.add_argument('--local_batch_size', type=int, default=64)
parser.add_argument('--valid_batch_size', type=int, default=64)
parser.add_argument('--total_batch_size', type=int, default=64)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--dataset_root', type=str, default='datasets')
parser.add_argument('--reduce_sample', type=int, default=0)
parser.add_argument('--context_tag', type=str, default='set')

args = parser.parse_args()

def setup_callbacks(config, result_path):
    # Setup callbacks
    ckpt_path = os.path.join(result_path, 'ckpt')
    log_path = os.path.join(result_path, 'log')

    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_path,
        filename=config.dataset.dataset+"-lddim{epoch:02d}",
        every_n_epochs=config.experiment.save_ckpt_freq,
        save_top_k=-1,
        save_weights_only=True,
        save_last=False # do not save the last
    )
    logger_tb = TensorBoardLogger(log_path, name="latent-ddpm")
    logger_cu = LatentDDPMLogger(config, result_path)
    return checkpoint_callback, logger_tb, logger_cu


if __name__ == '__main__':
    pl.seed_everything(args.seed)

    # Setup
    config, result_path = build_config(args)
    ckpt_callback, logger_tb, logger_cu = setup_callbacks(config, result_path)

    if len(args.dataset_root) > 0:
        root_path = args.dataset_root
    else:
        root_path = None

    # Build data modules
    dname = config.dataset.dataset.lower()

    if 'celeba' in dname:
        data_res = config.dataset.image_resolution
        downsampled = False
        tf_dataset = 'tf' in dname.lower()
        train_dataset = CelebAHQ(split='train', downsampled=downsampled, resolution=data_res, dataset_root=root_path, tf_dataset=tf_dataset)
        valid_dataset = CelebAHQ(split='test', downsampled=downsampled, resolution=data_res, dataset_root=root_path, tf_dataset=tf_dataset)
        resampling = 'bicubic'

    elif 'cifar10' in dname:
        data_res = config.dataset.image_resolution
        train_dataset = torchvision.datasets.CIFAR10(root=root_path, train=True, download=True)
        valid_dataset = torchvision.datasets.CIFAR10(root=root_path, train=False, download=True)
    elif 'shapenet' in dname:
        train_dataset = ShapeNet(split='train', sampling=4096, dataset_root=root_path)
        valid_dataset = ShapeNet(split='test', sampling=4096, dataset_root=root_path)
    elif 'srncars' in dname:
        train_dataset = None
        valid_dataset = None
        
    else:
        raise ValueError()

    checkpoint_path = args.checkpoint_path
    print(checkpoint_path)
    input_res = config.stage2.hparams_inr.image_resolution
    train_dataset = INRWeightWrapper(train_dataset,
                                     sidelength=input_res,
                                     checkpoint_path=checkpoint_path,
                                     checkpoint_step=args.stage1_epoch,
                                     reduce_sample=args.reduce_sample,
                                     feed_type=config.stage2.feat_type,
                                     context_tag=args.context_tag,
                                     istuple='cifar10' in dname)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.local_batch_size, pin_memory=True, num_workers=8)

    # # Ignore validation dataset because fixed training epoch.
    # valid_checkpoint_path = checkpoint_path + '-test'
    # valid_dataset = INRWeightWrapper(valid_dataset,
    #                                  sidelength=input_res,
    #                                  checkpoint_path=valid_checkpoint_path,
    #                                  checkpoint_step=args.ckpt_step,
    #                                  reduce_sample=args.reduce_sample,
    #                                  feed_type=config.stage2.feat_type)
    # valid_dataloader = DataLoader(valid_dataset, shuffle=False, batch_size=args.valid_batch_size, pin_memory=True, num_workers=8)
    # valid_dataloader = DataLoader(train_dataset, shuffle=False, batch_size=args.local_batch_size, pin_memory=True, num_workers=4)
    valid_dataloader = None

    # Calculate how many batches are accumulated
    total_gpus = args.n_gpus * args.n_nodes
    assert args.total_batch_size % total_gpus == 0
    grad_accm_steps = args.total_batch_size // (args.local_batch_size * total_gpus)
    config.optimizer.max_steps = len(train_dataset) // args.total_batch_size * config.experiment.epochs
    config.optimizer.steps_per_epoch = len(train_dataset) // args.total_batch_size
    config.stage2.hparams_metainr.init_path = os.path.join(args.checkpoint_path, 'metainits', f'epoch{args.stage1_epoch}.pth')

    # Build a model
    model = build_model_stage2(cfg_stage2=config.stage2, cfg_opt=config.optimizer, affine=train_dataset.affine)
    logging_model_size(model, logger_cu._logger)

    # Build a trainer
    trainer = pl.Trainer(max_epochs=config.experiment.epochs,
                         accumulate_grad_batches=grad_accm_steps,
                         gradient_clip_val=config.optimizer.grad_clip_norm,
                         precision=16 if config.optimizer.use_amp else 32,
                         callbacks=[ckpt_callback, logger_cu],
                         accelerator="gpu",
                         num_nodes=args.n_nodes,
                         devices=args.n_gpus,
                         strategy=DDPPlugin(ddp_comm_hook=default_hooks.fp16_compress_hook) if
                         config.experiment.fp16_grad_comp else "ddp",
                         logger=logger_tb,
                         log_every_n_steps=10)

    trainer.fit(model, train_dataloader, valid_dataloader)
