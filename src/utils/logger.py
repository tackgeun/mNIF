# ------------------------------------------------------------------------------------
# HQ-Transformer
# Copyright (c) 2022 KakaoBrain. All Rights Reserved.
# Licensed under the MIT License [see LICENSE for details]
# ------------------------------------------------------------------------------------

import os
import logging
import math

import torch
import torchvision
import torch.distributed as dist

from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.distributed import rank_zero_only
from omegaconf import OmegaConf

import pdb

class DefaultLogger(Callback):
    def __init__(self, config, result_path, is_eval=False):
        super().__init__()

        self._config = config
        self._result_path = result_path
        self._logger = self._init_logger(is_eval=is_eval)

    @rank_zero_only
    def _init_logger(self, is_eval=False):
        self.save_config()
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        # create console handler and set level to info
        ch = logging.FileHandler(os.path.join(self._result_path, 'eval.log' if is_eval else 'train.log'))
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter(
            fmt="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S")
        )
        # add ch to logger
        logger.addHandler(ch)
        logger.info(f"Logs will be recorded in {self._result_path}...")
        return logger

    @rank_zero_only
    def save_config(self):
        if not os.path.exists(self._result_path):
            os.makedirs(self._result_path)
        with open(os.path.join(self._result_path, 'config.yaml'), 'w') as fp:
            OmegaConf.save(config=self._config, f=fp)

    @rank_zero_only
    def log_metrics(self, trainer, split='valid'):
        metrics = []
        for k, v in trainer.callback_metrics.items():
            if split == 'valid':
                if k.startswith('valid'):
                    k = k.split('/')[-1].strip()
                    metrics.append((k, v))
            elif split == 'test':
                if k.startswith('test'):
                    k = k.split('/')[-1].strip()
                    metrics.append((k, v))                    
            else:
                if k.startswith('train') and k.endswith('epoch'):
                    k = k.split('/')[-1].strip()[:-6]
                    metrics.append((k, v))
        metrics = sorted(metrics, key=lambda x: x[0])
        line = ','.join([f"  {metric[0]}:{metric[1].item():.4f}" for metric in metrics])
        line = f'EPOCH:{trainer.current_epoch}, {split.upper()}\t' + line
        self._logger.info(line)

    def on_train_epoch_end(self, trainer, pl_module):
        if dist.get_rank() == 0:
            self.log_metrics(trainer, split='train')

    def on_validation_epoch_end(self, trainer, pl_module):
        if dist.get_rank() == 0:
            self.log_metrics(trainer, split='valid')

class AsymmetricAutoEncoderLogger(DefaultLogger):
    def __init__(self, config, result_path, is_eval=False):
        super().__init__(config, result_path, is_eval)

    @rank_zero_only
    def log_img(self, pl_module, batch, global_step, split="train"):
        with torch.no_grad():
            images, _ = batch
            images = images.cpu()

            recons = recons.cpu()

            grid_org = (torchvision.utils.make_grid(images, nrow=4) + 1.0) / 2.0
            grid_rec = (torchvision.utils.make_grid(recons, nrow=4) + 1.0) / 2.0
            grid_rec = torch. clip(grid_rec, min=0, max=1)

            pl_module.logger.experiment.add_image(f"images_org/{split}", grid_org, global_step=global_step)            

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if hasattr(pl_module.discriminator, 'perceptual_loss'):
            pl_module.discriminator.perceptual_loss.eval()
        pl_module.generator.encoder.eval()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if pl_module._num_opt_steps % self._config.experiment.img_logging_freq == 0:
            pl_module.eval()
            self.log_img(pl_module, batch, global_step=pl_module._num_opt_steps, split="train")
            pl_module.train()

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if batch_idx == 0:
            pl_module.eval()
            self.log_img(pl_module, batch, global_step=trainer.current_epoch, split="valid")


class MetaINRLogger(DefaultLogger):
    def __init__(self, config, result_path, is_eval=False):
        super().__init__(config, result_path, is_eval)

class LatentDDPMLogger(DefaultLogger):
    def __init__(self, config, result_path, is_eval=False):
        super().__init__(config, result_path, is_eval)

    @rank_zero_only
    def log_img(self, pl_module, batch, current_epoch, split="train"):
        with torch.no_grad():
            inputs, gts = batch
            images = gts['img']
            L = int(math.sqrt(images.size(1)))
            images = images.view(-1, L, L, 3).permute(0,3,1,2).cpu()
            pl_module.logger.experiment.add_image(f"images_org/{split}", grid_org, global_step=current_epoch)

    @rank_zero_only
    def log_sample(self, pl_module, current_epoch, split="train"):
        with torch.no_grad():
            if pl_module.feat_type == 'weight':
                samples = pl_module.sample(16, pl_module._diffusion_kwargs.timestep_respacing, resolution=32)
                samples = torch.clamp((torchvision.utils.make_grid(samples, nrow=4) + 1.0) / 2.0, 0, 1)
                pl_module.logger.experiment.add_image(f"samples/32px/{split}", samples, global_step=current_epoch)

                samples = pl_module.sample(16, pl_module._diffusion_kwargs.timestep_respacing, resolution=64)
                samples = torch.clamp((torchvision.utils.make_grid(samples, nrow=4) + 1.0) / 2.0, 0, 1)
                pl_module.logger.experiment.add_image(f"samples/64px/{split}", samples, global_step=current_epoch) 

                samples = pl_module.sample(4, pl_module._diffusion_kwargs.timestep_respacing, resolution=128)
                samples = torch.clamp((torchvision.utils.make_grid(samples, nrow=2) + 1.0) / 2.0, 0, 1)
                pl_module.logger.experiment.add_image(f"samples/128px/{split}", samples, global_step=current_epoch) 
            else:
                samples = pl_module.sample(16, pl_module._diffusion_kwargs.timestep_respacing)
                samples = torch.clamp((torchvision.utils.make_grid(samples, nrow=4) + 1.0) / 2.0, 0, 1)
                pl_module.logger.experiment.add_image(f"samples/{split}", samples, global_step=current_epoch) 

    # def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
    #     if batch_idx == 0:
    #         self.log_img(pl_module, batch, current_epoch=trainer.current_epoch, split="train")

    def on_train_epoch_end(self, trainer, pl_module):
        self.log_metrics(trainer, split='train')
        self.log_sample(pl_module, current_epoch=trainer.current_epoch, split="train")

    def on_validation_epoch_end(self, trainer, pl_module):
        self.log_metrics(trainer, split='valid')
        