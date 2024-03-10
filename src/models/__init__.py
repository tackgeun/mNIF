# ------------------------------------------------------------------------------------
# HQ-Transformer
# Copyright (c) 2022 KakaoBrain. All Rights Reserved.
# Licensed under the MIT License [see LICENSE for details]
# ------------------------------------------------------------------------------------

import os
import torch
import torch.nn as nn
import pytorch_lightning as pl

from typing import Optional, Tuple
from omegaconf import OmegaConf
from torch.nn import functional as F
from torch.cuda.amp import autocast

from src.optimizers.scheduler import build_scheduler

# stage 1 model builder
def build_model_stage1(stage1_type, config):
    if stage1_type == 'asym-diff':
        from .stage1.asymautoenc import AsymmetricAutoEncoder
        return AsymmetricAutoEncoder(hparams=config.stage1,
                                     hparams_opt=config.optimizer)
    elif stage1_type == 'parammix':
        from .stage1.metainr import MetaINR
        return MetaINR(hparams=config)

def build_model_stage2(cfg_stage2, cfg_opt, affine):
    from .stage2.latentddim import LatentDDIM
    return LatentDDIM(hparams=cfg_stage2, hparams_opt=cfg_opt, affine=affine)



