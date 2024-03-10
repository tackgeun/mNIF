# ------------------------------------------------------------------------------------
# Minimal DALL-E
# Copyright (c) 2021 KakaoBrain. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------

import os
from datetime import datetime
from typing import Optional, List, Tuple
from dataclasses import dataclass, field
from omegaconf import OmegaConf
from .config1 import LatentMixtureINRConfig, SimpleNeRFConfig

@dataclass
class DataConfig:
    dataset: Optional[str] = None
    image_resolution: int = 64
    data_dimension: int = 2307
    hparams_nerf: Optional[SimpleNeRFConfig] = SimpleNeRFConfig()

@dataclass
class INRParams:
    model_type: str = 'sine'
    sparse_type: str = 'none'
    hidden_features: int = 32
    num_hidden_layers: int = 2
    image_resolution: int = 64

@dataclass
class DiffusionParams:
  steps: int = 1000
  learn_sigma: bool = False
  sigma_small: bool = False
  noise_schedule: str = 'linear'
  use_kl: bool = False
  predict_xstart: bool = False
  rescale_learned_sigmas: bool = False
  timestep_respacing: str = '1000'


@dataclass
class MLPSkipNetParams:
    net_type: str = 'skip'
    num_layers: int = 20
    skip_layers: List[int] = field(default_factory=lambda: [i for i in range(1, 20)])
    activation: str = 'silu'
    num_hid_channels: int = 2048
    num_hid_groups: int = 0
    use_first_hid_layer: bool = False
    use_norm: bool = True # use LayerNorm
    norm_type: str = 'layernorm'
    condition_bias: int = 1
    num_channels: int = 2307
    num_emb_channels: int = 0
    num_out_channels: int = 0 # determined by learn_sigma in diffusion params.
    num_time_emb_channels: int = 64
    dropout: float = 0.0
    last_act: Optional[str] = None
    num_time_layers: int = 2
    time_last_act: bool = False

    use_cond_index: bool = False # use index condition
    num_indices: int = 27000 # for CelebA
    type_cond_index: str = 'concat'

    # use_parallel_input: bool = False
    # parallel_input: Optional[List[int]] = field(default_factory=lambda: [96, 2211])
    # parallel_embed: Optional[List[int]] = field(default_factory=lambda: [2048, 2048])

# @dataclass
# class LatentMixtureINRConfig:
#     width: int = 64
#     depth: int = 4
#     out_channels: int = 3
#     image_resolution: int = 64
#     w0: float = 30.
#     k_mixtures: int = 64
#     mixture_type: str = 'layerwise'
#     embedding_type: str = 'none'
#     use_latent_embedding: bool = True
#     normalize_mixture: bool = False
#     latent_channels: int = 256
#     latent_init_scale: Tuple[float, float] = (0.95, 1.05)
#     use_meta_sgd: bool = True
#     meta_sgd_init_range: Tuple[float, float] = (0.005, 0.1)
#     meta_sgd_clip_range: Tuple[float, float] = (0., 1.)
#     init_path: str = ''

@dataclass
class Stage2Config:
    hparams_diffusion: DiffusionParams = DiffusionParams() 
    hparams_model: MLPSkipNetParams = MLPSkipNetParams()
    hparams_inr: INRParams = INRParams()
    hparams_metainr: LatentMixtureINRConfig = LatentMixtureINRConfig()
    feat_std_scale: float = 4.0
    feat_type: str = 'weight'
    image_std_scale: float = 1.0
    crop_feature: bool = False
    crop_dim: int = 0

@dataclass
class WarmupConfig:
    multiplier: float = 1.0
    warmup_epoch: float = 0.0
    buffer_epoch: float = 0.0
    min_lr: float = 0.0
    mode: str = 'fix'
    start_from_zero: bool = True

@dataclass
class OptConfig:
    opt_type: str = 'adamw'
    betas: Optional[Tuple[float]] = field(default_factory=lambda: [0.9, 0.999])
    base_lr: float = 1e-4
    weight_decay: float = 0.01
    use_amp: bool = False
    use_ema: bool = False
    grad_clip_norm: Optional[float] = None
    max_steps: Optional[int] = None
    steps_per_epoch: Optional[int] = None
    sched_type: str = 'cosine'
    warmup: WarmupConfig = WarmupConfig()


@dataclass
class ExpConfig:
    epochs: int = 100
    save_ckpt_freq: int = 2
    test_freq: int = 1
    img_logging_freq: int = 5000
    fp16_grad_comp: bool = False


@dataclass
class DefaultConfig2:
    dataset: DataConfig = DataConfig()
    stage2: Stage2Config = Stage2Config()
    optimizer: OptConfig = OptConfig()
    experiment: ExpConfig = ExpConfig()


def update_config(cfg_base, cfg_new):
    cfg_update = OmegaConf.merge(cfg_base, cfg_new)
    return cfg_update

def build_config(args):
    cfg_base = OmegaConf.structured(DefaultConfig2)
    if args.eval:
        cfg_new = OmegaConf.load(os.path.join(args.result_path, 'config.yaml'))
        cfg_update = update_config(cfg_base, cfg_new)
        result_path = args.result_path
    else:
        cfg_new = OmegaConf.load(args.config_path)
        cfg_update = update_config(cfg_base, cfg_new)
        #now = datetime.now().strftime('%d%m%Y_%H%M%S')
        result_path = os.path.join(args.result_path,
                                   os.path.basename(args.config_path).split('.')[0])
    return cfg_update, result_path
