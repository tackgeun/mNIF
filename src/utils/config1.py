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

@dataclass
class SimpleNeRFConfig:
    epoch_for_full_rendering: int = 0
    subsampled_views: int = 16
    subsampled_pixels: int = 512
    num_samples_per_ray: int = 31
    near: float = 0.8
    far: float = 1.8
    randomized: bool = False
    prob_mask_sampling: float = 0.0
    rgb_activation: str = 'sigmoid'
    density_activation: str = 'relu'
    lindisp: bool = False
    white_bkgd: bool = True
    chuncking_unit: int = -1
    functa_rendering: bool = False
    rendering_type: str = 'baseline'
    debug: bool = False
    resolution: int = 0
    H: int = 0
    W: int = 0


@dataclass
class DataConfig:
    dataset: Optional[str] = None
    resolution: int = 256
    num_subsampling: int = 0
    pred_type: str = 'image'
    repeat_sampling: bool = False
    hparams_nerf: Optional[SimpleNeRFConfig] = SimpleNeRFConfig()


@dataclass
class Stage1EncoderHparams:
    #type: str = "ViT-B/32"
    type: str = "resnet50-conv"
    num_embed_conv: int = 1
    dim_embed_conv: int = 0
    num_groups: int = 16
    type_init_norm: str = ''
    num_flatten_linear: int = 1
    dim_flatten_linear: int = 0
    num_residual_embed: int = 0
    dim_residual_embed: int = 0


@dataclass
class Stage1GeneratorHparams:
    type: str = 'CIPSskip'
    size: int = 256 # is this resolution
    hidden_size: int = 512
    n_mlp: int = 8
    style_dim: int = 512
    lr_mlp: float = 0.01
    activation: Optional[str] = None
    channel_multiplier: int = 2


@dataclass
class Stage1HparamsDisc:
    disc_in_channels: int = 3
    disc_start: int = 0
    disc_weight: float = 0.0
    codebook_weight: float = 0.0
    pixelloss_weight: float = 1.0
    perceptual_weight: float = 0.0


@dataclass
class Stage1Config:
    hparams_enc: Stage1EncoderHparams = Stage1EncoderHparams() 
    hparams_dec: Stage1GeneratorHparams = Stage1GeneratorHparams()
    hparams_disc: Optional[Stage1HparamsDisc] = Stage1HparamsDisc()    


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
    opt_type: str = 'adam'
    betas: Optional[Tuple[float]] = None
    base_lr: float = 1e-4
    g_ratio: float = 1.0
    d_ratio: float = 1.0
    use_amp: bool = True
    grad_clip_norm: Optional[float] = 1.0
    max_steps: Optional[int] = None
    steps_per_epoch: Optional[int] = None
    warmup_config: WarmupConfig = WarmupConfig()


@dataclass
class ExpConfig:
    epochs: int = 100
    save_ckpt_freq: int = 2
    test_freq: int = 1
    img_logging_freq: int = 5000
    fp16_grad_comp: bool = False


@dataclass
class DefaultConfig:
    dataset: DataConfig = DataConfig()
    stage1: Stage1Config = Stage1Config()
    optimizer: OptConfig = OptConfig()
    experiment: ExpConfig = ExpConfig()


@dataclass
class LatentModulatedSIRENConfig:
    width: int = 256
    depth: int = 5
    out_channels: int = 3
    latent_dim: int = 128
    latent_vector_type: str = 'instance' 
    layer_sizes: Tuple[int, ...] = ()
    w0: float = 30.
    modulate_scale: bool = False
    modulate_shift: bool = True
    latent_init_scale: float = 0.01
    use_meta_sgd: bool = True
    meta_sgd_init_range: Tuple[float, float] = (0.005, 0.1)
    meta_sgd_clip_range: Tuple[float, float] = (0., 1.)


@dataclass
class LatentMixtureINRConfig:
    width: int = 64
    depth: int = 4
    in_channels: int = 2
    out_channels: int = 3
    image_resolution: int = 64
    w0: float = 30.
    k_mixtures: int = 64
    mixture_type: str = 'layerwise'
    embedding_type: str = 'none'
    use_latent_embedding: bool = True
    std_latent: float = 0.0
    #normalize_mixture: bool = False
    latent_channels: int = 256
    latent_init_scale: Tuple[float, float] = (0.95, 1.05)
    use_meta_sgd: bool = True
    meta_sgd_init_range: Tuple[float, float] = (0.005, 0.1)
    meta_sgd_clip_range: Tuple[float, float] = (0., 1.)
    #use_residual_param: bool = False
    #type_lipschitz: str = 'none'
    #lipschitz_const: float = 1.0
    init_path: str = ''
    pred_type: str = 'none'
    outermost_linear: Optional[bool] = None


@dataclass
class MetaOptConfig:
    use_amp: bool = False
    optim_outer: str = 'adamw'
    betas: Optional[Tuple[float]] = None
    lr_outer: float = 3e-6
    lr_inner: float = 1e-2
    num_steps: int = 3
    weight_decay_outer: float = 0.0
    weight_decay_inner: float = 0.0
    sparsity_inner: float = 0.0
    sparsity_outer: float = 0.0

    clip_grad: bool = False
    grad_clip_norm: float = 4.0

    # num_epochs: int = 50
    # save_freq: int = 1
    double_precision: bool = False
    max_steps: Optional[int] = None
    use_lr_scheduler: bool = True
    min_lr_outer: float = 0.0
    first_order: bool = False

    steps_per_epoch: Optional[int] = None


@dataclass
class DefaultMetaINR:
    dataset: DataConfig = DataConfig()
    model_type: str = 'mixtureinr'
    hparams_inr: LatentMixtureINRConfig = LatentMixtureINRConfig()
    optimizer: MetaOptConfig = MetaOptConfig()
    experiment: ExpConfig = ExpConfig()


def update_config(cfg_base, cfg_new):
    cfg_update = OmegaConf.merge(cfg_base, cfg_new)
    return cfg_update


def build_config(args):
    if args.stage1_type == 'asym-diff':
        cfg_base = OmegaConf.structured(DefaultConfig)
    elif args.stage1_type == 'parammix':
        cfg_base = OmegaConf.structured(DefaultMetaINR)

    if args.eval:
        cfg_new = OmegaConf.load(os.path.join(args.result_path, 'config.yaml'))
        cfg_update = update_config(cfg_base, cfg_new)
        result_path = args.result_path
    else:
        cfg_new = OmegaConf.load(args.config_path)
        cfg_update = update_config(cfg_base, cfg_new)
        now = datetime.now().strftime('%d%m%Y_%H%M%S')
        result_path = os.path.join(args.result_path,
                                   os.path.basename(args.config_path).split('.')[0],
                                   now)
    return cfg_update, result_path