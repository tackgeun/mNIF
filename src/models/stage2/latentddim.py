import copy
import functools
from inspect import isfunction
from contextlib import contextmanager
import os 
import math


import torch
import pytorch_lightning as pl

from src.models.stage1.siren.modules import get_model as get_inr_model
from src.datasets import get_mgrid, get_mgrid_voxel
from .diffusion import gaussian_diffusion as gd
from .diffusion.resample import LossAwareSampler, UniformSampler
from .diffusion.respace import (
    SpacedDiffusion,
    space_timesteps,
)

from .diffae.latentnet import MLPSkipNet, FunctaMLP, ParallelMLP, MLPEmbedNet
from src.optimizers.scheduler import build_scheduler
from src.models.stage2.ema import LitEma
from src.models.stage1.siren import meta_modules

import pdb

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def create_gaussian_diffusion(
    steps,
    learn_sigma,
    sigma_small,
    noise_schedule,
    use_kl,
    predict_xstart,
    rescale_learned_sigmas,
    timestep_respacing,
):
    betas = gd.get_named_beta_schedule(noise_schedule, steps)
    if use_kl:
        loss_type = gd.LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type = gd.LossType.RESCALED_MSE
    else:
        loss_type = gd.LossType.MSE
    if not timestep_respacing:
        timestep_respacing = [steps]

    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
    )

class LatentDDIM(pl.LightningModule):
    def __init__(
        self,
        hparams,
        hparams_opt,
        affine
    ):
        super().__init__()
        self.feat_type = hparams.feat_type
        self.crop_feature = hparams.crop_feature
        self.crop_dim = hparams.crop_dim
        if hparams.feat_type in ['weight', 'dict', 'context', 'context-set']:
            self.mean, self.std = affine # feature statistics
            if hparams.feat_type in ['weight', 'context', 'context-set']:
                self.std *= hparams.feat_std_scale
        
        if hparams.feat_type in ['image']:
            self.image_std_scale = hparams.image_std_scale

        self._diffusion_kwargs = hparams.hparams_diffusion
        self.diffusion = create_gaussian_diffusion(**self._diffusion_kwargs)
        self.schedule_sampler = UniformSampler(self.diffusion)

        if hparams.feat_type in ['context', 'context-set']:
            self.INRParams = hparams.hparams_metainr
            # try:
            #     self.decoder = meta_modules.LinearMixtureSIREN(**self.INRParams)
            #     if os.path.exists(self.INRParams.init_path):                
            #         ckpt_inr = torch.load(self.INRParams.init_path)
            #         self.decoder.load_state_dict(ckpt_inr)                
            #         print('success to load MetaSIREN model.')
            #     else:
            #         print('fail to load MetaSIREN model.')
            # except:
            #     print('load LinearMixtureINR instead')
            #     self.decoder = meta_modules.LinearMixtureINR(**self.INRParams)
            #     if os.path.exists(self.INRParams.init_path):                
            #         ckpt_inr = torch.load(self.INRParams.init_path)
            #         self.decoder.load_state_dict(ckpt_inr)                
            #         print('success to load MetaINR model.')
            #     else:
            #         print('fail to load MetaINR model.')
            self.decoder = meta_modules.LinearMixtureINR(**self.INRParams)
            if os.path.exists(self.INRParams.init_path):
                ckpt_inr = torch.load(self.INRParams.init_path)
                self.decoder.load_state_dict(ckpt_inr)                
                print('success to load MetaINR model.')
            else:
                print('fail to load MetaINR model.')


            self.input_coords = get_mgrid(self.INRParams.image_resolution, dim=self.INRParams.in_channels)
        else:
            self.INRParams = hparams.hparams_inr
            self.decoder = get_inr_model(**self.INRParams)
            self.input_coords = get_mgrid(self.INRParams.image_resolution, dim=self.INRParams.in_channels)

        self.MLPParams = hparams.hparams_model        
        if self.diffusion.model_var_type in [gd.ModelVarType.LEARNED, gd.ModelVarType.LEARNED_RANGE]:
            self.MLPParams.num_out_channels = self.MLPParams.num_channels * 2
        else:
            self.MLPParams.num_out_channels = self.MLPParams.num_channels
        if self.MLPParams.net_type == 'skip':
            self.model = MLPSkipNet(self.MLPParams)
        elif self.MLPParams.net_type == 'parallel':
            self.model = ParallelMLP(self.decoder.state_dict(), self.MLPParams)
        elif self.MLPParams.net_type == 'embed':
            self.model = MLPEmbedNet(self.decoder.state_dict(), self.MLPParams)
        elif self.MLPParams.net_type in ['functa', 'functa-affine', 'functa-skip']:
            self.model = FunctaMLP(self.MLPParams)

        self.hparams_opt = hparams_opt
        self.use_ema = hparams_opt.use_ema
        if self.use_ema:
            self.model_ema = LitEma(self.model)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")
        

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.model.parameters())
            self.model_ema.copy_to(self.model)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.model.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    @classmethod
    def load_from_checkpoint(cls, config, tokenizer, ckpt_path, strict: bool = True):
        ckpt = torch.load(ckpt_path, map_location="cpu")["state_dict"]

        model = cls(config, tokenizer)
        model.load_state_dict(ckpt, strict=strict)
        return model

    def get_sample_fn(self, timestep_respacing):
        use_ddim = timestep_respacing.startswith(("ddim", "fast"))

        diffusion_kwargs = copy.deepcopy(self._diffusion_kwargs)
        diffusion_kwargs.update(timestep_respacing=timestep_respacing)
        diffusion = create_gaussian_diffusion(**diffusion_kwargs)
        sample_fn = (
            diffusion.ddim_sample_loop_progressive
            if use_ddim
            else diffusion.p_sample_loop_progressive
        )

        return sample_fn

    def sample_latent(
        self,
        batch_size=8,
        timestep_respacing=None,
    ):
        
        if self.MLPParams.use_cond_index:
            cond = {'cond_idx': torch.randint(0, self.MLPParams.num_indices, (batch_size,), device=self.device)}
        else:
            cond = None
        sample_fn = self.get_sample_fn(timestep_respacing)
        sample_outputs = sample_fn(
            self.model,
            (batch_size, self.model.num_channels),
            noise=None,
            device=self.device,
            clip_denoised=False, # clip gradient value within (-1, +1)
            model_kwargs=cond,
        )
        for out in sample_outputs:
            sample = out["sample"]

            if self.crop_feature:
                padded_sample = torch.zeros(sample.size(0), self.std.size(1), device=self.device)
                padded_sample[:, 0:self.crop_dim].copy_(sample)
                sample = padded_sample

            # denormalize
            if self.feat_type in ['weight', 'context', 'context-set'] and self.mean is not None:
                sample =  sample * (self.std.to(self.device) + 1e-8) + self.mean.to(self.device)
            yield sample

    def sample(
        self,
        batch_size = 16,
        timestep_respacing = None,
        resolution = 64,
        output_type = 'image'
    ):

        samples = self.sample_latent(batch_size, timestep_respacing)
        samples = list(samples)

        if output_type == 'image':
            input_coords = get_mgrid(resolution, dim=self.INRParams.in_channels).to(self.device)
        elif output_type == 'voxel':
            input_coords = get_mgrid_voxel(resolution).to(self.device)
        elif output_type == 'scene':
            input_coords = get_mgrid(resolution, dim=self.INRParams.in_channels).to(self.device)

        if self.feat_type in ['weight', 'dict']:
            images = []
            for sample in samples[-1]:

                    # # debugging purpose
                    # if len(images) == 0 and root_path is not None:
                    #     ckpt = torch.load(os.path.join(root_path, 'checkpoints', f'model10000.pth'), map_location=self.device)
                    #     W = []
                    #     for k in ckpt.keys():
                    #         w = ckpt[k]
                    #         W.append(w.view(-1))   
                    #     sample = torch.cat(W, dim=0).contiguous()

                    count = 0
                    
                    # for param in self.decoder.parameters():
                    #     param.copy_(sample[count:count+param.numel()].view(param.size()))
                    #     count += param.numel()

                    # load weights for decoder
                    new_dict = self.decoder.state_dict()
                    for key in new_dict.keys():
                        new_dict[key].copy_(sample[count:count+new_dict[key].numel()].view(new_dict[key].size()))
                        count += new_dict[key].numel()
                    
                    self.decoder.load_state_dict(new_dict)

                    # generate images
                    outputs = self.decoder({'coords': input_coords})
                    image = outputs['model_out'].view(resolution, resolution, 3).permute(2, 0, 1)
                    images.append(image)

        elif self.feat_type == 'image':
            images = samples[-1] # final diffusion step
            in_res = int(math.sqrt(self.MLPParams.num_channels // 3))
            images = images.view(images.size(0), 3, in_res, in_res) #.permute(2, 0, 1)
            images = images * self.image_std_scale

        elif self.feat_type in ['context', 'context-set']:
            if output_type == 'image':
                if self.INRParams.in_channels == 2:
                    images = []
                    for sample in samples[-1]:
                        # generate images
                        outputs = self.decoder({'coords': input_coords}, sample)
                        image = outputs['model_out'].view(resolution, resolution, 3).permute(2, 0, 1)
                        images.append(image)

                    return images

                elif self.INRParams.in_channels == 3 and self.INRParams.out_channels == 1:
                    images = []
                    for batch in range(0, batch_size):
                        images.append(torch.zeros(3, resolution, resolution))
                    return images                        

                elif self.INRParams.in_channels == 3 and self.INRParams.out_channels == 4:
                    images = []
                    for batch in range(0, batch_size):
                        images.append(torch.zeros(3, resolution, resolution))
                    return images                        
    
                else:
                    print('sampling in this type of input, output pair is not defined.')
                    assert(False)

            elif output_type == 'voxel':
                if self.INRParams.in_channels == 3 and self.INRParams.out_channels == 1:
                    voxels = []
                    for sample in samples[-1]:
                        # generate images
                        outputs = self.decoder({'coords': input_coords}, sample)
                        voxel = outputs['model_out'].view(resolution, resolution, resolution).cpu()
                        voxel = (voxel + 1.0)/2.0
                        voxels.append(voxel)

                    return voxels


    def forward(self, x, log_prefix='', *args, **kwargs):
        bsz = x.shape[0]

        t, weights = self.schedule_sampler.sample(bsz, self.device)
        
        compute_losses = functools.partial(
            self.diffusion.training_losses,
            self.model,
            x,
            t,
            model_kwargs=kwargs,
        )
        losses = compute_losses()
        loss = (losses['loss'] * weights).mean()
        loss_dict = {log_prefix + k: (v * weights).mean() for k, v in losses.items()}

        return loss, loss_dict

    def get_input(self, batch):
        model_input, gts = batch
        idx = model_input['idx']

        if self.feat_type in ['weight', 'dict', 'context', 'context-set']:
            weights = gts['weights']

            if self.mean is not None:
                weights = (weights - self.mean.to(self.device)) / (self.std.to(self.device) + 1e-8)

            if self.crop_feature:
                weights = weights[:, 0:self.crop_dim]

            return weights, idx

        elif self.feat_type == 'image':
            images = gts['img']
            images = images.view(images.size(0), -1)
            images = images / self.image_std_scale
        
            return images, idx
            

    def shared_step(self, batch, log_prefix=''):
        x, idx = self.get_input(batch)
        if self.MLPParams.use_cond_index:
            loss, loss_dict = self(x, log_prefix, cond_idx=idx)
        else:
            loss, loss_dict = self(x, log_prefix)
        return loss, loss_dict

    def training_step(self, batch, batch_idx):

        loss, loss_dict = self.shared_step(batch, log_prefix='train/')
        
        self.log_dict(loss_dict, prog_bar=True,
                      logger=True, on_step=True, on_epoch=True)

        self.log("global_step", self.global_step,
                 prog_bar=True, logger=True, on_step=True, on_epoch=False)

        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        _, loss_dict_no_ema = self.shared_step(batch, log_prefix='valid/')
        with self.ema_scope():
            _, loss_dict_ema = self.shared_step(batch, log_prefix='valid/')
            loss_dict_ema = {key + '_ema': loss_dict_ema[key] for key in loss_dict_ema}
        self.log_dict(loss_dict_no_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        self.log_dict(loss_dict_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)


    def configure_optimizers(self):
        assert self.hparams_opt.opt_type == 'adamw'

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()

        for mn, _ in self.model.named_parameters():
            if mn.endswith('bias'):
                # all biases will not be decayed
                no_decay.add(mn)
            elif mn.endswith('weight'):
                # weights will be weight decayed
                decay.add(mn)

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.model.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, \
            "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, \
            "parameters %s were not separated into either decay/no_decay set!" \
            % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": self.hparams_opt.weight_decay
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0
            },
        ]

        opt = torch.optim.AdamW(optim_groups,
                                lr=self.hparams_opt.base_lr,
                                betas=self.hparams_opt.betas,
                                weight_decay=self.hparams_opt.weight_decay)

        sched = build_scheduler(opt,
                                self.hparams_opt.base_lr,
                                self.hparams_opt.steps_per_epoch,
                                self.hparams_opt.max_steps,
                                self.hparams_opt.warmup,
                                self.hparams_opt.sched_type)

        return [opt], [sched]

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure,
                       on_tpu=False, using_native_amp=False, using_lbfgs=False):
        optimizer.step(closure=optimizer_closure)
        self.lr_schedulers().step()
        self.log("lr", self.lr_schedulers().get_last_lr()[0], on_step=True, on_epoch=False, prog_bar=True, logger=True)
        
    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self.model)
