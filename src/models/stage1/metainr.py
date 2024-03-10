import os
import copy
from functools import partial
from typing import Optional, Tuple, List

import torch
import torch.distributed as dist

import pytorch_lightning as pl
from omegaconf import OmegaConf
from torchmetrics import PeakSignalNoiseRatio

from src.models.stage1.siren import meta_modules
from src.models.stage1.nerf.helper import *
from ...optimizers import build_scheduler
from src.utils.utils import image_mse
from src.utils.utils import cond_mkdir

import pdb

class MetaINR(pl.LightningModule):
    def __init__(self,
                 hparams: OmegaConf) -> None:
        super().__init__()
        
        # setting required configurations
        self.pred_type = hparams.dataset.pred_type
        if self.pred_type in ['scene', 'nerf']:
            self._hparams_nerf = hparams.dataset.hparams_nerf
            self.zero_to_one = True if self._hparams_nerf.rgb_activation == 'sigmoid' else False
            self.resolution = hparams.dataset.hparams_nerf.resolution
            hparams.dataset.hparams_nerf.H = self.resolution
            hparams.dataset.hparams_nerf.W = self.resolution            
        else:
            self.resolution = hparams.dataset.resolution
        self.repeat_sampling = hparams.dataset.repeat_sampling # for nerf scene

        hparams_inr = hparams.hparams_inr
        if hparams.model_type == 'latentmodulatedsiren':
            self.inr = meta_modules.LatentModulatedSiren(width=hparams_inr.width,
                                                         depth=hparams_inr.depth,
                                                         latent_dim=hparams_inr.hidden_features,
                                                         use_meta_sgd=hparams_inr.use_meta_sgd)
        elif hparams.model_type == 'linearmixture':
            self.inr = meta_modules.LinearMixtureSIREN(**hparams_inr)
        elif hparams.model_type == 'mixtureinr':
            self.inr = meta_modules.LinearMixtureINR(**hparams_inr)

        # count parameters
        nparameters = 0
        nparams = []
        for param in self.inr.get_parameters():
            nparam = param.nelement()
            nparams.append(nparam)
            nparameters += nparam
        print("Parameter count: ", nparameters)

        self.loss_fn = partial(image_mse, None)

        self._num_opt_steps = 0
        self._hparams_exp = hparams.experiment
        self._hparams_opt = hparams.optimizer
        self._hparams_inr = hparams_inr

        # optimization
        self.automatic_optimization=False
        self.meta_grad_init = [0 for _ in self.inr.get_parameters()]

        # metrics
        self.PSNR = PeakSignalNoiseRatio()


    def forward(self, x: torch.FloatTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        return self.self.inr(x)


    def shared_step(self, batch, context_params):
        model_input_batch, gt_batch = batch

        if self.pred_type == 'scene':
            if self.repeat_sampling:
                model_input, gt = get_samples_for_nerf(copy.deepcopy(model_input_batch), copy.deepcopy(gt_batch), self._hparams_nerf)
            else:
                model_input, gt = get_samples_for_nerf(model_input_batch, gt_batch, self._hparams_nerf)
        else:
            model_input, gt = model_input_batch, gt_batch


        # model_input = {key: value.cuda() for key, value in model_input.items()}
        # gt = {key: value.cuda() for key, value in gt.items()}
        
        model_input = {key: value.to(self.device)   for key, value in model_input.items()}
        gt = {key: value.to(self.device) for key, value in gt.items()}

        if self._hparams_opt.double_precision:
            model_input = {key: value.double() for key, value in model_input.items()}
            gt = {key: value.double() for key, value in gt.items()}

        if self._hparams_inr.use_meta_sgd:
            meta_sgd_inner = self.inr.meta_sgd_lrs()

        for inner_step in range(self._hparams_opt.num_steps):
            if self.pred_type == 'scene' and self.repeat_sampling and inner_step > 0:
                model_input, gt = get_samples_for_nerf(copy.deepcopy(model_input_batch), copy.deepcopy(gt_batch), self._hparams_nerf)
                model_input = {key: value.cuda() for key, value in model_input.items()}
                gt = {key: value.cuda() for key, value in gt.items()}

            pred_inner = self.inr(model_input, context_params)
            if self.pred_type == 'scene': # volume rendering
                pred_inner = nerf_volume_rendering(pred_inner, self._hparams_nerf)
            loss_inner = self.loss_fn(pred_inner, gt)

            if self._hparams_opt.weight_decay_inner:
                loss_inner['img_loss'] = loss_inner['img_loss'] + (self._hparams_opt.weight_decay_inner / 2) * (context_params**2).mean()
            if self._hparams_opt.sparsity_inner:
                loss_inner['img_loss'] = loss_inner['img_loss'] + self._hparams_opt.sparsity_inner * pred_inner['l1_mean']

            grad_inner = torch.autograd.grad(loss_inner['img_loss'],
                                             context_params,
                                             create_graph=not self._hparams_opt.first_order)[0]
            if self.world_size > 1:
                grad_inner = grad_inner / self.world_size
            if self._hparams_inr.use_meta_sgd:
                context_params = context_params - self._hparams_opt.lr_inner * (meta_sgd_inner * grad_inner)
            else:
                context_params = context_params - self._hparams_opt.lr_inner * grad_inner

        model_output = self.inr(model_input, context_params)
        if self.pred_type == 'scene': # volume rendering
            model_output = nerf_volume_rendering(model_output, self._hparams_nerf)
        losses = self.loss_fn(model_output, gt)

        if self._hparams_opt.weight_decay_inner:
            losses['wd_inner'] = (self._hparams_opt.weight_decay_inner / 2) * (context_params**2).mean()
        if self._hparams_opt.sparsity_inner:
            losses['sparsity_inner'] = self._hparams_opt.sparsity_inner * model_output['l1_mean']

        train_loss = 0.
        for loss_name, loss in losses.items():
            single_loss = loss.mean()
            train_loss += single_loss.cpu()

        log_dict_train = dict()
        log_dict_train['mse'] = float(train_loss)

        # PSNR
        for pred_img, gt_img in zip(model_output['model_out'], gt['img']):
            if self.pred_type == 'voxel':
                psnr = self.PSNR(pred_img * 0.5 + 0.5, gt_img * 0.5 + 0.5) # rescale from [-1, 1] to [0, 1]
            elif self.pred_type == 'scene':
                if self.zero_to_one:
                    psnr = self.PSNR(pred_img, gt_img)
                else:
                    psnr = self.PSNR(pred_img * 0.5 + 0.5, gt_img * 0.5 + 0.5)
            else:
                psnr = self.PSNR(pred_img * 0.5 + 0.5, gt_img * 0.5 + 0.5)                
                    
            log_dict_train['PSNR'] = float(psnr)

        # voxel accuracy
        if self.pred_type == 'voxel':
            pred_voxel = model_output['model_out'] >= 0.0 # [non-exist (-1), exists (+1)]
            gt_voxel = gt['img'] >= 0.0
            acc = (pred_voxel == gt_voxel).float().mean()
            log_dict_train['/voxel_acc'] = float(acc)
        
        if self._hparams_inr.use_meta_sgd:
            log_dict_train['mLR'] = float(torch.abs(meta_sgd_inner).mean())

        log_dict_train['ctx'] = float(torch.abs(context_params).mean())
        
        return losses, log_dict_train, model_output, gt, context_params

    def on_train_epoch_start(self):
        self.world_size = float(dist.get_world_size())
        self.train()
        self.is_first_step = True

    def on_train_epoch_end(self):
        if self._hparams_opt.use_lr_scheduler:
            self.scheduler.step()

    def training_step(self, batch, batch_idx):
        if self.pred_type == 'scene' and self._hparams_nerf.epoch_for_full_rendering > 0:
            if self.current_epoch % self._hparams_nerf.epoch_for_full_rendering == 0 and self.is_first_step:
                if dist.get_rank() == 0:
                    self.rendering_dir = os.path.join(self.trainer.checkpoint_callback.dirpath, '..', 'rendering')
                    cond_mkdir(self.rendering_dir)

                model_input, gt = batch
                model_input_eval, gt_eval = get_samples_for_nerf(copy.deepcopy(model_input), copy.deepcopy(gt), self._hparams_nerf, view_num=1, pixel_sampling=False)

        # initialise meta-gradient
        meta_grad = copy.deepcopy(self.meta_grad_init)

        # reset context-params
        batch_size = batch[1]['img'].size(0)
        context_params = self.inr.get_context_params(batch_size, False)

        losses, log_dict, model_output, gt, context_params = self.shared_step(batch, context_params)
        
        # if batch_idx % 5 == 0:
        #     save_gt = gt['img'].reshape(-1, 64, 64, 3).permute(0,3,1,2)
        #     pred_rgb = model_output['model_out'].reshape(-1, 64, 64, 3).permute(0,3,1,2)
        #     combined = torch.cat((save_gt, pred_rgb), -1)
        #     combined = torchvision.utils.make_grid(combined, nrow=1)
        #     torchvision.utils.save_image(combined, 'test.png')

        losses_all = losses['img_loss']
        if self._hparams_opt.sparsity_outer:
            losses_all = losses_all + self._hparams_opt.sparsity_outer * model_output['l1_mean']

        ############################################
        ### optimization procedure during training
        ############################################
        # compute gradient for shared parameters
        task_grad = torch.autograd.grad(losses_all, self.inr.get_parameters())

        # add to meta-gradient
        for g in range(len(task_grad)):
            meta_grad[g] += task_grad[g].detach()

        self.optim.zero_grad()

        # set gradients of parameters manually
        for c, param in enumerate(self.inr.get_parameters()):
            param.grad = meta_grad[c]

            if self.world_size > 1:
                # sync parameters during multi-gpu training
                dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)

        if self._hparams_opt.clip_grad:
            torch.nn.utils.clip_grad_norm_(self.inr.get_parameters(), max_norm=self._hparams_opt.grad_clip_norm)

        self.optim.step()

        log_dict_train = dict()

        for key in log_dict.keys():
            log_dict_train['train/' + key] = log_dict[key]

        self.log_dict(log_dict_train, prog_bar=True, logger=True, on_step=True, on_epoch=True)

        if self.pred_type == 'scene' and self._hparams_nerf.epoch_for_full_rendering > 0:
            if self.current_epoch % self._hparams_nerf.epoch_for_full_rendering == 0 and self.is_first_step:
                self.nerf_visualization(model_input_eval, gt_eval, context_params, os.path.join(self.rendering_dir, f'E{self.current_epoch}_S0.png'))
                self.is_first_step = False

        return losses["img_loss"]


    def on_validation_epoch_start(self):
        self.world_size = float(dist.get_world_size())


    def validation_step(self, batch, batch_idx):
        torch.set_grad_enabled(True)

        # reset context-params
        batch_size = batch[1]['img'].size(0)
        context_params = self.inr.get_context_params(batch_size, True)

        losses, log_dict, model_output, gt, context_params = self.shared_step(batch, context_params)

        log_dict_valid = dict()

        for key in log_dict.keys():
            log_dict_valid['valid/' + key] = log_dict[key]

        self.log_dict(log_dict_valid, prog_bar=True, logger=True, on_step=False, on_epoch=True)

        return losses["img_loss"]

    def nerf_visualization(self, model_input_eval, gt_eval, context_params, save_name):
        assert  model_input_eval['coords'].dim() == 3 
        start, end, total_coords = 0, self._hparams_nerf.chuncking_unit, model_input_eval['coords'].size(1)
        rgbd_outputs = []
        with torch.no_grad():
            if self._hparams_nerf.chuncking_unit != -1:
                while start < total_coords:
                    chuncked_input_eval, start, end = split_dict(model_input_eval, start, end, self._hparams_nerf.chuncking_unit)
                    chuncked_input_eval = {key: value.cuda()   for key, value in chuncked_input_eval.items()}
                    rgbd_outputs.append(self.inr(chuncked_input_eval, context_params)['model_out'])
                rgbd_outputs = torch.cat(rgbd_outputs, 1)
                model_output_full = {'model_in': model_input_eval,
                                    'model_out': rgbd_outputs,
                                    }
            else:
                model_output_full = self.inr(model_input_eval, context_params)
            model_output_full = nerf_volume_rendering(model_output_full, self._hparams_nerf, 'all')
            if torch.distributed.get_rank() == 0:            
                save_rendering_output(model_output_full, gt_eval, self._hparams_nerf, save_name)

    def configure_optimizers(self):
        param_group = [{'params': self.inr.get_parameters(), 'lr': self._hparams_opt.lr_outer}]
        if self._hparams_opt.optim_outer == 'sgd':
            optim = torch.optim.SGD(param_group, lr=self._hparams_opt.lr_outer, weight_decay=self._hparams_opt.weight_decay_outer)
        elif self._hparams_opt.optim_outer in ['adam', 'adamw']:
            optim = torch.optim.AdamW(param_group, lr=self._hparams_opt.lr_outer, weight_decay=self._hparams_opt.weight_decay_outer)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=self._hparams_exp.epochs, eta_min=self._hparams_opt.min_lr_outer)

        self.optim = optim
        self.scheduler = scheduler

        opts, lr_schs = [optim], [{'scheduler': scheduler, 'name': 'lr-outer'}]

        return opts, lr_schs 


    def from_ckpt(self, path: str, strict: bool = True, ignore_keys: Optional[List] = None) -> None:
        ckpt = torch.load(path, map_location='cpu')['state_dict']
        if ignore_keys:
            for k in ignore_keys:
                del ckpt[k]
        self.load_state_dict(ckpt, strict=strict)
        print(f'{path} successfully restored..')