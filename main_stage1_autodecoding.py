# Enable import from parent package
import sys
import os
import shutil
import copy
import time
import itertools

from tqdm.autonotebook import tqdm
import configargparse
from functools import partial
import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable

import torchvision
import torchvision.transforms as transforms
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.image.fid import FrechetInceptionDistance

from src.models.stage1.siren import meta_modules
from src.datasets import CelebAHQ, Implicit2DWrapper, ShapeNet, ShapeNetVoxel, SRNDatasetsLMDB
from src.utils.utils import cond_mkdir, set_seed, compute_psnr, frechet_distance, mean_covar_numpy
from src.models.stage1.nerf.helper import *


# reproducibility
random_seed = 0
set_seed(random_seed)

def image_mse(mask, model_output, gt):
    if mask is None:
        return {'img_loss': ((model_output['model_out'] - gt['img']) ** 2).mean()}
    else:
        return {'img_loss': (mask * (model_output['model_out'] - gt['img']) ** 2).mean()}

p = configargparse.ArgumentParser()

# experiment
p.add_argument('--checkpoint_path', '-r', default='', help='Checkpoint to trained model.')
p.add_argument('--model_path', '-m', default='', help='model to load')
p.add_argument('-p', '--pretrained_path', type=str, default='')
p.add_argument('--save_images', action='store_true')
p.add_argument('--eval', action='store_true')
p.add_argument('--num_workers', default=8, type=int)
p.add_argument('--debug', action='store_true')
p.add_argument('--path_init_context_params', default='', help='context param to load')

# dataset
p.add_argument('--dataset', type=str, default='celeba',
               help='Time interval in seconds until tensorboard summary is saved.')
p.add_argument('--dataset_root', type=str, default='datasets')
p.add_argument('--split', type=str, default='train')
p.add_argument('--resolution', type=int, default=64)
p.add_argument('--num_subsampling', type=int, default=0)

# optimization
p.add_argument('--batch_size', type=int, default=1)
p.add_argument('--optim_outer', type=str, default='adam')
p.add_argument('--lr_outer', type=float, default=3e-6, help='learning rate. default=3e-6')
p.add_argument('--lr_inner', type=float, default=1e-2, help='learning rate. default=1e-2')
p.add_argument('--repeat_sampling', action='store_true')
p.add_argument('--use_same_lr', action='store_true')
p.add_argument('--weight_decay_outer', type=float, default=0.0)
p.add_argument('--weight_decay_inner', type=float, default=0.0)
p.add_argument('--weight_orthogonal', type=float, default=0.0)
p.add_argument('--sparsity_inner', type=float, default=0.0)
p.add_argument('--sparsity_outer', type=float, default=0.0)
p.add_argument('--num_epochs', type=int, default=50,
               help='Number of epochs to train for.')
p.add_argument('--save_freq', type=int, default=50,
               help='Number of epochs to train for.')
p.add_argument('--num_steps', type=int, default=0,
               help='Number of epochs to train for.')
p.add_argument('--clip_grad', action='store_true')
p.add_argument('--clip_grad_norm', default=1.0, type=float)
p.add_argument('--double_precision', action='store_true')
p.add_argument('--min_lr_outer', type=float, default=0.0)
p.add_argument('--use_lr_scheduler', action='store_true')
p.add_argument('--first_order', action='store_true')
p.add_argument('--start_from_zero', action='store_true')

# NeRF strategy
p.add_argument('--epoch_for_full_rendering', type=int, default=5)
p.add_argument('--subsampled_views', type=int, default=16,
               help='Number of sampling views per scene when training')               
p.add_argument('--subsampled_pixels', type=int, default=512,
               help='Number of sampling pixels per each view when training')
p.add_argument('--num_samples_per_ray', type=int, default=31,
               help='Number of points per each ray')
p.add_argument('--near', type=float, default=0.8)
p.add_argument('--far', type=float, default=1.8)
p.add_argument('--use_noise', action='store_true')
p.add_argument('--prob_mask_sampling', type=float, default=0)
p.add_argument('--rgb_activation', type=str, default='no_use')
p.add_argument('--density_activation', type=str, default='elu')

p.add_argument('--num_view_eval', type=int, default=1,
               help='Number of sampling views per scene when training')               
p.add_argument('--use_full_view', action='store_true')

p.add_argument('--zero_to_one', action='store_true')
p.add_argument('--rendering_type', type=str, default='baseline')

p.add_argument('--functa_rendering', action='store_true')

p.add_argument('--mip_rendering', action='store_true')
p.add_argument('--ray_shape', type=str, default='cone', choices=['cylinder', 'cone'])

# ShapeNet
p.add_argument('--random_scale', action='store_true')

# model
p.add_argument('--model_type', type=str, default='latentmodulatedsiren')
p.add_argument('--width', type=int, default=512)
p.add_argument('--depth', type=int, default=15)
p.add_argument('--w0', type=float, default=30.0)
p.add_argument('--hidden_features', type=int, default=256)
p.add_argument('--k_mixtures', type=int, default=64)
p.add_argument('--init_path', type=str, default='')

p.add_argument('--use_ema', action='store_true')

# FFNet configs
p.add_argument('--sigma_ffnet', type=float, default=1.0)
p.add_argument('--num_frequencies_ffnet', type=int, default=64)

# HashGrid configs
p.add_argument('--n_levels_hash', type=int, default=16)
p.add_argument('--n_features_hash', type=int, default=2)
p.add_argument('--n_log2_hash', type=int, default=15)
p.add_argument('--base_res_hash', type=int, default=16)
p.add_argument('--scale_hash', type=float, default=1.5)

# TensoRF configs
p.add_argument('--grid_size_tensorf', type=int, default=64)
p.add_argument('--shadingMode', type=str, default='RGB')
p.add_argument('--density_n_comp', type=int, default=8)
p.add_argument('--appearance_n_comp', type=int, default=24)
p.add_argument('--lr_outer_tensorf', type=float, default=20.0, help='scale for learning rate. default=20.0')


opt = p.parse_args()

if opt.eval:
    sampling = None
elif opt.num_subsampling == 0:
    sampling = None
else:
    sampling = opt.num_subsampling

if opt.functa_rendering:
    opt.rendering_type = 'functa'
if opt.mip_rendering:
    opt.rendering_type = 'mip-nerf'

# setup dataset
if 'shapenet' == opt.dataset.lower():
    dataset = ShapeNet(split=opt.split, sampling=sampling, dataset_root=opt.dataset_root, random_scale=opt.random_scale)
    in_channels, out_channels, pred_type = 3, 1, 'voxel'
elif 'shapenetvoxel' == opt.dataset.lower():
    dataset = ShapeNetVoxel(split=opt.split, sampling=sampling, dataset_root=opt.dataset_root)
    in_channels, out_channels, pred_type = 3, 1, 'voxel'
elif 'srn_cars' in opt.dataset.lower():
    dataset = SRNDatasetsLMDB(category='cars', split=opt.split, opt=opt, dataset_root=opt.dataset_root, zero_to_one=opt.zero_to_one)
    opt.lindisp = dataset.lindisp
    opt.randomized = opt.use_noise 
    opt.white_bkgd = True 
    opt.H, opt.W = opt.resolution, opt.resolution
#    assert opt.resolution == 128
    in_channels, out_channels, pred_type = 3, 4, 'scene'
else:
    # image dataset
    if 'celeba' in opt.dataset.lower():
        dataset = CelebAHQ(split=opt.split, downsampled=False, dataset_root=opt.dataset_root)
        resampling = 'bicubic'
    elif 'ffhq' in opt.dataset.lower():
        assert opt.resolution == 256
        dataset = FFHQ(split=opt.split, dataset_root=opt.dataset_root)
    elif 'cifar10' in opt.dataset.lower():
        assert opt.resolution == 32
        dataset = torchvision.datasets.CIFAR10(root=opt.dataset_root, train=not opt.eval,
                                               download=True)
        resampling = ''

    # coordinate dataset
    dataset = Implicit2DWrapper(dataset, sidelength=opt.resolution, sampling=sampling, resampling=resampling, istuple=opt.dataset == 'cifar10')
    in_channels, out_channels, pred_type = 2, 3, 'image'

dataloader = DataLoader(dataset, shuffle=not opt.eval, batch_size=opt.batch_size, pin_memory=True, num_workers=opt.num_workers)

# checkpoint
if not opt.eval:
    assert len(opt.checkpoint_path) > 0
    model_dir = os.path.join(opt.checkpoint_path)
    cond_mkdir(model_dir)

    checkpoints_dir = os.path.join(model_dir, 'metainits')
    cond_mkdir(checkpoints_dir)

    contexts_dir = os.path.join(model_dir, 'contexts')
    cond_mkdir(contexts_dir)

    with open(os.path.join(model_dir, 'train.log'), 'a') as f:
        f.write('train log\n')
else:
    opt.checkpoint_path = opt.model_path.split('/metainits')[0]
    model_dir = os.path.join(opt.checkpoint_path)
    cond_mkdir(model_dir)
    
    contexts_dir = os.path.join(model_dir, 'contexts')
    cond_mkdir(contexts_dir)

# full image rendering
if pred_type == 'scene':
    rendering_dir = os.path.join(model_dir, 'rendering')
    os.makedirs(rendering_dir, exist_ok=True)

# define the loss       
loss_fn = partial(image_mse, None)
PSNR = PeakSignalNoiseRatio()
if opt.eval:
    if pred_type == 'image':
        from src.utils.inception import get_inception_model
        from src.utils.prdc import compute_prdc

        inception = get_inception_model().cuda()
        inception.eval()

        gt_act_path = f'{opt.dataset_root}/{opt.dataset}-{opt.split}-res{opt.resolution}-activations.pth'
       
        if not os.path.exists(gt_act_path):
            compute_activation = True
            acts = []
        else:
            acts = torch.load(gt_act_path)
            compute_activation = False


if opt.model_type.lower() == 'latentmodulatedsiren':
    model = meta_modules.LatentModulatedSiren(width=opt.width,
                                              depth=opt.depth,
                                              latent_dim=opt.hidden_features,
                                              use_meta_sgd=opt.use_meta_sgd)

elif 'linearmixture' in opt.model_type.lower() or 'mixtureinr' in opt.model_type.lower():
    if 'hashgrid' in opt.model_type.lower():
        embedding_type = 'hashgrid'
    elif 'hashensemble' in opt.model_type.lower():
        embedding_type = 'hashensemble'
    elif 'triangleffnet' in opt.model_type.lower():
        embedding_type = 'triangleffnet'
    elif 'reluffnet' in opt.model_type.lower():
        embedding_type = 'reluffnet'        
    elif 'ffnet' in opt.model_type.lower():
        embedding_type = 'ffnet'
    elif 'gaussian' in opt.model_type.lower():
        embedding_type = 'gaussian'
    else:
        embedding_type = 'none'

    if 'layerwise' in opt.model_type.lower():
        mixture_type = 'layerwise'
    else:
        mixture_type = 'all-layer'

    if 'repeat' in opt.model_type.lower():
        repeat_weight = True
    else:
        repeat_weight = False

    if 'latent' in opt.model_type.lower():
        use_latent_embedding = True
        std_latent = opt.model_type.lower().split('latent')[-1].split('-')[0]
        if len(std_latent) > 0:
            std_latent = float(std_latent)
        else:
            #std_latent = 0.00001
            std_latent = 0.0
            
        print(f'use latent with std={std_latent}')            
    else:
        use_latent_embedding = False
        std_latent = 0.0


    if 'mixtureinr' in opt.model_type.lower():
        model = meta_modules.LinearMixtureINR(width=opt.width,
                                              depth=opt.depth,
                                              in_channels=in_channels,
                                              out_channels=out_channels,
                                              k_mixtures=opt.k_mixtures,                                             
                                              w0=opt.w0,
                                              mixture_type=mixture_type,
                                              embedding_type=embedding_type,
                                              pred_type=pred_type,
                                              use_meta_sgd=False,
                                              use_latent_embedding=use_latent_embedding,
                                              latent_channels=opt.hidden_features,
                                             )

    elif 'linearmixture' in opt.model_type.lower(): # for the compatability of previous version
        assert(False) # do not use anymore
        # model = meta_modules.LinearMixtureSIREN(width=opt.width,
        #                                         depth=opt.depth,
        #                                         k_mixtures=opt.k_mixtures,
        #                                         in_channels=in_channels,
        #                                         out_channels=out_channels,
        #                                         w0=opt.w0,
        #                                         mixture_type=mixture_type,
        #                                         embedding_type=embedding_type,
        #                                         pred_type=pred_type,
        #                                         use_meta_sgd=False,
        #                                         init_path=opt.init_path,
        #                                         use_latent_embedding=use_latent_embedding,
        #                                         normalize_mixture=use_normalize_mixture,
        #                                         latent_channels=opt.hidden_features,
        #                                         num_frequencies_ffnet=opt.num_frequencies_ffnet,
        #                                         sigma_ffnet=opt.sigma_ffnet)

elif 'tensorf' in opt.model_type.lower():
    assert(out_channels == 4)
    
    if 'vm' in opt.model_type.lower():
        from src.models.stage1.tensorf.tensoRF import MixtureTensorVMNet as TensoRFNet
    else:
        from src.models.stage1.tensorf.tensoRF import MixtureTensorCPNet as TensoRFNet

    if 'latent' in opt.model_type.lower():
        use_latent_embedding = True
        std_latent = opt.model_type.lower().split('latent')[-1].split('-')[0]
        if len(std_latent) > 0:
            std_latent = float(std_latent)
        else:
            std_latent = 0.0

        print(f'use latent with std={std_latent}')
    else:
        use_latent_embedding = False
        std_latent = 0.0
                
    model = TensoRFNet(opt)

model = model.cuda()
if opt.eval:
    model.eval()
else:
    model.train()
# count parameters
nparameters = 0
nparams = []
for param in model.get_parameters():
    nparam = param.nelement()
    nparams.append(nparam)
    nparameters += nparam
print("Parameter count: ", nparameters)

if opt.use_ema and not opt.eval:
    from torch_ema import ExponentialMovingAverage
    model_ema = ExponentialMovingAverage(model.get_parameters(), decay=0.9999)

if not opt.eval:
    if(len(opt.pretrained_path) > 0):
        ckpt = torch.load(opt.pretrained_path, map_location='cpu')
        model.load_state_dict(ckpt, strict=False)
        
        loaded_epoch = opt.pretrained_path.split('epoch')[-1].split('.pth')[0]
        pretrained_root = opt.pretrained_path.split('metainits')[0]
        if opt.split == 'train':
            all_context_params = torch.load(os.path.join(pretrained_root, 'contexts', f'context-epoch{loaded_epoch}.pth'))
        elif opt.split == 'test':
            all_context_params = torch.load(os.path.join(pretrained_root, 'contexts', f'context-epoch{loaded_epoch}-test.pth'))
    elif len(opt.path_init_context_params) > 0:
        all_context_params = torch.load(opt.path_init_context_params)
        all_context_params = (all_context_params - torch.mean(all_context_params, dim=0))/(1e-8 + torch.std(all_context_params, dim=0)) * std_latent
    else:
        #all_context_params = torch.zeros(len(dataset), opt.hidden_features)
        if use_latent_embedding:
            hidden_features = opt.hidden_features 
        else:
            hidden_features = model.get_context_params(1).numel()
        all_context_params = torch.randn(len(dataset), hidden_features) * std_latent
    
    if 'tensorf' in opt.model_type:
        param_group = model.get_optparam_groups(opt.lr_outer*opt.lr_outer_tensorf, opt.lr_outer)
    else:
        param_group = [{'params': model.get_parameters(), 'lr': opt.lr_outer}]

    if opt.optim_outer.lower() == 'sgd':
        optim = torch.optim.SGD(param_group, lr=opt.lr_outer, weight_decay=opt.weight_decay_outer)
    elif opt.optim_outer.lower() == 'adam':
        optim = torch.optim.AdamW(param_group, lr=opt.lr_outer, weight_decay=opt.weight_decay_outer)  

    if opt.use_lr_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=opt.num_epochs, eta_min=opt.min_lr_outer)

else:
    assert len(opt.model_path) > 0
    model.load_state_dict(torch.load(opt.model_path))
    opt.num_epochs = 1
    if opt.use_ema:
        loaded_epoch = opt.model_path.split('ema-epoch')[-1].split('.pth')[0]
    else:
        loaded_epoch = opt.model_path.split('epoch')[-1].split('.pth')[0]

    if opt.num_steps > 0 and opt.start_from_zero:
        if use_latent_embedding:
            hidden_features = opt.hidden_features 
        else:
            hidden_features = model.get_context_params(1).numel()
        all_context_params = torch.zeros(len(dataset), hidden_features)
    else:
        if opt.split == 'train':
            all_context_params = torch.load(os.path.join(contexts_dir, f'context-epoch{loaded_epoch}.pth'))
        elif opt.split == 'test':
            all_context_params = torch.load(os.path.join(contexts_dir, f'context-epoch{loaded_epoch}-test.pth'))

total_steps = 0
with tqdm(total=len(dataloader) * opt.num_epochs) as pbar:
    list_context_params, gen_acts = [], []
    for epoch in range(opt.num_epochs):
        all_losses, all_psnr, all_acc, steps = 0.0, 0.0, 0.0, 0
        for step, (model_input_batch, gt_batch) in enumerate(dataloader):
            start_time = time.time()
            if pred_type == 'scene' and ((epoch % opt.epoch_for_full_rendering == 0 and step == 0) or opt.use_full_view):
                model_input_eval, gt_eval = get_samples_for_nerf(copy.deepcopy(model_input_batch),
                                                                 copy.deepcopy(gt_batch),
                                                                 opt,
                                                                 view_num=opt.num_view_eval,
                                                                 pixel_sampling=False
                                                                )

            if pred_type == 'scene':
                if opt.use_full_view:
                    model_input, gt = model_input_eval, gt_eval
                else:
                    if opt.repeat_sampling:
                        model_input, gt = get_samples_for_nerf(copy.deepcopy(model_input_batch), copy.deepcopy(gt_batch), opt)
                    else:
                        model_input, gt = get_samples_for_nerf(model_input_batch, gt_batch, opt)
            else:
                model_input, gt = model_input_batch, gt_batch

            model_input = {key: value.cuda() for key, value in model_input.items()}
            gt = {key: value.cuda() for key, value in gt.items()}

            batch_size = gt['img'].size(0)
            if opt.double_precision:
                model_input = {key: value.double() for key, value in model_input.items()}
                gt = {key: value.double() for key, value in gt.items()}

            # reset context-params
            all_context_params = all_context_params.cuda()
            context_params = all_context_params[model_input['idx']].clone().detach()

            if (not opt.eval) or (opt.eval and opt.num_steps > 0):
                context_params.requires_grad_()

            for inner_step in range(opt.num_steps):
                if pred_type == 'scene' and opt.repeat_sampling and inner_step > 0:
                    model_input, gt = get_samples_for_nerf(copy.deepcopy(model_input_batch), copy.deepcopy(gt_batch), opt)
                    model_input = {key: value.cuda()   for key, value in model_input.items()}
                    gt = {key: value.cuda() for key, value in gt.items()}

                pred_inner = model(model_input, context_params)
                if pred_type == 'scene': # volume rendering
                    pred_inner = nerf_volume_rendering(pred_inner, opt)
                loss_inner = loss_fn(pred_inner, gt)

                if opt.weight_decay_inner:
                    loss_inner['img_loss'] = loss_inner['img_loss'] + (opt.weight_decay_inner / 2) * (context_params**2).sum()
                if opt.sparsity_inner:
                    loss_inner['img_loss'] = loss_inner['img_loss'] + opt.sparsity_inner * pred_inner['l1_mean']

                grad_inner = torch.autograd.grad(loss_inner['img_loss'],
                                                 context_params,
                                                 create_graph=False)[0]
                if opt.use_same_lr:
                    lr_inner = scheduler.get_lr()
                else:
                    lr_inner = opt.lr_inner

                context_params = context_params - lr_inner * grad_inner

            model_output = model(model_input, context_params)
            if pred_type == 'scene': # volume rendering
                model_output = nerf_volume_rendering(model_output, opt)
            losses = loss_fn(model_output, gt)

            train_loss = 0.
            for loss_name, loss in losses.items():
                single_loss = loss.mean()
                train_loss += single_loss.cpu()

            # mse_loss (default)
            all_losses += float(train_loss) * batch_size
            
            # PSNR
            for pred_img, gt_img in zip(model_output['model_out'].cpu(), gt['img'].cpu()):
                if pred_type == 'voxel':
                    #psnr = PSNR(pred_img * 0.5 + 0.5, gt_img * 0.5 + 0.5) # rescale from [-1, 1] to [0, 1]
                    psnr = compute_psnr(pred_img * 0.5 + 0.5, gt_img * 0.5 + 0.5)
                elif pred_type == 'scene':
                    if dataset.zero_to_one:
                        #psnr = PSNR(pred_img, gt_img)
                        psnr = compute_psnr(pred_img, gt_img)
                    else:
                       #psnr = PSNR(pred_img * 0.5 + 0.5, gt_img * 0.5 + 0.5)
                       psnr = compute_psnr(pred_img * 0.5 + 0.5, gt_img * 0.5 + 0.5)
                else:
                    #psnr = PSNR(pred_img * 0.5 + 0.5, gt_img * 0.5 + 0.5)
                    psnr = compute_psnr(pred_img * 0.5 + 0.5, gt_img * 0.5 + 0.5)
                all_psnr += float(psnr)
                steps += 1

            # voxel accuracy
            if pred_type == 'voxel':
                pred_voxel = model_output['model_out'] >= 0.0 # [non-exist (-1), exists (+1)]
                gt_voxel = gt['img'] >= 0.0
                acc = (pred_voxel == gt_voxel).float().mean()
                all_acc += float(acc) * batch_size

            # FID
            if opt.eval and pred_type == 'image':
                pred_imgs = model_output['model_out'].view(-1, opt.resolution, opt.resolution, 3).permute(0, 3, 1, 2) * 0.5 + 0.5
                gt_imgs = gt['img'].view(-1, opt.resolution, opt.resolution, 3).permute(0, 3, 1, 2) * 0.5 + 0.5
                # xs_fid = (gt_imgs * 255.).to(dtype=torch.uint8)
                # xs_rec_fid = (pred_imgs * 255.).to(dtype=torch.uint8)
                # FID.update(xs_fid, real=True)
                # FID.update(xs_rec_fid, real=False)
                if total_steps == 0:
                    grid_samples = torchvision.utils.make_grid(pred_imgs, nrow=int(math.sqrt(pred_imgs.size(0))))
                    torchvision.utils.save_image(grid_samples, os.path.join(opt.checkpoint_path, f'samples{opt.split}.png'))            
                    grid_gts = torchvision.utils.make_grid(gt_imgs, nrow=int(math.sqrt(gt_imgs.size(0))))
                    torchvision.utils.save_image(grid_gts, os.path.join(opt.checkpoint_path, f'gts{opt.split}.png'))

                xs_fid = (gt_imgs * 255.).to(dtype=torch.uint8)
                xs_rec_fid = (pred_imgs * 255.).to(dtype=torch.uint8)

                # ground truth activation
                if compute_activation:
                    act = inception(gt_imgs.cuda())
                    acts.append(act.cpu())

                # reconstruction activation
                #act = inception(pred_imgs)
                #gen_acts.append(act.detach().cpu())
                act = inception(xs_rec_fid.float() / 255.0)
                gen_acts.append(act.detach().cpu()) 

            total_steps += 1

            # optimization procedure during training
            if not opt.eval:
                losses_all = losses['img_loss']
                if opt.sparsity_outer:
                    losses_all = losses_all + opt.sparsity_outer * model_output['l1_mean']
                if opt.weight_decay_inner:
                    losses_all = losses_all + (opt.weight_decay_inner / 2) * (context_params**2).sum()
                if opt.weight_orthogonal > 0.0 and total_steps % 2 == 0: # for faster training
                    fi, reg = 0, []
                    for _, all_weights in model.model_param.named_parameters():
                        if fi % 2 == 0: # only weights, not biases
                            w = all_weights.view(model.k_mixtures, -1)
                            #W = w @ w.T - torch.eye(model.k_mixtures).to(w.device)
                            W = F.cosine_similarity(w, w) - torch.eye(model.k_mixtures).to(w.device)
                            reg.append(torch.abs(W).sum() / (model.k_mixtures*(model.k_mixtures-1)))

                    losses_all = losses_all + sum(reg) * opt.weight_orthogonal

                all_grad = torch.autograd.grad(losses_all, [context_params] + list(model.get_parameters()), allow_unused=True)
   
                context_grad, task_grad = all_grad[0], all_grad[1:]

                idx = model_input['idx']
                if opt.use_same_lr:
                    lr_inner = scheduler.get_lr()[0]
                else:
                    lr_inner = opt.lr_inner

                all_context_params[idx] = all_context_params[idx] - lr_inner * context_grad.cpu()

                # model parameter
                #task_grad = torch.autograd.grad(losses_all, model.get_parameters())

                if opt.clip_grad:
                    torch.nn.utils.clip_grad_norm_(model.get_parameters(), max_norm=opt.clip_grad_norm)

                if opt.lr_outer > 0.0:
                    optim.zero_grad()

                    #grad_norm = dict()
                    # set gradients of parameters manually
                    for c, param in enumerate(model.get_parameters()):
                        param.grad = task_grad[c].detach()
                        #grad_norm[c] = float(param.grad.data.norm(2))

                    optim.step()

                    #grad_description = '[norm]'
                    
                    #for key, param in model.get_named_parameters():
                        #val = float(param.grad.data.norm(2))
                        #grad_description += f' {key}:{val:.1e}'
                    #print(grad_description)
            else:
                if opt.num_steps > 0:
                    idx = model_input['idx']
                    all_context_params[idx] = context_params

            if opt.use_ema and not opt.eval:
                model_ema.update(model.get_parameters())

            context_params = all_context_params[model_input['idx']].clone().detach()
            description = f'mse_loss:{all_losses/steps:.4f} PSNR:{all_psnr/steps:.2f} Ctx-mean:{float(torch.abs(context_params).mean()):.4e}'
            if opt.use_lr_scheduler:
                description += f' LR:{scheduler.get_last_lr()[-1]:.4e}'
            if pred_type == 'voxel':
                description += f' acc:{all_acc/steps:.4f}'
            if opt.sparsity_inner or opt.sparsity_outer:
                l1_mean = float(model_output['l1_mean'])
                description += f' sparsity_l1:{l1_mean:.4f}'
                               
            pbar.set_description(description)
            pbar.update(1)


            if pred_type == 'scene':
                if epoch % opt.epoch_for_full_rendering == 0 and step == 0:
                    model_input_eval = {key: value.cuda() for key, value in model_input_eval.items()}
                    #gt_eval = {key: value.cuda() for key, value in gt_eval.items()}
                    with torch.no_grad():
                        model_output_full = model(model_input_eval, context_params.cuda())
                        model_output_full = nerf_volume_rendering(model_output_full, opt, 'all')
                        if not opt.eval:
                            save_rendering_output(model_output_full, gt_eval, opt, 
                                                os.path.join(rendering_dir, f'E{epoch}_S{step}.png'))
                        else:
                            save_rendering_output(model_output_full, gt_eval, opt, 
                                                os.path.join(rendering_dir, f'Eval-E{loaded_epoch}_S{step}.png'))


        if not opt.eval:
            if ((epoch % opt.save_freq == 0) or (epoch == opt.num_epochs - 1) ):
                torch.save(model.state_dict(), os.path.join(checkpoints_dir, f'epoch{epoch}.pth'))

                if opt.use_ema:
                    shadow_params = model_ema.shadow_params
                    ema_dict = {}
                    named_model_params = list(model.get_named_parameters())

                    for (k, v), param in zip(named_model_params, shadow_params):
                        ema_dict['model.' + k] = param                    
                    torch.save(ema_dict, os.path.join(checkpoints_dir, f'ema-epoch{epoch}.pth'))

                if opt.num_steps == 0:
                    if opt.split == 'train':
                        torch.save(all_context_params, os.path.join(contexts_dir, f'context-epoch{epoch}.pth'))
                    elif opt.split == 'test':
                        torch.save(all_context_params, os.path.join(contexts_dir, f'context-epoch{epoch}-test.pth'))

            if opt.use_lr_scheduler:
                scheduler.step()

            with open(os.path.join(model_dir, 'train.log'), 'a') as f:    
                f.write(f'[epoch {epoch}]' + description + '\n')

        else:
            if opt.num_steps > 0:
                if opt.split == 'train':
                    torch.save(all_context_params, os.path.join(contexts_dir, f'context-epoch{epoch}-{opt.num_steps}step.pth'))
                elif opt.split == 'test':
                    torch.save(all_context_params, os.path.join(contexts_dir, f'context-epoch{epoch}-test-{opt.num_steps}step.pth'))

            description = f'mse_loss:{all_losses/steps:.4f} PSNR:{all_psnr/steps:.2f}'

            if pred_type == 'image':
                if compute_activation:
                    acts = torch.cat(acts, dim=0)
                    torch.save(acts, gt_act_path)

                acts = acts.numpy()
                gen_acts = torch.cat(gen_acts, dim=0).numpy()
                
                mu_gt, sigma_gt = mean_covar_numpy(acts)
                mu_gen, sigma_gen = mean_covar_numpy(gen_acts)

                fid = frechet_distance(mu_gt, sigma_gt, mu_gen, sigma_gen)
                prdc = compute_prdc(acts, gen_acts, nearest_k=3)
                precision, recall = prdc['precision'], prdc['recall']
                description += f' rFID: {fid} rPrecision: {precision} rRecall: {recall}'

                #fid_score = FID.compute()
                #description += f' [0-1] rFID:{float(fid_score):.4f}'
            if pred_type == 'voxel':
                description += f' acc:{all_acc/steps:.4f}'                

            print(description)

            with open(os.path.join(model_dir, 'eval.log'), 'a') as f:
                f.write(f'Dataset: {opt.dataset} Split: {opt.split}')
                print(opt, file=f)
                f.write(description + '\n')



