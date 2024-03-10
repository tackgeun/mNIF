# Enable import from parent package
import os
import copy
import time
import math

import configargparse
from tqdm.autonotebook import tqdm
from functools import partial

import torch
from torch.utils.data import DataLoader

import torchvision

from src.models.stage1.siren import meta_modules
from src.datasets import CelebAHQ
from src.datasets import ShapeNet, ShapeNetVoxel, SRNDatasetsLMDB
from src.datasets import Implicit2DWrapper

from src.utils.utils import cond_mkdir, set_seed, frechet_distance, mean_covar_numpy, compute_psnr
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
p.add_argument('--result_path', '-c', default='', help='Checkpoint to trained model.')
p.add_argument('--model_path', '-m', default='', help='model to load')
p.add_argument('--save_images', action='store_true')
p.add_argument('--eval', action='store_true')
p.add_argument('--num_workers', default=8, type=int)
p.add_argument('--debug', action='store_true')

p.add_argument('--ckpt_epoch', default=-1, type=int)

p.add_argument('--context_tag', type=str, default='')

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
p.add_argument('--weight_decay_outer', type=float, default=0.0)
p.add_argument('--weight_decay_inner', type=float, default=0.0)
p.add_argument('--sparsity_inner', type=float, default=0.0)
p.add_argument('--sparsity_outer', type=float, default=0.0)
p.add_argument('--num_epochs', type=int, default=50,
               help='Number of epochs to train for.')
p.add_argument('--save_freq', type=int, default=1,
               help='Number of epochs to train for.')
p.add_argument('--num_inner', type=int, default=3,
               help='Number of epochs to train for.')
p.add_argument('--clip_grad', action='store_true')
p.add_argument('--double_precision', action='store_true')
p.add_argument('--min_lr_outer', type=float, default=0.0)
p.add_argument('--use_lr_scheduler', action='store_true')
p.add_argument('--first_order', action='store_true')

# NeRF strategy
p.add_argument('--epoch_for_full_rendering', type=int, default=5)
p.add_argument('--subsampled_views', type=int, default=32,
               help='Number of sampling views per scene when training')
p.add_argument('--subsampled_pixels', type=int, default=512,
               help='Number of sampling pixels per each view when training')
p.add_argument('--num_samples_per_ray', type=int, default=31,
               help='Number of points per each ray')
p.add_argument('--near', type=float, default=0.8)
p.add_argument('--far', type=float, default=1.8)
p.add_argument('--use_noise', action='store_true')
p.add_argument('--prob_mask_sampling', type=float, default=0)
p.add_argument('--rgb_activation', type=str, default='sigmoid')
p.add_argument('--density_activation', type=str, default='elu')
p.add_argument('--zero_to_one', action='store_true')
p.add_argument('--functa_rendering', action='store_true')
p.add_argument('--chuncking_unit', default=1, type=int)
p.add_argument('--rendering_type', type=str, default='baseline')

# ShapeNet
p.add_argument('--random_scale', action='store_true')

# CelebAHQ
p.add_argument('--transform_type', default='baseline', type=str)

# model

# Mixture of INR configs
p.add_argument('--model_type', type=str, default='latentmodulatedsiren')
p.add_argument('--width', type=int, default=512)
p.add_argument('--depth', type=int, default=15)
p.add_argument('--w0', type=float, default=30.0)
p.add_argument('--hidden_features', type=int, default=256)
p.add_argument('--k_mixtures', type=int, default=64)
p.add_argument('--use_meta_sgd', action='store_true')
p.add_argument('--init_path', type=str, default='')
p.add_argument('--outermost_linear', type=bool, default=False)

opt = p.parse_args()

if opt.eval:
    sampling = None
elif opt.num_subsampling == 0:
    sampling = None
else:
    sampling = opt.num_subsampling

# setup dataset
if 'shapenet' == opt.dataset.lower():
    dataset = ShapeNet(split=opt.split,
                       sampling=sampling,
                       random_scale=opt.random_scale,
                       dataset_root=opt.dataset_root)
    in_channels, out_channels, pred_type = 3, 1, 'voxel'

elif 'shapenetvoxel' == opt.dataset.lower():
    dataset = ShapeNetVoxel(split=opt.split,
                            sampling=sampling,
                            dataset_root=opt.dataset_root)
    in_channels, out_channels, pred_type = 3, 1, 'voxel'

elif 'srn_cars' in opt.dataset.lower():
    assert opt.resolution == 128

    dataset = SRNDatasetsLMDB(split=opt.split,
                              category='cars',
                              zero_to_one=opt.zero_to_one,
                              opt=opt,                              
                              dataset_root=opt.dataset_root)
    opt.lindisp = dataset.lindisp
    opt.randomized = opt.use_noise 
    opt.white_bkgd = True 
    opt.H, opt.W = opt.resolution, opt.resolution
    in_channels, out_channels, pred_type = 3, 4, 'scene'
    
else:
    # image dataset
    if 'celeba' in opt.dataset.lower():
        tf_dataset = 'tf' in opt.dataset.lower()
        dataset = CelebAHQ(split=opt.split,
                           downsampled=False,
                           dataset_root=opt.dataset_root,
                           tf_dataset=tf_dataset)
        #resampling = 'bilinear'
        resampling = 'bicubic'

    elif 'cifar10' in opt.dataset.lower():
        assert opt.resolution == 32
        dataset = torchvision.datasets.CIFAR10(root=opt.dataset_root,
                                               train=opt.split == 'train',
                                               download=True)
        resampling = ''

    # coordinate dataset
    dataset = Implicit2DWrapper(dataset,
                                sidelength=opt.resolution,
                                sampling=sampling,
                                resampling=resampling,
                                transform_type=opt.transform_type,
                                istuple='cifar10' in opt.dataset,
                                ispair=opt.dataset == 'cifar10-contrastive')
    in_channels, out_channels, pred_type = 2, 3, 'image'
   

dataloader = DataLoader(dataset,
                        shuffle=not opt.eval,
                        batch_size=opt.batch_size,
                        pin_memory=True,
                        num_workers=opt.num_workers)

# checkpoint
if not opt.eval:
    assert len(opt.checkpoint_path) > 0
    model_dir = os.path.join(opt.checkpoint_path)
    cond_mkdir(model_dir)

    checkpoints_dir = os.path.join(model_dir, 'metainits')
    cond_mkdir(checkpoints_dir)

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
if opt.eval:
    if pred_type == 'image' and len(opt.context_tag) == 0:
        from src.utils.inception import get_inception_model
        from src.utils.prdc import compute_prdc

        inception = get_inception_model().cuda()
        inception.eval()

        #gt_act_path = f'{opt.dataset_root}/{opt.dataset}-train-activations.pth'
        gt_act_path = f'{opt.dataset_root}/{opt.dataset}-{opt.split}-res{opt.resolution}-activations.pth'
       
        if not os.path.exists(gt_act_path):
            compute_activation = True
            acts = []
        else:
            acts = torch.load(gt_act_path)
            compute_activation = False

if opt.model_type.lower() == 'latentmodulatedsiren': # Functa
    model = meta_modules.LatentModulatedSiren(in_channels=in_channels,
                                                out_channels=out_channels,
                                                width=opt.width,
                                                depth=opt.depth,
                                                latent_dim=opt.hidden_features,
                                                latent_vector_type='instance',
                                                use_meta_sgd=opt.use_meta_sgd,
                                                w0=opt.w0)
    
elif 'linearmixture' in opt.model_type.lower() or 'mixtureinr' in opt.model_type.lower():
    embedding_type = 'none'

    if 'latent' in opt.model_type.lower():
        use_latent_embedding = True
        std_latent = opt.model_type.lower().split('latent')[-1].split('-')[0]
        if (len(std_latent) > 0) and (not opt.eval):
            std_latent = float(std_latent)
        else:
            std_latent = 0.0
    else:
        use_latent_embedding = False
        std_latent = 0.0

    if 'layerwise' in opt.model_type.lower():
        mixture_type = 'layerwise'
    else:
        mixture_type = 'all-layer'

    if 'mixtureinr' in opt.model_type.lower():
        model = meta_modules.LinearMixtureINR(width=opt.width,
                                              depth=opt.depth,
                                              in_channels=in_channels,
                                              out_channels=out_channels,
                                              k_mixtures=opt.k_mixtures,                                             
                                              w0=opt.w0,
                                              mixture_type=mixture_type,
                                              embedding_type=embedding_type,
                                              outermost_linear=opt.outermost_linear,
                                              pred_type=pred_type,
                                              use_meta_sgd=opt.use_meta_sgd,
                                              use_latent_embedding=use_latent_embedding,
                                              std_latent=std_latent,
                                              latent_channels=opt.hidden_features,
                                             )



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

if not opt.eval:
    if 'tensorf' in opt.model_type:
        param_group = model.get_optparam_groups(opt.lr_outer*opt.lr_outer_tensorf, opt.lr_outer)
    else:
        param_group = [{'params': model.get_parameters(), 'lr': opt.lr_outer}]

    if opt.optim_outer.lower() == 'sgd':
        optim = torch.optim.SGD(param_group, lr=opt.lr_outer, weight_decay=opt.weight_decay_outer)
    elif opt.optim_outer.lower() == 'adam':
        optim = torch.optim.AdamW(param_group, lr=opt.lr_outer, weight_decay=opt.weight_decay_outer)

    keep_params = dict()
    with torch.no_grad():
        for name, param in model.get_named_parameters():
            keep_params[name] = param.clone()

    # initialise the starting point for the meta gradient (it's faster to copy this than to create new object)
    meta_grad_init = [0 for _ in model.get_parameters()]

    if opt.use_lr_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=opt.num_epochs, eta_min=opt.min_lr_outer)

else:
    opt.num_epochs = 1
    assert len(opt.model_path) > 0
    model.load_state_dict(torch.load(opt.model_path))   

total_steps = 0
with tqdm(total=len(dataloader) * opt.num_epochs) as pbar:
    if not opt.eval:
        with torch.no_grad():
            for name, param in model.get_named_parameters():
                param = keep_params[name].clone()

    list_context_params, gen_acts = [], []
    for epoch in range(opt.num_epochs):
        all_losses, all_psnr, all_acc, steps = 0.0, 0.0, 0.0, 0
        for step, (model_input_batch, gt_batch) in enumerate(dataloader):
            start_time = time.time()
            if pred_type == 'scene' and epoch % opt.epoch_for_full_rendering == 0 and step == 0:
                model_input_eval, gt_eval = get_samples_for_nerf(copy.deepcopy(model_input_batch), copy.deepcopy(gt_batch), opt, view_num=1, pixel_sampling=False)
            
            if pred_type == 'scene':
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

            if not opt.eval:
                # initialise meta-gradient
                meta_grad = copy.deepcopy(meta_grad_init)

            # reset context-params
            context_params = model.get_context_params(batch_size, opt.eval)

            if opt.use_meta_sgd:
                meta_sgd_inner = model.meta_sgd_lrs()
            for inner_step in range(opt.num_inner):
                if pred_type == 'scene' and opt.repeat_sampling and inner_step > 0:
                    model_input, gt = get_samples_for_nerf(copy.deepcopy(model_input_batch), copy.deepcopy(gt_batch), opt)
                    model_input = {key: value.cuda() for key, value in model_input.items()}
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
                                                create_graph=not opt.eval and not opt.first_order)[0]

                if opt.use_meta_sgd:
                    context_params = context_params - opt.lr_inner * (meta_sgd_inner * grad_inner)
                else:
                    context_params = context_params - opt.lr_inner * grad_inner

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
                    psnr = compute_psnr(pred_img * 0.5 + 0.5, gt_img * 0.5 + 0.5) # rescale from [-1, 1] to [0, 1]
                elif pred_type == 'scene':
                    if dataset.zero_to_one:
                        psnr = compute_psnr(pred_img, gt_img)
                    else:
                       psnr = compute_psnr(pred_img * 0.5 + 0.5, gt_img * 0.5 + 0.5) # rescale from [-1, 1] to [0, 1]
                else:
                    psnr = compute_psnr(pred_img * 0.5 + 0.5, gt_img * 0.5 + 0.5) # rescale from [-1, 1] to [0, 1]
                        
                all_psnr += psnr
                steps += 1

            # voxel accuracy
            if pred_type == 'voxel':
                pred_voxel = model_output['model_out'] >= 0.0 # [non-exist (-1), exists (+1)]
                gt_voxel = gt['img'] >= 0.0
                acc = (pred_voxel == gt_voxel).float().mean()
                all_acc += float(acc) * batch_size

            # FID
            if pred_type == 'image':
                pred_imgs = model_output['model_out'].view(-1, opt.resolution, opt.resolution, 3).permute(0, 3, 1, 2) * 0.5 + 0.5
                gt_imgs = gt['img'].view(-1, opt.resolution, opt.resolution, 3).permute(0, 3, 1, 2) * 0.5 + 0.5
                if total_steps == 0:
                    grid_samples = torchvision.utils.make_grid(pred_imgs, nrow=int(math.sqrt(pred_imgs.size(0))))
                    torchvision.utils.save_image(grid_samples, os.path.join(opt.checkpoint_path, f'samples{opt.split}.png'))            
                    grid_gts = torchvision.utils.make_grid(gt_imgs, nrow=int(math.sqrt(gt_imgs.size(0))))
                    torchvision.utils.save_image(grid_gts, os.path.join(opt.checkpoint_path, f'gts{opt.split}.png'))

            if opt.eval and pred_type == 'image' and len(opt.context_tag) == 0:
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

            description = f'mse_loss:{all_losses/steps:.4f} PSNR:{all_psnr/steps:.2f} Ctx-mean:{float(context_params.mean()):.8f}'
            if opt.use_meta_sgd:
                description += f' LR-mean:{float(torch.abs(meta_sgd_inner).mean()):.8f}'
            if opt.use_lr_scheduler:
                description += f' LR:{scheduler.get_last_lr()[-1]:.4e}'
            if pred_type == 'voxel':
                description += f' acc:{all_acc/steps:.4f}'
            if opt.sparsity_inner or opt.sparsity_outer:
                l1_mean = float(model_output['l1_mean'])
                description += f' sparsity_l1:{l1_mean:.4f}'
            pbar.set_description(description)
            pbar.update(1)

            if torch.isnan(train_loss):
                with open(os.path.join(model_dir, 'train.log'), 'a') as f:
                    f.write('exit due to NaN')
                exit()

            # optimization procedure during training
            if not opt.eval:
                # compute gradient for shared parameters
                losses_all = losses['img_loss']
                if opt.sparsity_outer:
                    losses_all = losses_all + opt.sparsity_outer * model_output['l1_mean']
                task_grad = torch.autograd.grad(losses_all, model.get_parameters())

                # add to meta-gradient
                for g in range(len(task_grad)):
                    meta_grad[g] += task_grad[g].detach()

                optim.zero_grad()

                # set gradients of parameters manually
                for c, param in enumerate(model.get_parameters()):
                    param.grad = meta_grad[c]

                if opt.clip_grad:
                    torch.nn.utils.clip_grad_norm_(model.get_parameters(), max_norm=1.)

                optim.step()

            else:
                list_context_params.append(context_params.view(context_params.size(0),-1).detach().cpu())

            if pred_type == 'scene':
                with torch.no_grad():
                    if epoch % opt.epoch_for_full_rendering == 0 and step == 0:
                        model_input_eval = {key: value.cuda() for key, value in model_input_eval.items()}
                        gt_eval = {key: value.cuda() for key, value in gt_eval.items()}
                        model_output_full = model(model_input_eval, context_params)
                        model_output_full = nerf_volume_rendering(model_output_full, opt, 'all')
                        save_rendering_output(model_output_full, gt_eval, opt, 
                                            os.path.join(rendering_dir, f'E{epoch}_S{step}.png'))

        if not opt.eval:
            if ((epoch % opt.save_freq == 0) or (epoch == opt.num_epochs - 1) ):
                torch.save(model.state_dict(), os.path.join(checkpoints_dir, f'epoch{epoch}.pth'))

            if opt.use_lr_scheduler:
                scheduler.step()

            with open(os.path.join(model_dir, 'train.log'), 'a') as f:    
                f.write(f'[epoch {epoch}]' + description + '\n')

        else:
            loaded_epoch = opt.model_path.split('epoch')[-1].split('.pth')[0]
            if len(loaded_epoch) == 0:
                loaded_epoch = str(opt.ckpt_epoch)

            if len(opt.context_tag) == 0:
                context_tag = ''
            else:
                context_tag = '-' + opt.context_tag

            if opt.split == 'train':
                context_file = f'context{context_tag}-epoch{loaded_epoch}.pth'
            else:
                context_file = f'context{context_tag}-epoch{loaded_epoch}-{opt.split}.pth'

            torch.save(torch.cat(list_context_params, dim=0), os.path.join(contexts_dir, context_file))

            description = f'mse_loss:{all_losses/steps:.4f} PSNR:{all_psnr/steps:.2f}'

            if pred_type == 'image' and len(opt.context_tag) == 0:
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

            if pred_type == 'voxel':
                description += f' acc:{all_acc/steps:.4f}'                

            print(description)

            with open(os.path.join(model_dir, 'eval.log'), 'a') as f:
                f.write(f'Dataset: {opt.dataset} Split: {opt.split}')
                print(opt, file=f)
                f.write(description + '\n')
