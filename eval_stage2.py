import os
import math
import copy
import gc 

import argparse
from tqdm.autonotebook import tqdm
import numpy as np

import torch
from torch.utils.data.dataloader import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

import torchvision
from src.datasets import CelebAHQ, INRWeightWrapper, ShapeNetVoxel, SRNDatasetsLMDB
from src.models import build_model_stage2
from src.utils.logger import LatentDDPMLogger
from src.utils.config2 import build_config
from src.utils.utils import cond_mkdir, frechet_distance, mean_covar_numpy
from src.utils.inception import get_inception_model
from src.utils.prdc import compute_prdc
from src.models.stage1.siren import meta_modules


parser = argparse.ArgumentParser()

parser.add_argument('-r', '--result_path', type=str, default=None, required=True)
parser.add_argument('-e', '--stage2_epoch', type=int, default=999)
parser.add_argument('-c', '--stage1_path', type=str, default=None)
parser.add_argument('--stage1_epoch', type=int, default=399)

parser.add_argument('--eval', default=True)
parser.add_argument('--dataset_root', type=str, default='datasets')
parser.add_argument('--context_tag', type=str, default='')
parser.add_argument('--dname', type=str, default="")

parser.add_argument('--bicubic', action='store_true')
parser.add_argument('--antialias', action='store_true')

parser.add_argument('--world_size', default=-1, type=int, help='number of nodes for distributed training')
parser.add_argument('--local_rank', default=-1, type=int, help='local rank for distributed training')
parser.add_argument('--node_rank', default=-1, type=int, help='node rank for distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')
parser.add_argument('--n-nodes', type=int, default=1)
parser.add_argument('--n-gpus', type=int, default=1)
parser.add_argument('--local_batch_size', type=int, default=32)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--reduce_sample', type=int, default=0)
parser.add_argument('--num_samples', type=int, default=50000)
parser.add_argument('--use_ddim', action='store_true')
parser.add_argument('--sampling_step', default=0, type=int)
parser.add_argument('--save_images', action='store_true')

parser.add_argument('--image_resolution', default=-1, type=int)
parser.add_argument('--pred_resolution', default=-1, type=int)
parser.add_argument('--upsampling_method', default='')

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

    # Build data modules
    dname = config.dataset.dataset.lower()

    if args.pred_resolution > 0:
        pred_res = args.pred_resolution
    else:
        pred_res = config.stage2.hparams_metainr.image_resolution
    if args.image_resolution > 0:
        input_res = args.image_resolution
    else:
        input_res = config.dataset.image_resolution

    if 'shapenet' in dname:
        from src.utils.utils import sample_points_triangle
        data_type = 'voxel'
        dataset = ShapeNetVoxel(split='test', dataset_root=args.dataset_root)

        print('we use 8762 samples which is equivalent size with test set samples.')
        args.num_samples = 8762
        
        feat_path = os.path.join(args.stage1_path, 'contexts', f'context-epoch{args.stage1_epoch}.pth')
        feat = torch.load(feat_path)
        dataset_affine = (feat.mean(dim=0), feat.std(dim=0))

        points_result_path = os.path.join(args.result_path, 'triangle_points')
        cond_mkdir(points_result_path)

    elif 'srncars' in dname:
        data_type = 'scene'
        dataset = SRNDatasetsLMDB(category='cars', split='test', opt=config.dataset.hparams_nerf, dataset_root=args.dataset_root)

        inr_path = os.path.join(args.stage1_path, 'metainits', f'epoch{args.stage1_epoch}.pth')
        feat_path = os.path.join(args.stage1_path, 'contexts', f'context-epoch{args.stage1_epoch}.pth')
        feat = torch.load(feat_path)
        dataset_affine = (feat.mean(dim=0), feat.std(dim=0))

        args.num_samples = len(dataset) if args.num_samples > len(dataset) else args.num_samples
        for view in range(0, 251):
            cond_mkdir(os.path.join(args.result_path, 'views', f'view{view}'))
        
    else:
        data_type = 'image'
        if 'celeba' in dname:
            dataset = CelebAHQ(split='train', downsampled=False, dataset_root=args.dataset_root)
        elif 'cifar10' in dname:
            assert config.dataset.image_resolution == 32
            dataset = torchvision.datasets.CIFAR10(root=args.dataset_root, train=True, download=True)
        else:
            raise ValueError()

        if args.bicubic:
            resampling = 'bicubic'
        else:
            resampling = 'bilinear'

        dataset = INRWeightWrapper(dataset,
                                   sidelength=input_res,
                                   checkpoint_path=args.stage1_path,
                                   checkpoint_step=args.stage1_epoch,
                                   reduce_sample=args.reduce_sample,
                                   feed_type=config.stage2.feat_type,
                                   context_tag=args.context_tag,
                                   istuple='cifar10' in dname,
                                   antialias=args.antialias,
                                   resampling=resampling)
        dataset_affine = dataset.affine

        cond_mkdir(os.path.join(args.result_path, 'images'))

    batch_size = 1 if data_type == 'scene' else args.local_batch_size
    dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size, pin_memory=True, num_workers=8)

    # Build a model
    config.stage2.hparams_metainr.init_path = os.path.join(args.stage1_path, 'metainits', f'epoch{args.stage1_epoch}.pth')
    model = build_model_stage2(cfg_stage2=config.stage2, cfg_opt=config.optimizer, affine=dataset_affine)

    ckpt_path = os.path.join(result_path, 'ckpt', config.dataset.dataset + f"-lddimepoch={args.stage2_epoch:02d}.ckpt")
    ckpt = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    model.load_state_dict(ckpt, strict=True)

    if data_type == 'scene':
        print('load INR decoder from stage1_path')
        model.decoder.load_state_dict(torch.load(inr_path, map_location="cpu"))

    model = model.cuda()

    if args.sampling_step > 0:
        sampling_step = args.sampling_step
    else:
        sampling_step = model._diffusion_kwargs.timestep_respacing

    if args.use_ddim:
        sample_method = f'ddim{sampling_step}'
    else:
        sample_method = f'{sampling_step}'
        
    #sample_method = model._diffusion_kwargs.timestep_respacing if not args.use_ddim else 'ddim100'
    if data_type == 'image':
        eval_type = 'fid+prdc'
        if eval_type == 'fid+prdc':
            from src.utils.inception import get_inception_model
            from src.utils.prdc import compute_prdc
            inception = get_inception_model().cuda()
            inception.eval()

            if len(args.dname) > 0:
                gt_act_path = f'{args.dataset_root}/{args.dname}-train-res{input_res}-activations.pth'
            else:
                gt_act_path = f'{args.dataset_root}/{dname}-train-res{input_res}-activations.pth'

            if not os.path.exists(gt_act_path):
                print(f'compute activation and save to {gt_act_path}')
                acts = []
                with tqdm(total=len(dataloader)) as pbar:
                    for iter, (model_input, gt) in enumerate(dataloader):
                        #gt_imgs = gt['img'].view(-1, input_res, input_res, 3).permute(0, 3, 2, 1)
                        gt_imgs = (gt['img'] + 1.0)/2.0
                        for gt_img in gt_imgs:
                            act = inception(gt_img.unsqueeze(0).cuda())
                            acts.append(act.cpu())

                        description = f'[{iter+1}/{len(dataloader)}] compute dataset activations'
                        pbar.set_description(description)
                        pbar.update(1)

                        if iter == 0:
                            grid_samples = torchvision.utils.make_grid(gt_imgs, nrow=int(math.sqrt(args.local_batch_size)))
                            torchvision.utils.save_image(grid_samples, os.path.join(args.result_path, 'gt_sample.png'))

                        if iter > args.num_samples:
                            break

                acts = torch.cat(acts, dim=0)
                torch.save(acts, gt_act_path)
            else:
                print(f'load activation from {gt_act_path}')
                acts = torch.load(gt_act_path)

            acts = acts.numpy()

            with tqdm(total=args.num_samples) as pbar:
                counts = 0

                gen_acts = []
                while counts < args.num_samples:
                    batch_size = min(args.local_batch_size, args.num_samples-counts)

                    samples = model.sample(batch_size, sample_method, resolution=pred_res)
                    samples = torch.stack(samples, dim=0)
                    if pred_res != input_res:
                        samples = torch.nn.functional.interpolate(samples, size=input_res, mode=args.upsampling_method)

                    samples = torch.clamp((samples + 1.0)/2.0, 0, 1)
                    for sample in samples:
                        act = inception(sample.unsqueeze(0))
                        gen_acts.append(act.detach().cpu())

                    if counts == 0:
                        grid_samples = torchvision.utils.make_grid(samples, nrow=int(math.sqrt(args.local_batch_size)))
                        torchvision.utils.save_image(grid_samples, os.path.join(args.result_path, f'grid_sample{sample_method}.png'))
                    
                    if args.save_images:
                        for i, sample in enumerate(samples):
                            torchvision.utils.save_image(sample, os.path.join(args.result_path, 'images', f'image{counts + i}.png'))

                    counts += batch_size

                    description = f'[{counts}/{args.num_samples}] sampling'
                    pbar.set_description(description)
                    pbar.update(batch_size)

                gen_acts = torch.cat(gen_acts, dim=0).numpy()
                
                mu_gt, sigma_gt = mean_covar_numpy(acts)
                mu_gen, sigma_gen = mean_covar_numpy(gen_acts)

                fid = frechet_distance(mu_gt, sigma_gt, mu_gen, sigma_gen)
                prdc = compute_prdc(acts, gen_acts, nearest_k=3)            
                precision, recall = prdc['precision'], prdc['recall']
                description = f'[0-1] FID: {fid}\tPrecision: {precision}\tRecall: {recall}'
                print(description)


    elif data_type == 'voxel':
        import mcubes
        from src.utils.metric_voxel import coverage, minimum_mathing_distance
        input_res = config.stage2.hparams_metainr.image_resolution

        count = 0
        gt_points = []
        gt_points_path = "datasets/shapenet/all_vox256_img/triangle_points_2048_test.npy"
        if not os.path.exists(gt_points_path):
            with tqdm(total=len(dataloader)) as pbar:
                for iter, (model_input, gt) in enumerate(dataloader):

                    sdf_values = gt['img']
                    for sdf_value in sdf_values:
                        sample = sdf_value.reshape(input_res, input_res, input_res).detach().cpu().numpy()

                        vertices, triangle = mcubes.marching_cubes(sample, 0.5)
                        vertices = (vertices.astype(np.float32) - 0.5) / 64 - 0.5

                        points = sample_points_triangle(vertices, triangle, 2048)
                        gt_points.append(points)
                        
                        count += 1

                    description = f'[{iter+1}/{len(dataloader)}] compute dataset stats'
                    pbar.set_description(description)
                    pbar.update(1)
            np.save(gt_points_path, gt_points)
        else:
            gt_points = np.load(gt_points_path)
            gt_points = np.stack(gt_points, axis=0)
            gt_points = gt_points[:, :, :3]
            if len(gt_points) > args.num_samples:
                gt_points = gt_points[0:args.num_samples]

        pred_points_path = os.path.join(args.result_path, 'triangle_points', f'points-epoch{args.stage2_epoch}')
        if not os.path.exists(pred_points_path):
            with tqdm(total=args.num_samples) as pbar:
                count = 0
                pred_points = []            
                while count < args.num_samples:
                    batch_size = min(args.local_batch_size, args.num_samples-count)
                    samples = model.sample(batch_size, sample_method, resolution=64, output_type='voxel')
                    for sample in samples:
                        sample = sample.reshape(input_res, input_res, input_res).detach().cpu().numpy()

                        vertices, triangle = mcubes.marching_cubes(sample, 0.5)
                        vertices = (vertices.astype(np.float32) - 0.5) / 64 - 0.5

                        points = sample_points_triangle(vertices, triangle, 2048)
                        pred_points.append(points)

                    count += batch_size

                    description = f'[{count}/{args.num_samples}] sampling'
                    pbar.set_description(description)
                    pbar.update(batch_size)

                pred_points = np.stack(pred_points, axis=0)
                np.save(pred_points_path, pred_points)
                pred_points = pred_points[:, :, :3]
        else:
            pred_points = np.load(pred_points_path)
            pred_points = pred_points[:, :, :3]

        mmd, matched_dists = minimum_mathing_distance(pred_points, gt_points, 1024, normalize=True)
        cov, matched_ids = coverage(pred_points, gt_points, 1024, normalize=True)
        print(f'Coverage: {cov:.4f} MMD:{mmd:.4f}')
        description = f'Coverage: {cov:.4f} MMD:{mmd:.4f}'

    elif data_type == 'scene':
        from src.models.stage1.nerf.helper import *
        
        NUM_VIEW = 251

        with torch.no_grad():
            inception = get_inception_model().cuda()
            inception.eval()

            gen_acts, acts = [], []
            gt_act_path = f'{args.dataset_root}/{dname}-test-res{input_res}-activations.pth'
            if not os.path.exists(gt_act_path):
                compute_gt_feat = True
            else:
                print(f'load activation from {gt_act_path}')
                acts = torch.load(gt_act_path)
                compute_gt_feat = False
            
            with tqdm(total=len(dataloader)*251) as pbar:
                for iter, (model_input, gt) in enumerate(dataloader):

                    model_input_eval, gt_eval = get_samples_for_nerf(copy.deepcopy(model_input), copy.deepcopy(gt), config.dataset.hparams_nerf, view_num=1, pixel_sampling=False)
                    model_input_eval = {key: value.cuda()   for key, value in model_input_eval.items()}
                    gt_eval = {key: value.cuda() for key, value in gt_eval.items()}

                    ## For real scene
                    gt_imgs = gt['img'].view(-1, 3, input_res, input_res)
                    gt_imgs = (gt_imgs + 1.0)/2.0

                    xs_fid = (torch.clamp(gt_imgs, 0.0, 1.0) * 255.).to(dtype=torch.uint8)
                    if compute_gt_feat:
                        act = inception(gt_imgs.cuda())
                        acts.append(act.cpu())

                    if iter == 0:
                        grid_samples = torchvision.utils.make_grid(gt_imgs, nrow=16)
                        torchvision.utils.save_image(grid_samples, os.path.join(args.result_path, 'gt_sample.png'))
                        assert float(gt_imgs.max()) <= 1.0
                        assert float(gt_imgs.min()) >= 0

                    del xs_fid

                    # Generated scene
                    samples = model.sample_latent(1, sample_method)
                    sample = list(samples)[0]                   
                    scenes = []

                    if pred_res != input_res:
                        config.dataset.hparams_nerf.resolution = pred_res
                        config.dataset.hparams_nerf.H = pred_res
                        config.dataset.hparams_nerf.W = pred_res

                    for view_ind in range(0, NUM_VIEW):
                        # generate images
                        _model_input = get_test_samples_for_nerf(copy.deepcopy(model_input), view_ind, config.dataset.hparams_nerf, focal_ratio=pred_res/input_res)
                        _model_input = {key: value.cuda() for key, value in _model_input.items()}

                        outputs = model.decoder(_model_input, sample)
                        outputs = nerf_volume_rendering(outputs, config.dataset.hparams_nerf, 'all')                        
                        scene = outputs['model_out']['rgb']

                        scene = scene.view(pred_res, pred_res, 3)
                        scene = scene.permute(2, 0, 1)

                        if pred_res != input_res:
                            scene = F.interpolate(scene.unsqueeze(0), size=(input_res, input_res), mode='bilinear')
                            scene = scene.squeeze(0)

                        if config.dataset.hparams_nerf.rgb_activation == 'sigmoid':
                            scene_clamped = torch.clamp(scene, 0, 1)
                        else:
                            scene_clamped = torch.clamp((scene + 1)/2, 0, 1)

                        act = inception(scene_clamped.unsqueeze(0))
                        gen_acts.append(act.detach().cpu())

                        scene_fid = (scene_clamped * 255.).to(dtype=torch.uint8)
                        
                        description = f'[{(iter)*NUM_VIEW + view_ind + 1}/{251*len(dataloader)}] compute dataset stats'
                        pbar.set_description(description)
                        pbar.update(1)

                        if args.save_images:
                            scene_path = os.path.join(args.result_path, 'views', f'view{view_ind}', f'image{iter}.png')
                            torchvision.utils.save_image(scene_clamped, scene_path)
                        if iter == 0:
                            scenes.append(scene_clamped.cpu())

                    if pred_res != input_res:
                        config.dataset.hparams_nerf.resolution = input_res
                        config.dataset.hparams_nerf.H = input_res
                        config.dataset.hparams_nerf.W = input_res

                    if iter == 0:
                        grid_samples = torchvision.utils.make_grid(torch.stack(scenes, dim=0), nrow=int(math.ceil(math.sqrt(NUM_VIEW))))
                        torchvision.utils.save_image(grid_samples, os.path.join(args.result_path, f'grid_sample251-{sample_method}.png'))

                    if iter > args.num_samples:
                        break

        description = ''

        if compute_gt_feat:
            acts = torch.cat(acts, dim=0)
            torch.save(acts, gt_act_path)

        acts = acts.numpy()
        gen_acts = torch.cat(gen_acts, dim=0).numpy()
        
        mu_gt, sigma_gt = mean_covar_numpy(acts)
        mu_gen, sigma_gen = mean_covar_numpy(gen_acts)

        fid = frechet_distance(mu_gt, sigma_gt, mu_gen, sigma_gen)
        description += f'FID: {fid}'
        print(description)


    with open(os.path.join(args.result_path, 'eval.log'), 'a') as f:
        f.write(f'Dataset: {dname}')
        print(args, file=f)
        f.write(description + '\n')            

