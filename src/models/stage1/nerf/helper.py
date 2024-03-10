import torch
import torch.nn.functional as F
import torchvision 
import numpy as np

# baseline and functa
from .nerf_utils import *
# mip-nerf
from .ray_utils import sample_along_rays_mip
from .ray_utils import volumetric_rendering as volumetric_rendering_mip

'''
nerf_utils.py: general nerf functions
helper.py : helping functions for our work
'''

MAX_DENSITY = 10. 

def get_rays_batch(H, W, focal, c2w, compute_radii=False):
    # TODO: faster
    bsz = c2w.shape[0]
    all_rays = []
    for i in range(bsz):
        # per image
        cam_rays = get_rays(H, W, focal[i], c2w[i], compute_radii=compute_radii)
        all_rays.append(cam_rays)
    results = torch.stack(all_rays, 1)
    #results = torch.stack(list(map(lambda f, c: get_rays(H,W,f,c), focal, c2w)), 1)
    return results

def get_samples_for_nerf(model_input, gt, opt, view_sampling=True, pixel_sampling=True, view_num=None):
    all_scene_rays = []
    all_scene_rgb = []
    all_scene_idx = []
    ALL_VIEW = gt['img'].shape[1]
    bsz = gt['img'].shape[0]

    for i_batch in range(bsz):
        focal = model_input['focal'][i_batch] #(ALL_VIEW)
        c2w =  model_input['c2w'][i_batch] #(ALL_VIEW,4,4)
        idx_ = model_input['idx'][i_batch] #1
        rgb = gt['img'][i_batch] #(ALL_VIEW,3,H,W)

        # sampling view
        if view_sampling and opt.subsampled_views > 0:
            NV = opt.subsampled_views if view_num is None else view_num
            view_inds = np.random.choice(ALL_VIEW, NV)
            focal = focal.repeat(NV)
            c2w = c2w[view_inds, :,:]
            rgb = rgb[view_inds, :, :, :]
        else:
            focal = focal.repeat(ALL_VIEW)
            NV = ALL_VIEW   
           
        # get origin & direction of all pixels
        compute_radii = opt.rendering_type == 'mip-nerf'
        cam_rays = get_rays_batch(opt.H, opt.W, focal, c2w, compute_radii=compute_radii) #(2 or 3,NV,H,W,3)

        # sampling [H,W] indices
        NM = 3 if compute_radii else 2

        assert cam_rays.size(0) == NM and cam_rays.size(1) == NV and cam_rays.size(4) == 3

        assert rgb.size(0) == NV and rgb.size(1) == 3
        cam_rays = cam_rays.permute(0, 1, 4, 2, 3) #(2,NV,3,H,W)
        cam_rays = cam_rays.reshape(NM, NV, 3, -1) #(2,NV,3,H*W)
        rgb = rgb.reshape(NV,3,-1) #(NV,3,H*W)

        if pixel_sampling and opt.subsampled_pixels > 0:
            if 'bbox' in model_input.keys():
                pass
            else:
                pix_inds = np.random.choice(opt.H * opt.W, opt.subsampled_pixels)

            cam_rays = cam_rays[:,:,:,pix_inds].permute(0, 1, 3, 2) #(2,NV,NP,3)
            rgb = rgb[:,:,pix_inds].permute(0, 2, 1) #(NV,NP,3)
        else:
            cam_rays = cam_rays.permute(0, 1, 3, 2) #(2,NV,NP,3)
            rgb = rgb.permute(0, 2, 1) #(NV,NP,3)
 
        all_scene_rays.append(cam_rays)
        all_scene_rgb.append(rgb)
    
    all_scene_rgb = torch.stack(all_scene_rgb, 0).reshape(bsz, -1, 3) #(B*NV*NP,3)
    all_scene_rays = torch.stack(all_scene_rays, 1).reshape(NM, -1, 3) #(3,B*NV*NP,3)
    if opt.rendering_type == 'mip-nerf':
        t_vals, (coords, coords_covs) = sample_along_rays_mip(all_scene_rays, opt)
    else:
        t_vals, coords = sample_along_rays(all_scene_rays, opt)
    
    coords = coords.reshape(bsz, -1, 3)

    # Model input
    model_input['coords'] = coords
    model_input['rays_d'] = all_scene_rays[1]
    model_input['t_vals'] = t_vals
    del model_input['focal']
    del model_input['c2w']

    # GT
    gt['img'] = all_scene_rgb
    return model_input, gt


def get_test_samples_for_nerf(model_input, view_inds, opt, focal_ratio=1.0):
    all_scene_rays = []
    NV = 1
    if focal_ratio != 1.0:
        focal = model_input['focal'] * focal_ratio #(ALL_VIEW)
    else:
        focal = model_input['focal'] #(ALL_VIEW)
    #focal = focal.repeat(ALL_VIEW)
    c2w =  model_input['c2w'][0] #(ALL_VIEW,4,4)

    #focal = focal.repeat(NV)
    _c2w = c2w[view_inds:view_inds+1, :,:]
    _focal = focal.unsqueeze(0)
    
    # get origin & direction of all pixels
    # cam_rays = get_rays_batch(opt.H, opt.W, _focal, _c2w) #(2,NV,H,W,3)
    compute_radii = opt.rendering_type == 'mip-nerf'
    cam_rays = get_rays_batch(opt.H, opt.W, _focal, _c2w, compute_radii=compute_radii) #(2 or 3,NV,H,W,3)

    # sampling [H,W] indices
    #compute_radii = opt.rendering_type == 'mip-nerf':

    NM = 3 if compute_radii else 2
    # if compute_radii:
    #     NM = 3
    # else:
    #     NM = 2
    assert cam_rays.size(0) == NM and cam_rays.size(1) == NV and cam_rays.size(4) == 3
    #assert cam_rays.size(0) == 2 and cam_rays.size(1) == NV and cam_rays.size(4) == 3
    cam_rays = cam_rays.permute(0, 1, 4, 2, 3) #(2,NV,3,H,W)
    cam_rays = cam_rays.reshape(NM, NV, 3, -1) #(2,NV,3,H*W)
    cam_rays = cam_rays.permute(0, 1, 3, 2) #(2,NV,NP,3)

    all_scene_rays.append(cam_rays)
    
    all_scene_rays = torch.stack(all_scene_rays, 1).reshape(NM, -1, 3) #(2,B*NV*NP,3)
    if opt.rendering_type == 'mip-nerf':
        t_vals, (coords, coords_cov) = sample_along_rays_mip(all_scene_rays, opt)
    else:
        t_vals, coords = sample_along_rays(all_scene_rays, opt)
    
    coords = coords.reshape(1, -1, 3)

    # Model input
    model_input['coords'] = coords
    model_input['rays_d'] = all_scene_rays[1]
    model_input['t_vals'] = t_vals
    del model_input['focal']
    del model_input['c2w']

    return model_input

def nerf_volume_rendering(prediction, opt, out_type='rgb'):
    pred_rgb, pred_density = prediction['model_out'][..., :3], prediction['model_out'][..., -1:]

    bsz = pred_rgb.shape[0]
    # rgb activation
    pred_rgb = pred_rgb.reshape(-1, opt.num_samples_per_ray+1, 3)
    if opt.rgb_activation == 'sigmoid':
        pred_rgb = torch.sigmoid(pred_rgb)
    elif opt.rgb_activation == 'relu':
        pred_rgb = F.relu(pred_rgb)
    elif 'sine' in opt.rgb_activation:
        w0 = float(opt.rgb_activation.split('sine')[-1])
        pred_rgb = torch.sin(w0*pred_rgb)
    elif opt.rgb_activation == 'no_use':
        pass
    else:
        raise Exception("check rgb activation")

    # density activation
    pred_density = pred_density.reshape(-1, opt.num_samples_per_ray+1, 1)    
    if opt.density_activation == 'elu':
        pred_density = F.elu(pred_density, alpha=0.1) + 0.1
        pred_density = torch.clip(pred_density, 0, MAX_DENSITY)
    elif opt.density_activation == 'relu':
        pred_density = F.relu(pred_density)
    elif opt.density_activation == 'leakyrelu':
        pred_density = F.leaky_relu(pred_density) + 0.1
    elif opt.density_activation == 'shift1':
        pred_density = torch.clip(pred_density + 1.0, 0, MAX_DENSITY)
    elif opt.density_activation == 'shift':
        pred_density = pred_density + 0.5
        pred_density = torch.clip(pred_density, 0, MAX_DENSITY)
    elif opt.density_activation == 'sine5+shift0.9':
        pred_density = torch.sin(5.0 * pred_density)
        pred_density = torch.clip(pred_density + 0.9, 0, MAX_DENSITY)
    elif opt.density_activation == 'sine5+shift1':
        pred_density = torch.sin(5.0 * pred_density)
        pred_density = torch.clip(pred_density + 1, 0, MAX_DENSITY)
    elif opt.density_activation == 'sine5+shift1.1':
        pred_density = torch.sin(5.0 * pred_density)
        pred_density = torch.clip(pred_density + 1.1, 0, MAX_DENSITY)
    elif opt.density_activation == 'sine5+shift0.5+elu':
        pred_density = torch.sin(5.0 * pred_density) + 0.5
        pred_density = F.elu(pred_density, alpha=0.1) + 0.1
        pred_density = torch.clip(pred_density, 0, MAX_DENSITY)
    elif opt.density_activation == 'sine5+shift1+elu':
        pred_density = torch.sin(5.0 * pred_density) + 1.0
        pred_density = F.elu(pred_density, alpha=0.1) + 0.1
        pred_density = torch.clip(pred_density, 0, MAX_DENSITY)
    elif 'elu+scale' in opt.density_activation:
        scale = float(opt.density_activation.split('elu+scale')[-1])
        pred_density = pred_density * scale
        pred_density = F.elu(pred_density, alpha=0.1) + 0.1
        pred_density = torch.clip(pred_density, 0, MAX_DENSITY)
    elif opt.density_activation == 'no_use':
        pass        
    else:
        raise Exception("check density activation")
    
    t_vals, rays_d = prediction['model_in']['t_vals'], prediction['model_in']['rays_d']
    if opt.rendering_type == 'functa':
        color, acc, depth, weight = volumetric_rendering_functa(pred_rgb, pred_density, t_vals, rays_d, opt.white_bkgd)
    elif opt.rendering_type == 'mip-nerf':
        color, acc, depth, weight = volumetric_rendering_mip(pred_rgb, pred_density, t_vals, rays_d, opt.white_bkgd)
    else:
        color, acc, depth, weight = volumetric_rendering(pred_rgb, pred_density, t_vals, rays_d, opt.white_bkgd)
    

    # reshape
    color = color.reshape(bsz, -1, 3)
    depth = depth.reshape(bsz, -1)
    acc = acc.reshape(bsz, -1)
    if out_type == 'all':
        prediction['model_out'] = {
                                   'rgb': color,
                                   'depth': depth,
                                   'acc': acc,
                                  }
    else:
        prediction['model_out'] = color
    return prediction

def save_rendering_output(model_output, gt, opt, image_path, max_num=-1):
    save_gt = gt['img'].reshape(-1, opt.H, opt.W, 3).permute(0,3,1,2).detach().cpu()
    pred_rgb = model_output['model_out']['rgb'].reshape(-1, opt.H, opt.W, 3).permute(0,3,1,2).detach().cpu()
    pred_depth = model_output['model_out']['depth'].reshape(-1, opt.H, opt.W, 1).permute(0,3,1,2).detach().cpu()
    pred_acc = model_output['model_out']['acc'].reshape(-1, opt.H, opt.W, 1).permute(0,3,1,2).detach().cpu()
    pred_depth = ((pred_depth-pred_depth.min())/(pred_depth.max()-pred_depth.min())*2-1).repeat(1,3,1,1)
    pred_acc = pred_acc.repeat(1,3,1,1)
    combined_image = torch.cat((save_gt, pred_rgb, pred_depth, pred_acc), -1)
    if max_num > 0:
        save_num = min(combined_image.size(0), max_num)
        combined_image = combined_image[:save_num]

    combined_image = torchvision.utils.make_grid(combined_image, nrow=1)
    torchvision.utils.save_image(combined_image, image_path)

def split_dict(dict_, start, end, num_unit):
    output_dict = {}
    for k, v in  dict_.items():
        if k not in ['coords']:
            output_dict[k] = v
        else:
            if end >= v.size(1):
                output_dict[k] = v[:,start:,:]
            else:               
                output_dict[k] = v[:,start:end,:]
    return output_dict, start+num_unit, end+num_unit
