import math
import torch
import numpy as np
import pdb
'''
NeRF-pytorch & NeRF-Factory
https://github.com/yenchenlin/nerf-pytorch/blob/master/run_nerf_helpers.py
https://github.com/kakaobrain/NeRF-Factory/blob/main/src/model/nerf/helper.py
'''
# data preparing
def get_rays(H, W, focal, c2w, padding=None, compute_radii=False):
    # pytorch's meshgrid has indexing='ij'
    if padding is not None:
        i, j = torch.meshgrid(torch.linspace(-padding, W-1+padding, W+2*padding), torch.linspace(-padding, H-1+padding, H+2*padding)) 
    else:
        i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))
    i = i.t().to(c2w.device)
    j = j.t().to(c2w.device)
    extra_shift = 0.5
    dirs = torch.stack([(i-W*.5+extra_shift)/focal, -(j-H*.5+extra_shift)/focal, -torch.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3,-1].expand(rays_d.shape)
    
    if compute_radii:
        dx = torch.sqrt(torch.sum((rays_d[:-1, :, :] - rays_d[1:, :, :]) ** 2, -1))
        dx = torch.cat([dx, dx[-2:-1, :]], 0)

        # Cut the distance in half, and then round it out so that it's
        # halfway between inscribed by / circumscribed about the pixel.
        radii = dx[..., None] * 2 / math.sqrt(12)
        return torch.stack((rays_o, rays_d, radii.repeat(1,1,3)), 0)
    else:
        return torch.stack((rays_o, rays_d), 0)

def cast_rays(t_vals, origins, directions):
    return origins[..., None, :] + t_vals[..., None] * directions[..., None, :]

def sample_along_rays(
    cam_rays, 
    configs,
):
    # get configs
    num_samples = configs.num_samples_per_ray
    near, far = configs.near, configs.far
    lindisp = configs.lindisp
    randomized = configs.randomized # noise

    rays_o, rays_d = cam_rays[0], cam_rays[1]
    bsz = rays_o.shape[0]
    
    t_vals = torch.linspace(0.0, 1.0, num_samples + 1, device=rays_o.device)
    if lindisp:
        t_vals = 1.0 / (1.0 / near * (1.0 - t_vals) + 1.0 / far * t_vals)
    else:
        t_vals = near * (1.0 - t_vals) + far * t_vals

    if randomized:
        mids = 0.5 * (t_vals[..., 1:] + t_vals[..., :-1])
        upper = torch.cat([mids, t_vals[..., -1:]], -1)
        lower = torch.cat([t_vals[..., :1], mids], -1)
        t_rand = torch.rand((bsz, num_samples + 1), device=rays_o.device)
        t_vals = lower + (upper - lower) * t_rand
    else:
        t_vals = torch.broadcast_to(t_vals, (bsz, num_samples + 1))

    coords = cast_rays(t_vals, rays_o, rays_d)
    return t_vals, coords


def volumetric_rendering(rgb, density, t_vals, dirs, white_bkgd):
    eps = 1e-10

    dists = torch.cat(
        [
            t_vals[..., 1:] - t_vals[..., :-1],
            torch.ones(t_vals[..., :1].shape, device=t_vals.device) * 1e10,
        ],
        dim=-1,
    )
    dists = dists * torch.norm(dirs[..., None, :], dim=-1)
    alpha = 1.0 - torch.exp(-density[..., 0] * dists)
    accum_prod = torch.cat(
        [
            torch.ones_like(alpha[..., :1]),
            torch.cumprod(1.0 - alpha[..., :-1] + eps, dim=-1),
        ],
        dim=-1,
    )

    weights = alpha * accum_prod

    comp_rgb = (weights[..., None] * rgb).sum(dim=-2)
    depth = (weights * t_vals).sum(dim=-1)
    acc = weights.sum(dim=-1)
    inv_eps = 1 / eps

    if white_bkgd:
        comp_rgb = comp_rgb + (1.0 - acc[..., None])

    return comp_rgb, acc, depth, weights

def volumetric_rendering_functa(rgb, density, t_vals, dirs, white_bkgd):
    eps = 1e-10
    distance_between_points = t_vals[..., 1:] - t_vals[..., :-1]
    dists = torch.cat(
        [
            distance_between_points,
            torch.ones(distance_between_points[..., :1].shape, device=t_vals.device) * 1e-3,
        ],
        dim=-1,
    )
    dists = dists * torch.norm(dirs[..., None, :], dim=-1)
    alpha = 1.0 - torch.exp(-density[..., 0] * dists)

    trans = torch.minimum(torch.ones_like(alpha), 1.0 - alpha + eps)
    trans = torch.cat(
        [
            torch.ones_like(trans[..., :1]),
            trans[..., :-1],
        ],
        dim=-1,
    )

    cum_trans = torch.cumprod(trans, dim=-1)

    weights = alpha * cum_trans

    comp_rgb = (weights[..., None] * rgb).sum(dim=-2)
    depth = (weights * t_vals).sum(dim=-1)
    acc = weights.sum(dim=-1)

    if white_bkgd:
        comp_rgb = comp_rgb + (1.0 - acc[..., None])

    return comp_rgb, acc, depth, weights