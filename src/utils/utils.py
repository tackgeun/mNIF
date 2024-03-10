import os
import random
import math
import logging

from scipy import linalg
import numpy as np

import torch

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def cond_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def logging_model_size(model, logger):
    if logger is None:
        return
    logger.info(
        "[OPTION: ALL] #params: %.4fM", sum(p.numel() for p in model.parameters()) / 1e6
    )
    logger.info(
        "[OPTION: Trainable] #params: %.4fM", sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    )


def image_mse(mask, model_output, gt):
    if mask is None:
        return {'img_loss': ((model_output['model_out'] - gt['img']) ** 2).mean()}
    else:
        return {'img_loss': (mask * (model_output['model_out'] - gt['img']) ** 2).mean()}

def compute_psnr(signal, gt):
    mse = max(float(torch.mean((signal-gt)**2)), 1e-8)
    psnr = float(-10 * math.log10(mse))
    return psnr

# https://github.com/yilundu/gem/blob/main/experiment_scripts/gen_imnet_autodecoder.py#L32
def sample_points_triangle(vertices, triangles, num_of_points):
    epsilon = 1e-6
    triangle_area_list = np.zeros([len(triangles)],np.float32)
    triangle_normal_list = np.zeros([len(triangles),3],np.float32)
    for i in range(len(triangles)):
        #area = |u x v|/2 = |u||v|sin(uv)/2
        a,b,c = vertices[triangles[i,1]]-vertices[triangles[i,0]]
        x,y,z = vertices[triangles[i,2]]-vertices[triangles[i,0]]
        ti = b*z-c*y
        tj = c*x-a*z
        tk = a*y-b*x
        area2 = math.sqrt(ti*ti+tj*tj+tk*tk)
        if area2<epsilon:
            triangle_area_list[i] = 0
            triangle_normal_list[i,0] = 0
            triangle_normal_list[i,1] = 0
            triangle_normal_list[i,2] = 0
        else:
            triangle_area_list[i] = area2
            triangle_normal_list[i,0] = ti/area2
            triangle_normal_list[i,1] = tj/area2
            triangle_normal_list[i,2] = tk/area2
    
    triangle_area_sum = np.sum(triangle_area_list)
    sample_prob_list = (num_of_points/triangle_area_sum)*triangle_area_list

    triangle_index_list = np.arange(len(triangles))

    point_normal_list = np.zeros([num_of_points,6],np.float32)
    count = 0
    watchdog = 0

    while(count<num_of_points):
        np.random.shuffle(triangle_index_list)
        watchdog += 1
        if watchdog>100:
            print("infinite loop here!")
            return point_normal_list
        for i in range(len(triangle_index_list)):
            if count>=num_of_points: break
            dxb = triangle_index_list[i]
            prob = sample_prob_list[dxb]
            prob_i = int(prob)
            prob_f = prob-prob_i
            if np.random.random()<prob_f:
                prob_i += 1
            normal_direction = triangle_normal_list[dxb]
            u = vertices[triangles[dxb,1]]-vertices[triangles[dxb,0]]
            v = vertices[triangles[dxb,2]]-vertices[triangles[dxb,0]]
            base = vertices[triangles[dxb,0]]
            for j in range(prob_i):
                #sample a point here:
                u_x = np.random.random()
                v_y = np.random.random()
                if u_x+v_y>=1:
                    u_x = 1-u_x
                    v_y = 1-v_y
                ppp = u*u_x+v*v_y+base

                point_normal_list[count,:3] = ppp
                point_normal_list[count,3:] = normal_direction
                count += 1
                if count>=num_of_points: break

    return point_normal_list


# https://github.com/yilundu/gem/blob/main/experiment_scripts/gen_imnet_autodecoder.py#L99
def write_ply_triangle(name, vertices, triangles):
        fout = open(name, 'w')
        fout.write("ply\n")
        fout.write("format ascii 1.0\n")
        fout.write("element vertex "+str(len(vertices))+"\n")
        fout.write("property float x\n")
        fout.write("property float y\n")
        fout.write("property float z\n")
        fout.write("element face "+str(len(triangles))+"\n")
        fout.write("property list uchar int vertex_index\n")
        fout.write("end_header\n")
        for ii in range(len(vertices)):
                fout.write(str(vertices[ii,0])+" "+str(vertices[ii,1])+" "+str(vertices[ii,2])+"\n")
        for ii in range(len(triangles)):
                fout.write("3 "+str(triangles[ii,0])+" "+str(triangles[ii,1])+" "+str(triangles[ii,2])+"\n")
        fout.close()

def frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        logging.warning(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)

def mean_covar_numpy(xs):
    if isinstance(xs, torch.Tensor):
        xs = xs.cpu().numpy()
    return np.mean(xs, axis=0), np.cov(xs, rowvar=False)