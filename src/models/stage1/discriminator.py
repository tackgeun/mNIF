# ------------------------------------------------------------------------------------
# HQ-Transformer
# Copyright (c) 2022 KakaoBrain. All Rights Reserved.
# Licensed under the MIT License [see LICENSE for details]
# ------------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import lpips

from .CIPS.Discriminators import Discriminator
from .modules.utils import weights_init

import pdb

def adopt_weight(weight, global_step, threshold=0, value=0.):
    if global_step < threshold:
        weight = value
    return weight


def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def vanilla_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (
        torch.mean(torch.nn.functional.softplus(-logits_real)) +
        torch.mean(torch.nn.functional.softplus(logits_fake)))
    return d_loss


# def d_r1_loss(real_pred, real_img):
#     grad_real, = autograd.grad(
#         outputs=real_pred.sum(), inputs=real_img, create_graph=True
#     )
#     grad_penalty = grad_real.pow(2).view(grad_real.shape[0], -1).sum(1).mean()

#     return grad_penalty


class VQLPIPSWithDiscriminator(nn.Module):
    def __init__(self,
                 codebook_weight=0.0,
                 pixelloss_weight=1.0,
                 perceptual_weight=0.0,
                 disc_weight=1.0, 
                 disc_start=0, disc_in_channels=3, disc_factor=1.0,
                 disc_loss="vanilla"):
        super().__init__()
        assert disc_loss in ["hinge", "vanilla"]
        
        self.pixel_weight = pixelloss_weight

        self.perceptual_weight = perceptual_weight
        if self.perceptual_weight > 0.0:
            self.perceptual_loss = lpips.LPIPS(net='vgg', spatial=True)

        self.codebook_weight = codebook_weight

        self.discriminator = Discriminator(size=256, input_size=disc_in_channels)        
        self.discriminator_iter_start = disc_start
        if disc_loss == "hinge":
            self.disc_loss = hinge_d_loss
        elif disc_loss == "vanilla":
            self.disc_loss = vanilla_d_loss
        else:
            raise ValueError(f"Unknown GAN loss '{disc_loss}'.")
        print(f"VQLPIPSWithDiscriminator running with {disc_loss} loss.")
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
    

    def forward(self, codebook_loss, inputs, reconstructions, optimizer_idx,
                global_step, last_layer=None, split="train"):
        _inputs, _recons = inputs.contiguous(), reconstructions.contiguous()

        log = dict()
        
        if self.discriminator_weight == 0.0 or (self.discriminator_weight > 0.0 and optimizer_idx == 0):
            rec_loss = F.mse_loss(_inputs, _recons)
            log["{}/rec_loss".format(split)] = rec_loss.detach().mean()
            if self.pixel_weight > 0.0:
                loss = rec_loss
            else:
                loss = 0

            if self.perceptual_weight > 0.0:
                p_loss = self.perceptual_loss(_inputs, _recons).mean()
                log["{}/p_loss".format(split)] = p_loss.detach()
                loss += self.perceptual_weight * p_loss

            if codebook_loss is not None:
                log["{}/quant_loss".format(split)] = codebook_loss.detach().mean()

        # now the GAN part
        if self.discriminator_weight > 0.0:
            if optimizer_idx == 0:
                g_loss, logits_fake = self.forward_logits_fake(reconstructions)
                d_weight = self.discriminator_weight
                disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)

                loss += d_weight * disc_factor * g_loss
                if codebook_loss is not None:
                    loss += self.codebook_weight * codebook_loss.mean()

                log["{}/total_loss".format(split)] = loss.clone().detach().mean()
                log["{}/d_weight".format(split)] = d_weight
                log["{}/disc_factor".format(split)] = torch.tensor(disc_factor)
                log["{}/g_loss".format(split)] = g_loss.detach().mean()

                return loss, log

            if optimizer_idx == 1:
                d_loss, logits_real, logits_fake = self.forward_logits_real_fake(inputs,
                                                                                 reconstructions,
                                                                                 global_step)

                log["{}/disc_loss".format(split)] = d_loss.clone().detach().mean()
                log["{}/logits_real".format(split)] = logits_real.detach().mean()
                log["{}/logits_fake".format(split)] = logits_fake.detach().mean()
                      
                return d_loss, log
        else:
            log["{}/total_loss".format(split)] = loss.clone().detach().mean()
            return loss, log

    def forward_logits_fake(self, reconstructions):
        # generator update
        logits_fake = self.discriminator(reconstructions.contiguous())

        g_loss = -torch.mean(logits_fake)
        return g_loss, logits_fake

    def forward_logits_real_fake(self, inputs, reconstructions, global_step):
        # second pass for discriminator update
        logits_real = self.discriminator(inputs.contiguous().detach())
        logits_fake = self.discriminator(reconstructions.contiguous().detach())

        disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)

        d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)

        return d_loss, logits_real, logits_fake
