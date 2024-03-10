# ------------------------------------------------------------------------------------
# HQ-Transformer
# Copyright (c) 2022 KakaoBrain. All Rights Reserved.
# Licensed under the MIT License [see LICENSE for details]
# ------------------------------------------------------------------------------------

import torch
import torch.nn as nn
from typing import Tuple, List, Optional
from omegaconf import OmegaConf
from einops.layers.torch import Rearrange
import itertools
import torchvision

import clip
from .CIPS.GeneratorsCIPS import CIPSskip

import pdb

class ResMLP(nn.Module):
    def __init__(self,nin,nhidden,nblock):
        super(ResMLP, self).__init__()
        
        M = []
        for _ in range(nblock):
            M.append(nn.Sequential(nn.Linear(nin, nhidden), nn.LeakyReLU(), nn.Linear(nhidden, nin)))

        self.M = nn.ModuleList(M)

    def forward(self,x):
        for m in self.M:
            x = x + m(x)
        
        return x

class AutoEncoderGenerator(torch.nn.Module):
    def __init__(self,
                 hparams_enc: OmegaConf,
                 hparams_dec: OmegaConf,
                 device='cuda'):
        super().__init__()

        self.resmlp = None
        if hparams_enc.type =="ViT-B/32":
            self.encoder, _ = clip.load(hparams_enc.type, device=device)
            self.preprocess = lambda img: torch.nn.functional.interpolate(img, size=224)
            dim_input = 512
            self.enc_type = 'clip-flatten'
        elif 'densenet121' in hparams_enc.type:
            densenet = torchvision.models.densenet121(pretrained=True)
            net = list(densenet.children())[:-1] # last batchnorm w.o. classifier
            dim_input = 1024
            dim_spatial = 8*8

            self.enc_type = 'densenet-conv'
            self.encoder = torch.nn.Sequential(*net)
            self.preprocess = None

        elif 'resnet' in hparams_enc.type:
            if 'resnet18' in hparams_enc.type:
                resnet = torchvision.models.resnet18(pretrained=True)
                net = list(resnet.children())[:-2] # conv
                dim_input = 512
                dim_spatial = 8*8

            elif 'resnet50' in hparams_enc.type:
                resnet = torchvision.models.resnet50(pretrained=True)
                if 'layer2' in hparams_enc.type:
                    net = list(resnet.children())[:-4]
                    dim_input = 512
                    dim_spatial = 32*32
                elif 'layer3' in hparams_enc.type:
                    net = list(resnet.children())[:-3]
                    dim_input = 1024
                    dim_spatial = 16*16
                else:
                    net = list(resnet.children())[:-2]
                    dim_input = 2048
                    dim_spatial = 8*8

            if 'conv' in hparams_enc.type:
                self.enc_type = 'resnet-conv'
                self.encoder = torch.nn.Sequential(*net)
                self.preprocess = None
            else: # flatten
                dim_input = dim_input * dim_spatial
                dim_spatial = 1*1
                self.enc_type = 'resnet-flatten'
                resnet.append(Rearrange('b c h w -> b (c h w)'))
                self.encoder = torch.nn.Sequential(*net)
                self.preprocess = None

        else:
            assert(False)

        if hparams_dec.type == 'CIPSskip':
            self.decoder = CIPSskip(**hparams_dec)

        dim_output = hparams_dec.style_dim

        if 'conv' in self.enc_type:
            embed = []
            if 'groupnorm' in hparams_enc.type_init_norm:
                n_group = int(hparams_enc.type_init_norm.split('groupnorm')[-1])
                embed.append(torch.nn.GroupNorm(n_group, dim_input))

            # conv layers
            for ci in range(0, hparams_enc.num_embed_conv):
                if ci == 0:
                    cin = dim_input
                else:
                    if hparams_enc.dim_embed_conv > 0:
                        cin = hparams_enc.dim_embed_conv
                    else:
                        cin = dim_input

                if hparams_enc.dim_embed_conv > 0:
                    cout = hparams_enc.dim_embed_conv
                else:
                    cout = dim_input

                embed.append(torch.nn.Conv2d(cin, cout, 1))
                embed.append(torch.nn.GroupNorm(hparams_enc.num_groups, cout))
                embed.append(torch.nn.LeakyReLU())

            embed.pop() # remove last activation

            # flatten layer
            embed.append(Rearrange('b c h w -> b (c h w)'))

            num_flatten_linear = hparams_enc.num_flatten_linear
            dim_flatten_linear = hparams_enc.dim_flatten_linear
            for li in range(0, num_flatten_linear):
                if li == 0:
                    lin = cout*dim_spatial
                else:
                    if dim_flatten_linear > 0:
                        lin = dim_flatten_linear
                    else:
                        lin = dim_output

                if li+1 == num_flatten_linear:
                    lout = dim_output
                else:
                    if dim_flatten_linear > 0:
                        lout = dim_flatten_linear
                    else:
                        lout = dim_output

                embed.append(torch.nn.Linear(lin, lout, 1))
                embed.append(torch.nn.LeakyReLU())

            embed.pop() # remove last activation

            num_res = hparams_enc.num_residual_embed
            dim_res = hparams_enc.dim_residual_embed
            if num_res > 0:
                if dim_res == 0:
                    dim_res = lout
                self.resmlp = ResMLP(lout, dim_res, num_res)

            self.embed = nn.Sequential(*embed)

        elif 'flatten' in self.enc_type:
            self.embed = nn.Linear(dim_input, dim_output, 1)

        self.res = hparams_dec.size
        self.coords = None

    def forward(self, x, coords=None):
        latent = self.encode(x)
        recon = self.decode(latent, coords)
        return recon, latent

    def encode(self, x: torch.FloatTensor) -> torch.FloatTensor:
        if self.preprocess is not None:
            x = self.preprocess(x)
        if 'clip' in self.enc_type:
            h = self.encoder.encode_image(x)
        else:
            h = self.encoder(x)

        h = self.embed(h.float())

        if self.resmlp is not None:
            h = self.resmlp(h)
        return h

    def decode(self, latent: torch.FloatTensor, coords: Optional[torch.FloatTensor]=None) -> torch.FloatTensor:
        # if coords is not given, use the saved one.
        if coords is None:
            if self.coords is None:
                update_coords = True
            elif latent.size(0) != self.coords.size(0):
                update_coords = True
            else:
                update_coords = False

            if update_coords:
                b = latent.size(0)
                self.coords = self.convert_to_coord_format(b, self.res, self.res, integer_values=False)

            coords = self.coords

        recon, _ = self.decoder(coords, [latent])
        return recon

    # adopted from "tensor_transforms.py" CIPS
    def convert_to_coord_format(self, b, h, w, device='cuda', integer_values=False):
        if integer_values:
            x_channel = torch.arange(w, dtype=torch.float, device=device).view(1, 1, 1, -1).repeat(b, 1, w, 1)
            y_channel = torch.arange(h, dtype=torch.float, device=device).view(1, 1, -1, 1).repeat(b, 1, 1, h)
        else:
            x_channel = torch.linspace(-1, 1, w, device=device).view(1, 1, 1, -1).repeat(b, 1, w, 1)
            y_channel = torch.linspace(-1, 1, h, device=device).view(1, 1, -1, 1).repeat(b, 1, 1, h)
        return torch.cat((x_channel, y_channel), dim=1)

    def get_parameters(self):
        # fusing two generators by chain
        return itertools.chain(self.decoder.parameters(), self.embed.parameters())