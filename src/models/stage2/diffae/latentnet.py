import math
from dataclasses import dataclass
from enum import Enum
from typing import NamedTuple, Tuple, Optional
import copy
import itertools

import numpy as np
import torch
from torch import nn
from torch.nn import init
from omegaconf import OmegaConf

from .blocks import *
from .nn import timestep_embedding

import pdb

def get_act(act_type):
    if act_type == 'silu':
        return torch.nn.SiLU()
    elif act_type == 'relu':       
        return torch.nn.ReLU()
    elif act_type == 'leakyrelu':        
        return nn.LeakyReLU()
    elif act_type == 'sine':
        return Sine()
    elif act_type == 'tanh':
        return torch.nn.Tanh()
    else:
        return torch.nn.Identity()    

SCALE_SINE = 2

class Sine(nn.Module):
    def __init(self):
        super().__init__()

    def forward(self, input):
        # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
        return torch.sin(1 * input * SCALE_SINE)

def sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-np.sqrt(6 / num_input) / SCALE_SINE, np.sqrt(6 / num_input) / SCALE_SINE)


def first_layer_sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-1 / num_input, 1 / num_input)

class ParallelMLP(nn.Module):
    def __init__(self, weights: dict, conf: OmegaConf):
        super().__init__()
        mlps = []
        dims = []
        cnt = 0

        conf_old = copy.deepcopy(conf)
        for key in weights.keys():
            weight = weights[key]
            dim = weight.numel()
            factor = 2 if dim <= 1024 else 1
            conf.num_channels = dim
            conf.num_out_channels = dim
            conf.num_hid_channels = min(dim * factor, conf_old.num_hid_channels)
            mlps.append(MLPSkipNet(conf))
            dims.append((cnt, cnt+dim))
            cnt += dim
        
        self.mlps = nn.ModuleList(mlps)
        self.dims = dims
        self.num_channels = cnt

    def forward(self, x, t, **kwargs):
        outputs = []
        for (i0, i1), mlp in zip(self.dims, self.mlps):
            outputs.append(mlp(x[:, i0:i1], t))
        
        return torch.cat(outputs, dim=1)

class MLPEmbedNet(nn.Module):
    def __init__(self, weights: dict, conf: OmegaConf):
        super().__init__()
        
        inputs, outputs, indims, outdims = [], [], [], []
        for key in weights.keys():
            weight = weights[key]
            dim = weight.numel()
            indims.append(dim)
        
        embed_dim = conf.num_channels
        input_dim = sum(indims)
        for dim in indims:
            outdims.append(max(int(float(dim)*float(embed_dim)/float(input_dim)), 2))

        while sum(outdims) != embed_dim:
            if sum(outdims) > embed_dim:
                # find maximum element and minus 1
                max_idx = outdims.index(max(outdims))
                outdims[max_idx] = outdims[max_idx] - 1
            else:
                # find minimum element and add 1
                min_idx = outdims.index(min(outdims))
                outdims[min_idx] = outdims[min_idx] + 1

        for indim, outdim in zip(indims, outdims):
            inputs.append(nn.Linear(indim, outdim))
            outputs.append(nn.Linear(outdim, indim))

        self.inputs = nn.ModuleList(inputs)
        self.outputs = nn.ModuleList(outputs)
        self.indims = indims
        self.outdims = outdims
        self.main = MLPSkipNet(conf)
        self.num_channels = sum(indims)
        
    def forward(self, x, t, **kwargs):
        inputs = []
        cnt = 0
        for idim, inlayer in zip(self.indims, self.inputs):
            inputs.append(inlayer(x[:, cnt:cnt+idim]))
            cnt += idim        
        x = torch.cat(inputs, dim=1)

        x = self.main(x, t)
        
        outputs = []
        cnt = 0
        for edim, outlayer in zip(self.outdims, self.outputs):
            outputs.append(outlayer(x[:, cnt:cnt+edim]))
            cnt += edim

        return torch.cat(outputs, dim=1)
          
class MLPSkipNet(nn.Module):
    """
    concat x to hidden layers

    default MLP for the latent DPM in the paper!
    """
    def __init__(self, conf: OmegaConf):
        super().__init__()
        self.conf = conf
        self.groups = conf.num_hid_groups
        if conf.num_emb_channels == 0:
            conf.num_emb_channels = conf.num_channels

        self.cond_emb_channels = conf.num_emb_channels
        if conf.use_cond_index:
            self.index_embed = torch.nn.Embedding(conf.num_indices, conf.num_emb_channels)
            if conf.type_cond_index == 'concat':
                self.cond_emb_channels = conf.num_emb_channels * 2

        layers = []
        for i in range(conf.num_time_layers):
            if i == 0:
                a = conf.num_time_emb_channels
                b = conf.num_emb_channels
            else:
                a = conf.num_emb_channels
                b = conf.num_emb_channels
            layers.append(nn.Linear(a, b))
            if i < conf.num_time_layers - 1 or conf.time_last_act:
                layers.append(get_act(conf.activation))
        self.time_embed = nn.Sequential(*layers)

        self.layers = nn.ModuleList([])
        for i in range(conf.num_layers):
            if i == 0:
                act = conf.activation
                norm = conf.use_norm
                norm_type = conf.norm_type
                cond = True
                a, b = conf.num_channels, conf.num_hid_channels
                dropout = conf.dropout
                groups = 1 if conf.num_hid_groups > 0 else 0
            elif i == conf.num_layers - 1:
                act = ''
                norm = False
                norm_type = ''
                cond = False
                a, b = conf.num_hid_channels, conf.num_out_channels
                dropout = 0
                groups = 1 if conf.num_hid_groups > 0 else 0
            else:
                act = conf.activation
                norm = conf.use_norm
                norm_type = conf.norm_type
                cond = True
                a, b = conf.num_hid_channels, conf.num_hid_channels
                dropout = conf.dropout
                if conf.num_hid_groups == 0:
                    groups = 0
                elif conf.num_hid_groups > 0:
                    if conf.num_layers // 2 > i:
                        groups = conf.num_hid_groups // (2 ** (i-1))
                        if groups < 1:
                            groups = 1
                    elif conf.num_layers - i > conf.num_layers // 2:
                        groups = conf.num_hid_groups // (2 ** (conf.num_layers - i - 2))
                        if groups < 1:
                            groups = 1

            if i in conf.skip_layers:
                if conf.use_first_hid_layer:
                    a += conf.num_hid_channels
                else:
                    a += conf.num_channels
            
            self.layers.append(
                MLPLNAct(
                    a,
                    b,
                    norm=norm,
                    norm_type=norm_type,
                    activation=act,
                    cond_channels=self.cond_emb_channels,
                    use_cond=cond,
                    use_cond_norm=self.conf.type_cond_index == 'layernorm' and norm,
                    condition_bias=conf.condition_bias,
                    dropout=dropout,
                    groups=groups,
                ))
        self.last_act = get_act(conf.last_act)

        self.num_channels = conf.num_channels
        self.use_first_hid_layer = conf.use_first_hid_layer

    def forward(self, x, t, **kwargs):
        t = timestep_embedding(t, self.conf.num_time_emb_channels)
        cond = self.time_embed(t)
        
        cond_norm = None
        if self.conf.use_cond_index:
            if self.conf.type_cond_index == 'concat':
                cond2 = self.index_embed(kwargs['cond_idx'])
                cond = torch.cat([cond, cond2], dim=1)
            elif self.conf.type_cond_index == 'layernorm':
                cond_norm = self.index_embed(kwargs['cond_idx'])
        if self.groups > 0:
            x = x.unsqueeze(2)
        h = x
        for i in range(len(self.layers)):
            if i in self.conf.skip_layers:
                if self.use_first_hid_layer:
                    # injecting the first hidden layer into the hidden layers
                    h = h.view(h.size(0), 2, -1)
                    h1 = h1.view(h1.size(0), 2, -1)
                    h = torch.cat([h, h1], dim=1)
                    h = h.view(h.size(0),-1, 1)
                else:
                    # injecting input into the hidden layers
                    h = torch.cat([h, x], dim=1)
            h = self.layers[i].forward(x=h, cond=cond, cond_norm=cond_norm)
            if i == 0 and self.use_first_hid_layer:
                h1 = h
        h = self.last_act(h)
        if self.groups > 0:
            h = h.squeeze(2)
        return h

class MLPLNAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm: bool,
        norm_type: str, 
        use_cond: bool,
        use_cond_norm: bool,
        activation: str,
        cond_channels: int,
        condition_bias: float = 0,
        dropout: float = 0,
        groups: int = 0,
    ):
        super().__init__()
        self.activation = activation
        self.condition_bias = condition_bias
        self.use_cond = use_cond
        self.use_cond_norm = use_cond_norm
        self.act = get_act(activation)
        self.groups = groups 

        if groups == 0:
            self.linear = nn.Linear(in_channels, out_channels)
        elif groups > 0:
            self.linear = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, groups=groups)

        if self.use_cond:
            self.linear_emb = nn.Linear(cond_channels, out_channels)
            self.cond_layers = nn.Sequential(self.act, self.linear_emb)
        if norm and self.use_cond_norm:
            if norm_type == 'layernorm':
                self.norm = nn.LayerNorm(out_channels, elementwise_affine=False)
            elif norm_type == 'batchnorm':
                self.norm = nn.BatchNorm1d(out_channels, affine=False)
            elif norm_type == 'groupnorm':
                self.norm = nn.GroupNorm(16, out_channels, affine=False)
            self.norm_emb = nn.Linear(cond_channels, out_channels)
            self.cond_norm_layers = nn.Sequential(self.act, self.norm_emb)
        elif norm:
            if norm_type == 'layernorm':
                self.norm = nn.LayerNorm(out_channels)
            elif norm_type == 'batchnorm':
                self.norm = nn.BatchNorm1d(out_channels)
            elif norm_type == 'groupnorm':
                self.norm = nn.GroupNorm(16, out_channels)
        else:
            self.norm = nn.Identity()

        if dropout > 0:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = nn.Identity()

        self.init_weights()


    def init_weights(self):
        count = 0
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if self.activation == 'relu':
                    init.kaiming_normal_(module.weight,
                                         a=0,
                                         nonlinearity='relu')
                elif self.activation == 'lrelu':
                    init.kaiming_normal_(module.weight,
                                         a=0.2,
                                         nonlinearity='leaky_relu')
                elif self.activation == 'silu':
                    init.kaiming_normal_(module.weight,
                                         a=0,
                                         nonlinearity='relu')
                elif self.activation == 'sine':
                    # if count == 0:
                    #     first_layer_sine_init(module.weight)
                    # else:
                    sine_init(module.weight)

                else:
                    # leave it as default
                    pass

                count += 1 

    def forward(self, x, cond=None, cond_norm=None):
        
        x = self.linear(x)
        if self.use_cond:
            # (n, c) or (n, c * 2)
            cond = self.cond_layers(cond)
            cond = (cond, None)

            # scale shift first
            if self.groups == 0:
                x = x * (self.condition_bias + cond[0])
            elif self.groups > 0:
                x = x * (self.condition_bias + cond[0].unsqueeze(2))

            if cond[1] is not None:
                if self.groups == 0:
                    x = x + cond[1]
                elif self.groups > 0:
                    x = x + cond[1].unsqueeze(2)
            # then norm
            x = self.norm(x)
        else:
            # no condition
            x = self.norm(x)

        if self.use_cond_norm and self.norm:
            # (n, c) or (n, c * 2)
            cond_norm = self.cond_norm_layers(cond_norm)
            cond_norm = (cond_norm, None)

            # scale shift first
            x = x * (self.condition_bias + cond_norm[0])
            if cond_norm[1] is not None:
                x = x + cond_norm[1]

        x = self.act(x)
        x = self.dropout(x)
        return x


class FunctaMLP(nn.Module):
    """
    Residual MLP architecture introduced in functa (ICML 2022)
    """
    def __init__(self, conf: OmegaConf):
        super().__init__()
        self.conf = conf
        self.num_groups = conf.num_hid_groups
        self.num_hid_channels = conf.num_hid_channels
        self.num_channels = conf.num_channels # for sampling
        self.activation = conf.activation
        self.dropout = conf.dropout
        self.net_type = conf.net_type
        self.skip_type = 'skip' if 'skip' in self.net_type else 'unet'
        self.num_time_emb_channels = conf.num_time_emb_channels

        self.use_cond_index = conf.use_cond_index

        assert conf.num_layers % 2 == 0
        self.depth_blocks = conf.num_layers // 2

        layers = []
        for i in range(conf.num_time_layers):
            time_chn = self.num_time_emb_channels
            layers.append(nn.Linear(time_chn, time_chn))
            if i < conf.num_time_layers - 1 or conf.time_last_act:
                layers.append(get_act(conf.activation))
        self.block_time = nn.Sequential(*layers)

        self.blocks = nn.ModuleList([])
        for i in range(conf.num_layers):
            if self.use_cond_index:
                self.blocks.append(DoubleAffineResMLPBlock(self.num_hid_channels,
                                                           self.num_time_emb_channels,
                                                           self.num_groups,
                                                           self.activation,
                                                           self.dropout))
            elif self.net_type == 'functa-affine':
                self.blocks.append(AffineResMLPBlock(self.num_hid_channels,
                                                     self.num_time_emb_channels,
                                                     self.num_groups,
                                                     self.activation,
                                                     self.dropout))
            else:
                self.blocks.append(ResMLPBlock(self.num_hid_channels,
                                               self.num_time_emb_channels,
                                               self.num_groups,
                                               self.activation,
                                               self.dropout))

        self.block_input = nn.Linear(self.num_channels, self.num_hid_channels)
        self.block_output = nn.Sequential(nn.GroupNorm(self.num_groups, self.num_hid_channels),
                                          get_act(self.activation),
                                          nn.Linear(self.num_hid_channels, self.num_channels))

        if self.use_cond_index:
            self.index_embed = torch.nn.Embedding(conf.num_indices, self.num_hid_channels)

        self.init_weights()


    def init_weights(self):
        # blocks
        for blk in self.blocks:
            blk.init_weights()
        
        # input
        nn.init.kaiming_normal_(self.block_input.weight)
        init.zeros_(self.block_input.bias)

        # output
        for module in self.block_output.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                #init.zeros_(module.weight)
                init.zeros_(module.bias)
        
        for module in self.block_time.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                init.zeros_(module.bias)

    def forward(self, x, t, **kwargs):
        t_embed = timestep_embedding(t, self.num_time_emb_channels)
        t_embed = self.block_time(t_embed)
        
        if self.use_cond_index:
            idx_embed = self.index_embed(kwargs['cond_idx'])
        else:
            idx_embed = None

        if self.skip_type == 'unet':
            return self.forward_unet(x, t_embed, idx_embed)
       
        elif self.skip_type == 'skip':
            return self.forward_skip(x, t_embed, idx_embed)

    def forward_unet(self, x, t_embed, idx_embed):
        h_list = []

        # upstream
        h = self.block_input(x)
        for i in range(self.depth_blocks):
            h_list.append(h)
            if self.use_cond_index:
                h = self.blocks[i].forward(x=h, t=t_embed, idx=idx_embed)
            else:
                h = self.blocks[i].forward(x=h, t=t_embed)

        # downstream
        for i in range(self.depth_blocks):
            idx = self.depth_blocks + i
            if self.use_cond_index:
                h = self.blocks[idx].forward(x=h + h_list[-i-1], t=t_embed, idx=idx_embed)
            else:
                h = self.blocks[idx].forward(x=h + h_list[-i-1], t=t_embed)

        return self.block_output(h)
    
    def forward_skip(self, x, t_embed, idx_embed):
        # upstream
        x = self.block_input(x)
        for i in range(self.depth_blocks*2):
            if i > 0:
                h = h + x
            else:
                h = x
            if self.use_cond_index:
                h = self.blocks[i].forward(x=h, t=t_embed, idx=idx_embed)
            else:
                h = self.blocks[i].forward(x=h, t=t_embed)

        return self.block_output(h)

class ResMLPBlock(nn.Module):
    def __init__(
        self,
        num_channels: int,
        num_time_channels: int,
        num_groups: int,
        activation: str,
        dropout: float = 0,
    ):
        super().__init__()       
        self.block1 = nn.Sequential(nn.GroupNorm(num_groups, num_channels),
                                    get_act(activation),
                                    nn.Linear(num_channels, num_channels))

        self.block2 = nn.Sequential(nn.GroupNorm(num_groups, num_channels),
                                    get_act(activation),
                                    nn.Dropout(p=dropout),
                                    nn.Linear(num_channels, num_channels))

        self.blockt = nn.Sequential(get_act(activation),
                                    nn.Linear(num_time_channels, num_channels))

        self.init_weights()

    def init_weights(self):        
        for module in itertools.chain(self.block1.modules(), self.blockt.modules()):
            if isinstance(module, nn.Linear):
                init.kaiming_normal_(module.weight)
                init.zeros_(module.bias)
                # init.kaiming_normal_(module.weight,
                #                      a=0,
                #                      nonlinearity='relu')

        for module in self.block2.modules():
            if isinstance(module, nn.Linear):
                init.zeros_(module.weight)
                init.zeros_(module.bias)
                #init.kaiming_normal_(module.weight)

    def forward(self, x, t, cond=None):
        
        h1 = self.block1(x)
        ht = self.blockt(t)
        h2 = self.block2(h1 + ht)

        return h2 + x


class AffineResMLPBlock(nn.Module):
    def __init__(
        self,
        num_channels: int,
        num_time_channels: int,
        num_groups: int,
        activation: str,
        dropout: float = 0,
    ):
        super().__init__()       
        self.group1 = nn.GroupNorm(num_groups, num_channels, affine=False)
        self.block1 = nn.Sequential(get_act(activation),
                                    nn.Linear(num_channels, num_channels))
        # self.block1 = nn.Sequential(nn.GroupNorm(num_groups, num_channels),
        #                             get_act(activation),
        #                             nn.Linear(num_channels, num_channels))
        
        self.group2 = nn.GroupNorm(num_groups, num_channels, affine=False)
        self.block2 = nn.Sequential(get_act(activation),
                                    nn.Dropout(p=dropout),
                                    nn.Linear(num_channels, num_channels))

        self.blockt1 = nn.Sequential(get_act(activation),
                                    nn.Linear(num_time_channels, num_channels))

        self.blockt2 = nn.Sequential(get_act(activation),
                                    nn.Linear(num_time_channels, num_channels))

        # self.blockt3 = nn.Sequential(get_act(activation),
        #                             nn.Linear(num_time_channels, num_channels))

        # self.blockt4 = nn.Sequential(get_act(activation),
        #                             nn.Linear(num_time_channels, num_channels))


        self.init_weights()

    def init_weights(self):        
        #for module in itertools.chain(self.block1.modules(), self.blockt1.modules(), self.blockt2.modules(), self.blockt3.modules(), self.blockt4.modules()):
        for module in itertools.chain(self.block1.modules(), self.blockt1.modules(), self.blockt2.modules()):
            if isinstance(module, nn.Linear):
                init.kaiming_normal_(module.weight)
                init.zeros_(module.bias)
                # init.kaiming_normal_(module.weight,
                #                      a=0,
                #                      nonlinearity='relu')

        for module in self.block2.modules():
            if isinstance(module, nn.Linear):
                #init.zeros_(module.weight)
                init.zeros_(module.bias)
                init.kaiming_normal_(module.weight)

    def forward(self, x, t, cond=None):        
        h = self.block1(x)
        h = self.group2(h)
        h_std = self.blockt1(t)
        h_bias = self.blockt2(t)
                
        h1 = h * (h_std + 1) + h_bias # conditional bias 

        h2 = self.block2(h1)

        return h2 + x


class AffineTransform(nn.Module):
    def __init__(
        self,
        num_channels: int,
        num_input_channels: int,
        activation: str,
        offset: float = 1.0, 
    ):
        super().__init__()

        self.std = nn.Sequential(get_act(activation),
                                 nn.Linear(num_input_channels, num_channels))
 
        self.bias = nn.Sequential(get_act(activation),
                                  nn.Linear(num_input_channels, num_channels))
        self.offset = offset

    def forward(self, x, cond):
        return x * (self.std(cond) + self.offset) + self.bias(cond)


class DoubleAffineResMLPBlock(nn.Module):
    def __init__(
        self,
        num_channels: int,
        num_time_channels: int,
        num_groups: int,
        activation: str,
        dropout: float = 0,
    ):
        super().__init__()
        self.group1 = nn.GroupNorm(num_groups, num_channels, affine=False)
        self.block1 = nn.Sequential(get_act(activation),
                                    nn.Linear(num_channels, num_channels))
        
        self.group2 = nn.GroupNorm(num_groups, num_channels, affine=False)
        self.block2 = nn.Sequential(get_act(activation),
                                    nn.Dropout(p=dropout),
                                    nn.Linear(num_channels, num_channels))

        self.affine_idx = AffineTransform(num_channels, num_channels, activation)
        self.affine_time = AffineTransform(num_channels, num_time_channels, activation)

        self.init_weights()

    def init_weights(self):        
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.kaiming_normal_(module.weight)
                init.zeros_(module.bias)

    def forward(self, x, t, idx):
        h = self.group1(x)
        h = self.affine_idx(h, idx)
        h = self.block1(h)

        h = self.group2(h)
        h = self.affine_time(h, t)
        h = self.block2(h)

        return h + x

class AffineCondResMLPBlock(nn.Module):
    def __init__(
        self,
        num_channels: int,
        num_time_channels: int,
        num_groups: int,
        activation: str,
        dropout: float = 0,
    ):
        super().__init__()       
        self.block1 = nn.Sequential(nn.GroupNorm(num_groups, num_channels),
                                    get_act(activation),
                                    nn.Linear(num_channels, num_channels))
        
        self.block2 = nn.Sequential(nn.GroupNorm(num_groups, num_channels),
                                    get_act(activation),
                                    nn.Dropout(p=dropout),
                                    nn.Linear(num_channels, num_channels))

        self.affine_time = AffineTransform(num_channels, num_time_channels, activation)

        self.init_weights()

    def init_weights(self):        
        for module in itertools.chain(self.block1.modules(), self.affine_time.modules(), self.block2.modules()):
            if isinstance(module, nn.Linear):
                init.kaiming_normal_(module.weight)
                init.zeros_(module.bias)

    def forward(self, x, t, cond=None):        
        h = self.block1(x)
        h = self.affine_time(h, t)
        h = self.block2(h)

        return h + x