'''Modules for hypernetwork experiments, Paper Sec. 4.4
'''

import os
import math
import itertools
from collections import OrderedDict
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
import torchvision

from . import modules

import pdb


###############################
# Functa baseline and variants
###############################
class FiLM(nn.Module):
    """Applies a FiLM modulation: out = scale * in + shift.

    Notes:
        We currently initialize FiLM layers as the identity. However, this may not
        be optimal. In pi-GAN for example they initialize the layer with a random
        normal.
    """

    def __init__(self,
                f_in: int,
                modulate_scale: bool = True,
                modulate_shift: bool = True):
        """Constructor.

        Args:
        f_in: Number of input features.
        modulate_scale: If True, modulates scales.
        modulate_shift: If True, modulates shifts.
        """
        super().__init__()
        # Must modulate at least one of shift and scale
        assert modulate_scale or modulate_shift
        self.f_in = f_in
        # Initialize FiLM layers as identity
        self.scale = 1.
        self.shift = 0.
        if modulate_scale:
            self.scale = nn.Parameter(torch.ones(self.f_in))
        if modulate_shift:
            self.shift = nn.Parameter(torch.zeros(self.f_in))

    def forward(self, x):
        return self.scale * x + self.shift


class ModulatedSirenLayer(nn.Module):
    """Applies a linear layer followed by a modulation and sine activation."""

    def __init__(self,
                 f_in: int,
                 f_out: int,
                 w0: float = 1.,
                 is_first: bool = False,
                 is_last: bool = False,
                 modulate_scale: bool = True,
                 modulate_shift: bool = True,
                 apply_activation: bool = True):
        """Constructor.

        Args:
        f_in (int): Number of input features.
        f_out (int): Number of output features.
        w0 (float): Scale factor in sine activation.
        is_first (bool): Whether this is first layer of model.
        is_last (bool): Whether this is last layer of model.
        modulate_scale: If True, modulates scales.
        modulate_shift: If True, modulates shifts.
        apply_activation: If True, applies sine activation.
        """
        super().__init__()
        self.f_in = f_in
        self.f_out = f_out
        self.w0 = w0
        self.is_first = is_first
        self.is_last = is_last
        self.modulate_scale = modulate_scale
        self.modulate_shift = modulate_shift
        self.apply_activation = apply_activation
        # Follow initialization scheme from SIREN
        self.init_range = 1 / f_in if is_first else math.sqrt(6 / f_in) / w0

        self.w = torch.nn.Linear(self.f_in, self.f_out)

        if self.modulate_scale or self.modulate_shift:
            self.FiLM = FiLM(self.f_out,
                            modulate_scale=self.modulate_scale,
                            modulate_shift=self.modulate_shift)

        with torch.no_grad():
            # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
            self.w.weight.uniform_(-self.init_range, self.init_range)


    def forward(self, x):
        # Shape (n, f_in) -> (n, f_out)
        x = self.w(x)
        # Apply non-linearities
        if self.is_last:
        # We assume target data (e.g. RGB values of pixels) lies in [0, 1]. To
        # learn zero-centered features we therefore shift output by .5
            return x + .5
            #return x
        else:
        # Optionally apply modulation
            if self.modulate_scale or self.modulate_shift:
                x = self.FiLM(x)
            # Optionally apply activation
            if self.apply_activation:
                x = torch.sin(self.w0 * x)
            return x


class LatentVector(nn.Module):
    """Module that holds a latent vector.

    Notes:
    This module does not apply any transformation but simply stores a latent
    vector. This is to make sure that all data necessary to represent an image
    (or a NeRF scene or a video) is present in the model params. This also makes
    it easier to use the partition_params function.
    """

    def __init__(self, latent_dim: int, latent_init_scale: float = 0.0):
        """Constructor.

        Args:
        latent_dim: Dimension of latent vector.
        latent_init_scale: Scale at which to randomly initialize latent vector.
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.latent_init_scale = latent_init_scale
        # Initialize latent vector
        self.latent_vector = nn.Parameter(torch.zeros(latent_dim).uniform_(-latent_init_scale,latent_init_scale))

    def forward(self):
        return self.latent_vector


class LatentToModulation(nn.Module):
    """Function mapping latent vector to a set of modulations."""

    def __init__(self,
                latent_dim: int,
                latent_vector_type: str,
                layer_sizes: Tuple[int, ...],
                width: int,
                num_modulation_layers: int,
                modulate_scale: bool = True,
                modulate_shift: bool = True,
                zero_init_last: bool = False,
                batch_norm_init: bool = False,
                activation: str = 'relu'):
        """Constructor.

        Args:
        latent_dim: Dimension of latent vector (input of LatentToModulation
            network).
        layer_sizes: List of hidden layer sizes for MLP parameterizing the map
            from latent to modulations. Input dimension is inferred from latent_dim
            and output dimension is inferred from number of modulations.
        width: Width of each hidden layer in MLP of function rep.
        num_modulation_layers: Number of layers in MLP that contain modulations.
        modulate_scale: If True, returns scale modulations.
        modulate_shift: If True, returns shift modulations.
        activation: Activation function to use in MLP.
        """
        super().__init__()
        # Must modulate at least one of shift and scale
        assert modulate_scale or modulate_shift

        self.latent_dim = latent_dim
        self.latent_vector_type = latent_vector_type
        self.layer_sizes = tuple(layer_sizes)  # counteract XM that converts to list
        self.width = width
        self.num_modulation_layers = num_modulation_layers
        self.modulate_scale = modulate_scale
        self.modulate_shift = modulate_shift

        # MLP outputs all modulations. We apply modulations on every hidden unit
        # (i.e on width number of units) at every modulation layer.
        # At each of these we apply either a scale or a shift or both,
        # hence total output size is given by following formula
        self.modulations_per_unit = int(modulate_scale) + int(modulate_shift)
        self.modulations_per_layer = width * self.modulations_per_unit
        self.output_size = num_modulation_layers * self.modulations_per_layer

        # added configurations
        self.zero_init_last = zero_init_last
        self.batch_norm_init = batch_norm_init

        #self.mlp = torchvision.ops.MLP(hidden_channels=self.layer_sizes + (self.output_size,))
        layer_sizes = (self.latent_dim,) + self.layer_sizes + (self.output_size,)
        mlp = []

        if self.batch_norm_init:
            mlp.append(nn.BatchNorm1d(self.latent_dim, affine=False))

        for i in range(0, len(layer_sizes)-1):
            mlp.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
        self.mlp = nn.Sequential(*mlp)

        # TODO: support modulate_scale
        # only work for modulate_shift currently
        if self.zero_init_last:
            assert(not modulate_scale and modulate_shift)
            nn.init.zeros_(self.mlp[-1].weight)
            nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, latent_vector: torch.Tensor) -> Dict[int, Dict[str, torch.Tensor]]:
        modulations = self.mlp(latent_vector)
        # Partition modulations into scales and shifts at every layer
        outputs = {}

        if self.latent_vector_type == 'shared':
            for i in range(self.num_modulation_layers):
                single_layer_modulations = {}
                # Note that we add 1 to scales so that outputs of MLP will be centered
                # (since scale = 1 corresponds to identity function)
                if self.modulate_scale and self.modulate_shift:
                    start = 2 * self.width * i
                    single_layer_modulations['scale'] = modulations[start:start +
                                                                    self.width] + 1
                    single_layer_modulations['shift'] = modulations[start +
                                                                    self.width:start +
                                                                    2 * self.width]
                elif self.modulate_scale:
                    start = self.width * i
                    single_layer_modulations['scale'] = modulations[start:start +
                                                                    self.width] + 1
                elif self.modulate_shift:
                    start = self.width * i
                    single_layer_modulations['shift'] = modulations[start:start +
                                                                    self.width]
                outputs[i] = single_layer_modulations

        elif self.latent_vector_type == 'instance':
            for i in range(self.num_modulation_layers):
                single_layer_modulations = {}
                # Note that we add 1 to scales so that outputs of MLP will be centered
                # (since scale = 1 corresponds to identity function)
                if self.modulate_scale and self.modulate_shift:
                    start = 2 * self.width * i
                    single_layer_modulations['scale'] = modulations[:, start:start +
                                                                    self.width].unsqueeze(1) + 1
                    single_layer_modulations['shift'] = modulations[:, start +
                                                                    self.width:start +
                                                                    2 * self.width].unsqueeze(1)
                elif self.modulate_scale:
                    start = self.width * i
                    single_layer_modulations['scale'] = modulations[:, start:start +
                                                                    self.width].unsqueeze(1) + 1
                elif self.modulate_shift:
                    start = self.width * i
                    single_layer_modulations['shift'] = modulations[:, start:start +
                                                                    self.width].unsqueeze(1) # (batch, 1, width)
                outputs[i] = single_layer_modulations
        return outputs


class MetaSGDLrs(nn.Module):
    """Module storing learning rates for meta-SGD.

    Notes:
    This module does not apply any transformation but simply stores the learning
    rates. Since we also learn the learning rates we treat them the same as
    model params.
    """

    def __init__(self,
               num_lrs: int,
               lrs_init_range: Tuple[float, float] = (0.005, 0.1),
               lrs_clip_range: Tuple[float, float] = (-5., 5.),
               unsqueeze_first=False):

        """Constructor.

        Args:
        num_lrs: Number of learning rates to learn.
        lrs_init_range: Range from which initial learning rates will be
            uniformly sampled.
        lrs_clip_range: Range at which to clip learning rates. Default value will
            effectively avoid any clipping, but typically learning rates should
            be positive and small.
        """
        super().__init__()
        self.num_lrs = num_lrs
        self.lrs_init_range = lrs_init_range
        self.lrs_clip_range = lrs_clip_range
        # Initialize learning rates
        if unsqueeze_first:
            self.meta_sgd_lrs = torch.nn.Parameter(torch.zeros(1, self.num_lrs, 1).uniform_(self.lrs_init_range[0], self.lrs_init_range[1]),requires_grad=True)
        else:
            self.meta_sgd_lrs = torch.nn.Parameter(torch.zeros(self.num_lrs).uniform_(self.lrs_init_range[0], self.lrs_init_range[1]),requires_grad=True)

    def forward(self, step=0):
        # Clip learning rate values
        return self.meta_sgd_lrs.clamp(self.lrs_clip_range[0], self.lrs_clip_range[1])


class LatentModulatedSiren(nn.Module):
    """SIREN model with FiLM modulations generated from a latent vector."""
    def __init__(self,
                width: int = 256,
                depth: int = 5,
                in_channels: int = 2,
                out_channels: int = 3,
                latent_dim: int = 128,
                latent_vector_type: str = 'instance',
                layer_sizes: Tuple[int, ...] = (),
                w0: float = 30.,
                modulate_scale: bool = False,
                modulate_shift: bool = True,
                latent_init_scale: float = 0.01,
                use_meta_sgd: bool = False,
                meta_sgd_init_range: Tuple[float, float] = (0.005, 0.1),
                meta_sgd_clip_range: Tuple[float, float] = (0., 1.)):
        """Constructor.

        Args:
            width (int): Width of each hidden layer in MLP.
            depth (int): Number of layers in MLP.
            out_channels (int): Number of output channels.
            latent_dim: Dimension of latent vector (input of LatentToModulation
            network).
            layer_sizes: List of hidden layer sizes for MLP parameterizing the map
            from latent to modulations. Input dimension is inferred from latent_dim
            and output dimension is inferred from number of modulations.
            w0 (float): Scale factor in sine activation in first layer.
            modulate_scale: If True, modulates scales.
            modulate_shift: If True, modulates shifts.
            latent_init_scale: Scale at which to randomly initialize latent vector.
            use_meta_sgd: Whether to use meta-SGD.
            num_inner_steps: Number of inner loop steps.
            meta_sgd_init_range: Range from which initial meta_sgd learning rates will
            be uniformly sampled.
            meta_sgd_clip_range: Range at which to clip learning rates.
        """
        super().__init__()
        self.width = width
        self.depth = depth
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.latent_dim = latent_dim
        self.layer_sizes = layer_sizes
        self.w0 = w0
        self.modulate_scale = modulate_scale
        self.modulate_shift = modulate_shift
        self.latent_init_scale = latent_init_scale
        self.latent_vector_type = latent_vector_type
        self.use_meta_sgd = use_meta_sgd
        self.meta_sgd_init_range = meta_sgd_init_range
        self.meta_sgd_clip_range = meta_sgd_clip_range

        # Initialize meta-SGD learning rates
        if self.use_meta_sgd:
            self.meta_sgd_lrs = MetaSGDLrs(self.latent_dim,
                                           self.meta_sgd_init_range,
                                           self.meta_sgd_clip_range)

        # Initialize latent vector and map from latents to modulations
        #self.context_params = torch.zeros(latent_dim, device='cuda').uniform_(-latent_init_scale,latent_init_scale)
        self.latent_to_modulation = LatentToModulation(
            latent_dim=latent_dim,
            latent_vector_type=latent_vector_type,
            layer_sizes=layer_sizes,
            width=width,
            num_modulation_layers=depth-1,
            modulate_scale=modulate_scale,
            modulate_shift=modulate_shift)

        modsiren = [ModulatedSirenLayer(f_in=self.in_channels,
                                        f_out=self.width,
                                        is_first=True,
                                        w0=self.w0,
                                        modulate_scale=False,
                                        modulate_shift=False,
                                        apply_activation=False)]
        for i in range(0, self.depth-2):
            modsiren.append(ModulatedSirenLayer(f_in=self.width,
                                                f_out=self.width,
                                                w0=self.w0,
                                                modulate_scale=False,
                                                modulate_shift=False,
                                                apply_activation=False))

        modsiren.append(ModulatedSirenLayer(f_in=self.width,
                                            f_out=self.out_channels,
                                            is_last=True,
                                            w0=self.w0,
                                            modulate_scale=False,
                                            modulate_shift=False))
        
        self.modsiren = nn.ModuleList(modsiren)

    def shared_parameters(self):
        return itertools.chain(self.modsiren.parameters(), self.latent_to_modulation.parameters())

    def get_context_params(self, batch_size=None, eval_mode=False):
        if eval_mode:
            rand_params = torch.zeros(batch_size, self.latent_dim).cuda() #.to(self.device)
        else:
            rand_params = torch.zeros(batch_size, self.latent_dim).uniform_(-self.latent_init_scale,self.latent_init_scale).cuda() #.cuda()
        context_params = rand_params.detach()
        context_params.requires_grad_()

        #self.context_params = rand_params.detach()
        #self.context_params.requires_grad_()

        return context_params

    def modulate(self, x: torch.Tensor, modulations: Dict[str,torch.Tensor]) -> torch.Tensor:
        """Modulates input according to modulations.

        Args:
        x: Hidden features of MLP.
        modulations: Dict with keys 'scale' and 'shift' (or only one of them)
            containing modulations.

        Returns:
        Modulated vector.
        """
        if 'scale' in modulations:
            x = modulations['scale'] * x
        if 'shift' in modulations:
            x = x + modulations['shift']
        return x

    def forward(self, inputs: torch.Tensor, context_params: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Evaluates model at a batch of coordinates.

        Args:
        coords (Tensor): Tensor of coordinates. Should have shape (height, width, 2)
            for images and (depth/time, height, width, 3) for 3D shapes/videos.

        Returns:
        Output features at coords.
        """

        # Compute modulations based on latent vector
        if context_params is None:
            if 'context_params' in inputs:
                context_params = inputs['context_params']
            else:
                context_params = self.context_params
        modulations = self.latent_to_modulation(context_params)

        # Flatten coordinates
        coords = inputs['coords']

        # if self.latent_vector_type == 'shared':
        #     x = coords.view(-1, coords.shape[-1])
        # else:
        #     x = coords

        x = coords
        #print('len(self.modsiren)', len(self.modsiren), self.depth-1)
        # Hidden layers
        for i in range(0, self.depth-1):
            x = self.modsiren[i](x)
            x = self.modulate(x, modulations[i])
            x = torch.sin(self.w0 * x)

        # Final layer
        out = self.modsiren[-1](x)

        return {'model_in': inputs,
                'model_out': out}

        # if self.latent_vector_type == 'shared':
        #     out = out.view(coords.size(0), coords.size(1), self.out_channels) # Unflatten output

        # return {'model_in': input,
        #         'model_out': out, 
        #         'latent_vec': context_params}

    def get_parameters(self):
        return self.parameters()

    def get_named_parameters(self):
        return self.named_parameters()


class LinearMixtureINR(nn.Module):
    """K mixture of INR model"""

    def __init__(self,
                 width: int = 32,
                 depth: int = 4,
                 in_channels: int = 2,
                 out_channels: int = 3,
                 image_resolution: int = 64,
                 w0: float = 30.,                 
                 k_mixtures: int = 128,
                 mixture_type: str = 'all-layer',
                 embedding_type: str = 'none',
                 pred_type: str = 'image',
                 outermost_linear: bool = None,
                 use_latent_embedding: bool = False,
                 std_latent: float = 0.0,
                 latent_channels: int = 256,
                 latent_init_scale: Tuple[float, float] = (1.0 - 0.05, 1.0 + 0.05),
                 use_meta_sgd: bool = False,
                 meta_sgd_init_range: Tuple[float, float] = (0.005, 0.1),
                 meta_sgd_clip_range: Tuple[float, float] = (0., 1.),
                 init_path: str = '',
                ):
        """Constructor.

        Args:
        width (int): Width of each hidden layer in MLP.
        depth (int): Number of layers in MLP.
        out_channels (int): Number of output channels.
        k_mixtures: (int): Number of SIREN model.
        layer_sizes: List of hidden layer sizes for MLP parameterizing the map
            from latent to modulations. Input dimension is inferred from latent_dim
            and output dimension is inferred from number of modulations.
        w0 (float): Scale factor in sine activation in first layer.
        use_meta_sgd: Whether to use meta-SGD.
        num_inner_steps: Number of inner loop steps.
        meta_sgd_init_range: Range from which initial meta_sgd learning rates will
            be uniformly sampled.
        meta_sgd_clip_range: Range at which to clip learning rates.
        """
        super().__init__()
        self.width = width
        self.depth = depth
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.k_mixtures = k_mixtures
        self.mixture_type = mixture_type
        self.use_latent_embedding = use_latent_embedding
        self.std_latent = std_latent
        self.latent_channels = latent_channels
        self.w0 = w0
        self.latent_init_scale = latent_init_scale

        self.use_meta_sgd = use_meta_sgd
        self.meta_sgd_init_range = meta_sgd_init_range
        self.meta_sgd_clip_range = meta_sgd_clip_range
        # Initialize meta-SGD learning rates
        if self.use_meta_sgd:
            if use_latent_embedding:
                self.meta_sgd_lrs = MetaSGDLrs(self.latent_channels,
                                               self.meta_sgd_init_range,
                                               self.meta_sgd_clip_range)
            else:
                self.meta_sgd_lrs = MetaSGDLrs(self.k_mixtures,
                                               self.meta_sgd_init_range,
                                               self.meta_sgd_clip_range,
                                               unsqueeze_first=True)
        self.embedding_type = embedding_type
        self.context_type = self.mixture_type

        nonlinearity = f'sine-w0={w0}'

        self.pred_type = pred_type

        self.use_last_layer = False

        print(f'pred_type is {self.pred_type}')
        if self.pred_type in ['image', 'voxel']:
            outermost_linear = False
        elif self.pred_type in ['scene']:
            outermost_linear = True

        sirens = []
        for i in range(k_mixtures):
            sirens.append(list(modules.FCBlock(in_features=in_channels,
                                               out_features=out_channels,
                                               num_hidden_layers=depth,
                                               hidden_features=width,
                                               sparse_matrix='none',
                                               nonlinearity=nonlinearity,
                                               outermost_linear=outermost_linear).parameters()))

        # for inference
        self.model = modules.FCBlock(in_features=in_channels,
                                     out_features=out_channels,
                                     num_hidden_layers=depth,
                                     hidden_features=width,
                                     sparse_matrix='none',
                                     nonlinearity=nonlinearity,
                                     outermost_linear=outermost_linear)

        # for parameter
        self.model_param = modules.FCBlock(in_features=in_channels,
                                     out_features=out_channels,
                                     num_hidden_layers=depth,
                                     hidden_features=width,
                                     sparse_matrix='none',
                                     nonlinearity=nonlinearity,
                                     outermost_linear=outermost_linear)

        weights = []
        
        # count number of parameter in a single INR
        nparameters, num_blocks = 0, 0
        # aggregate the ensemble parameters into in the single parameter
        #for key, param in self.model.named_parameters():
        for pidx, param in enumerate(self.model_param.parameters()):
            weights = list(map(lambda k: k[pidx], sirens))
            concat_weights = torch.stack(weights, dim=0).requires_grad_()
            param.data = concat_weights
            nparam = param.nelement()
            nparameters += nparam
            num_blocks += 1
            pidx += 1

        #self.weights = nn.ParameterList(weights)
        self.net_size = nparameters
        self.num_blocks = num_blocks
        self.num_ctx_params = num_blocks // 2

        if self.use_latent_embedding:
            self.latent_channels = latent_channels
            if self.mixture_type in ['all-layer', 'layerwise']:
                if self.mixture_type == 'all-layer':
                    mixture_channels = self.k_mixtures
                elif self.mixture_type == 'layerwise':
                    mixture_channels = self.k_mixtures * self.num_ctx_params

                self.latent_to_mixture = nn.Linear(latent_channels, mixture_channels)
                self.latent_to_mixture.bias = nn.Parameter(torch.ones(self.latent_to_mixture.bias.size())/ self.k_mixtures)
            elif self.mixture_type in ['two-layers']:
                mixture_channels = self.k_mixtures * self.num_ctx_params
                self.latent_to_mixture = nn.Sequential(nn.Linear(latent_channels, latent_channels), nn.ReLU(), nn.Linear(latent_channels, mixture_channels))
                self.latent_to_mixture[2].bias = nn.Parameter(torch.ones(self.latent_to_mixture[2].bias.size())/ self.k_mixtures)
                self.latent_to_mixture[2].weight = nn.Parameter(torch.zeros(self.latent_to_mixture[2].weight.size()))
    
    def get_context_params(self, batch_size=None, eval_mode=False):
        if self.use_latent_embedding:
            if self.std_latent > 0:
                rand_params = torch.randn(batch_size, self.latent_channels).cuda() * self.std_latent
            else:
                rand_params = torch.zeros(batch_size, self.latent_channels).cuda()
        else:
            if self.context_type == 'all-layer':
                rand_params = torch.ones(batch_size, self.k_mixtures, 1)
            elif self.context_type == 'layerwise':
                rand_params = torch.ones(batch_size, self.k_mixtures, (self.num_blocks+1)//2)
                
            if eval_mode:
                rand_params = (rand_params / self.k_mixtures).cuda()
            else:
                rand_params.uniform_(self.latent_init_scale[0],self.latent_init_scale[1])
                rand_params = rand_params / rand_params.sum(dim=1, keepdim=True)
                rand_params = rand_params.cuda()

        context_params = rand_params.detach()
        context_params.requires_grad_()

        return context_params

    def get_mixture_weight(self, weights_mixture) -> torch.Tensor:
        if self.mixture_type == 'all-layer':
            weights_mixture = weights_mixture.unsqueeze(2)
        elif self.mixture_type in ['layerwise', 'two-layers']:
            weights_mixture = weights_mixture.view(-1, self.k_mixtures, self.num_ctx_params)

        return weights_mixture


    def get_combined_weight(self, weights_mixture) -> torch.Tensor:
        if weights_mixture is None:
            weights_mixture = self.context_params
        weights_mixture = weights_mixture

        params = dict()
        pi = 0
        for (name, all_weights), (_, res_weight) in zip(self.model_param.named_parameters(), self.model.named_parameters()):
            w_size = all_weights.size() # [k-mixtures, weight dimensions]
            all_weights = all_weights.view(1, self.k_mixtures, -1)

            if self.mixture_type == 'all-layer':
                new_weight = (all_weights * weights_mixture).sum(dim=1).view([-1] + list(w_size[1:]))

            elif self.mixture_type in ['layerwise', 'two-layers']:
                mi = pi // 2
                mixture = weights_mixture[:, :, mi:mi+1]
                
                new_weight = (mixture * all_weights).sum(dim=1, keepdim=False).view([-1] + list(w_size[1:]))

            params[name] = new_weight

            pi += 1
        return params

    def get_input_embedding(self, inputs, context_params):
        # Compute coordinate embedding is needed
        x = inputs['coords']
        
        return x


    def forward(self, inputs: torch.Tensor, context_params: Optional[torch.Tensor]=None, coefficients: Optional[torch.Tensor]=None) -> torch.Tensor:
        """Evaluates model at a batch of coordinates.

        Args:
        coords (Tensor): Tensor of coordinates. Should have shape (height, width, 2)
            for images and (depth/time, height, width, 3) for 3D shapes/videos.

        Returns:
        Output features at coords.
        """

        # Compute mixture weights based on latent vector if needed
        if self.use_latent_embedding:
            if coefficients is None:
                if context_params is None:
                    context_params = inputs['context_params']
                coefficients = self.latent_to_mixture(context_params)
            context_params = self.get_mixture_weight(coefficients)
        else:
            if len(context_params.size()) == 1:
                context_params = context_params.view(1, self.k_mixtures, -1)
            elif len(context_params.size()) == 2:
                context_params = context_params.view(context_params.size(0), self.k_mixtures, -1)


        params = self.get_combined_weight(context_params)

        x = self.get_input_embedding(inputs, context_params)
        out = self.model(x, params=params)

        if self.use_last_layer:
            out = self.last_layer(out)

        result = {'model_in': inputs,
                 'model_out': out,
                 'l1_mean': torch.abs(context_params).mean()}

        return result


    def get_parameters(self):       
        param_gen_list = []
        if self.use_latent_embedding:
            param_gen_list.append(self.latent_to_mixture.parameters())
        param_gen_list.append(self.model_param.parameters())
        if self.use_meta_sgd:
            param_gen_list.append(self.meta_sgd_lrs.parameters())
        if self.use_last_layer:
            param_gen_list.append(self.last_layer.parameters())

        return itertools.chain(*param_gen_list)

    def get_named_parameters(self):
        param_gen_list = []
        if self.use_latent_embedding:
            param_gen_list.append(self.latent_to_mixture.named_parameters())
        param_gen_list.append(self.model_param.named_parameters())
        if self.use_meta_sgd:
            param_gen_list.append(self.meta_sgd_lrs.named_parameters())
        if self.use_last_layer:
            param_gen_list.append(self.last_layer.named_parameters())

        return itertools.chain(*param_gen_list)

############################
# Initialization schemes
############################
def hyper_weight_init(m, in_features_main_net):
    if hasattr(m, 'weight'):
        nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')
        m.weight.data = m.weight.data / 1.e2

    if hasattr(m, 'bias'):
        with torch.no_grad():
            m.bias.uniform_(-1/in_features_main_net, 1/in_features_main_net)


def hyper_bias_init(m):
    if hasattr(m, 'weight'):
        nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')
        m.weight.data = m.weight.data / 1.e2

    if hasattr(m, 'bias'):
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
        with torch.no_grad():
            m.bias.uniform_(-1/fan_in, 1/fan_in)


class EMAmodel(nn.Module):
    def __init__(self, model, decay=0.9999, use_num_upates=True):
        super().__init__()
        if decay < 0.0 or decay > 1.0:
            raise ValueError('Decay must be between 0 and 1')

        self.m_name2s_name = {}
        self.register_buffer('decay', torch.tensor(decay, dtype=torch.float32))
        self.register_buffer('num_updates', torch.tensor(0,dtype=torch.int) if use_num_upates
                             else torch.tensor(-1,dtype=torch.int))

        for name, p in model.get_named_parameters():
            if p.requires_grad:
                #remove as '.'-character is not allowed in buffers
                s_name = name.replace('.','')
                self.m_name2s_name.update({name:s_name})
                self.register_buffer(s_name,p.clone().detach().data)

        self.collected_params = []

    def forward(self,model):
        decay = self.decay

        if self.num_updates >= 0:
            self.num_updates += 1
            decay = min(self.decay,(1 + self.num_updates) / (10 + self.num_updates))

        one_minus_decay = 1.0 - decay

        with torch.no_grad():
            m_param = dict(model.get_named_parameters())
            shadow_params = dict(self.named_buffers())

            for key in m_param:
                if m_param[key].requires_grad:
                    sname = self.m_name2s_name[key]
                    shadow_params[sname] = shadow_params[sname].type_as(m_param[key])
                    shadow_params[sname].sub_(one_minus_decay * (shadow_params[sname] - m_param[key]))
                else:
                    assert not key in self.m_name2s_name

    def copy_to(self, model):
        m_param = dict(model.get_named_parameters())
        shadow_params = dict(self.named_buffers())
        for key in m_param:
            if m_param[key].requires_grad:
                m_param[key].data.copy_(shadow_params[self.m_name2s_name[key]].data)
            else:
                assert not key in self.m_name2s_name

    def store(self, parameters):
        """
        Save the current parameters for restoring later.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            temporarily stored.
        """
        self.collected_params = [param.clone() for param in parameters]

    def restore(self, parameters):
        """
        Restore the parameters stored with the `store` method.
        Useful to validate the model with EMA parameters without affecting the
        original optimization process. Store the parameters before the
        `copy_to` method. After validation (or model saving), use this to
        restore the former parameters.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            updated with the stored parameters.
        """
        for c_param, param in zip(self.collected_params, parameters):
            param.data.copy_(c_param.data)
