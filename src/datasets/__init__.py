# ------------------------------------------------------------------------------------
# HQ-Transformer
# Copyright (c) 2022 KakaoBrain. All Rights Reserved.
# Licensed under the MIT License [see LICENSE for details]
# ------------------------------------------------------------------------------------

import os
import random

import h5py
import glob
from pathlib import Path
import numpy as np
from PIL import Image

import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import VisionDataset
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
import pytorch_lightning as pl
from .nerf_dataset import SRNDatasets, SRNDatasetsLMDB

import pdb
ROOT_DIR_IMAGENET = 'datasets/ImageNet'

class DatasetModule(pl.LightningDataModule):
    def __init__(self,
                 dataset: str = 'imagenet',
                 image_resolution: int = 256,
                 train_batch_size: int = 2,
                 valid_batch_size: int = 32,
                 num_workers: int = 8,
                 root: str = None,
                 jitter: bool = True,
                 ):

        super().__init__()
        self.dataset = dataset
        self.image_resolution = image_resolution
        self.train_batch_size = train_batch_size
        self.valid_batch_size = valid_batch_size
        self.num_workers = num_workers
        self.root = root

        train_t = transforms.Compose([
            transforms.Resize(image_resolution),
            transforms.RandomCrop(image_resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        valid_t = transforms.Compose([
            transforms.Resize(image_resolution),
            transforms.CenterCrop(image_resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        if jitter:
            self.train_transform = train_t
        else:
            self.train_transform = valid_t
        self.valid_transform = valid_t

    def setup(self, stage=None):
        root = self.root
        if self.dataset == 'imagenet':
            if self.root is None:
                root = ROOT_DIR_IMAGENET
            self.trainset = torchvision.datasets.ImageNet(root=root, split='train', transform=self.train_transform)
            self.validset = torchvision.datasets.ImageNet(root=root, split='val', transform=self.valid_transform)

    def train_dataloader(self):
        return DataLoader(self.trainset,
                          batch_size=self.train_batch_size,
                          num_workers=self.num_workers,
                          pin_memory=True)

    def valid_dataloader(self):
        return DataLoader(self.validset,
                          batch_size=self.valid_batch_size,
                          num_workers=self.num_workers,
                          pin_memory=True)

class SignalAgnosticDatasetModule(pl.LightningDataModule):
    def __init__(self,
                 opt,
                 train_batch_size: int = 2,
                 valid_batch_size: int = 32,
                 num_workers: int = 8,
                 dataset_root: str = None,
                 ):

        super().__init__()
        dataset = opt.dataset
        sampling = opt.num_subsampling if opt.num_subsampling > 0 else None

        self.train_batch_size = train_batch_size
        self.valid_batch_size = valid_batch_size
        self.num_workers = num_workers

        # setup dataset
        if 'shapenet' == dataset:
            self.trainset = ShapeNet(split='train', sampling=sampling, dataset_root=dataset_root)
            self.validset = ShapeNet(split='train', sampling=sampling, dataset_root=dataset_root)
        elif 'shapenetvoxel' == dataset:
            self.trainset = ShapeNetVoxel(split='train', sampling=sampling, dataset_root=dataset_root)
            self.validset = ShapeNetVoxel(split='test', sampling=sampling, dataset_root=dataset_root)
        elif 'srn_cars' == dataset:
            self.trainset = SRNDatasetsLMDB(category='cars', opt=opt.hparams_nerf, dataset_root=dataset_root)
            self.validset = SRNDatasetsLMDB(category='cars', opt=opt.hparams_nerf, dataset_root=dataset_root, split='test')
        else:
            # image dataset
            if 'celebahq' == dataset:
                dataset_train = CelebAHQ(split='train', downsampled=False, dataset_root=dataset_root)
                dataset_test = CelebAHQ(split='test', downsampled=False, dataset_root=dataset_root)

            # coordinate dataset
            self.trainset = Implicit2DWrapper(dataset_train, sidelength=opt.resolution, sampling=sampling)
            self.validset = Implicit2DWrapper(dataset_test, sidelength=opt.resolution, sampling=sampling)

    def train_dataloader(self):
        return DataLoader(self.trainset,
                          batch_size=self.train_batch_size,
                          num_workers=self.num_workers,
                          pin_memory=True)

    def valid_dataloader(self):
        return DataLoader(self.validset,
                          batch_size=self.valid_batch_size,
                          num_workers=self.num_workers,
                          pin_memory=True)


class ImageNet(torchvision.datasets.ImageNet):
    def __init__(self, split='train', transform=None):
        super().__init__(root=ROOT_DIR_IMAGENET, split=split, transform=transform)


class ImageFolder(VisionDataset):
    def __init__(self, root, train_list_file, val_list_file, split='train', **kwargs):
        super().__init__(root, **kwargs)

        self.train_list_file = train_list_file
        self.val_list_file = val_list_file

        self.split = self._verify_split(split)

        self.loader = torchvision.datasets.folder.default_loader
        self.extensions = torchvision.datasets.folder.IMG_EXTENSIONS

        if self.split == 'trainval':
            fname_list = os.listdir(self.root)
            samples = [self.root.joinpath(fname) for fname in fname_list
                       if fname.lower().endswith(self.extensions)]
        else:
            listfile = self.train_list_file if self.split == 'train' else self.val_list_file
            with open(listfile, 'r') as f:
                #samples = [self.root.joinpath(line.strip()) for line in f.readlines()]
                samples = [os.path.join(self.root, line.strip()) for line in f.readlines()]

        self.samples = samples

    def _verify_split(self, split):
        if split not in self.valid_splits:
            msg = "Unknown split {} .".format(split)
            msg += "Valid splits are {{}}.".format(", ".join(self.valid_splits))
            raise ValueError(msg)
        return split

    @property
    def valid_splits(self):
        return 'train', 'val', 'trainval'

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index, with_transform=True):
        path = self.samples[index]
        sample = self.loader(path)
        if self.transforms is not None and with_transform:
            sample, _ = self.transforms(sample, None)
        #return sample, 0
        return sample

class SampleImage(DataLoader):
    def __init__(self, path, size):
        super().__init__(self)
        self.img = Image.open(path).resize((size, size), Image.ANTIALIAS)
        self.img_channels = 3

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.img

class CelebAHQ(DataLoader):
    def __init__(self, split, downsampled=False, resolution=64, dataset_root='datasets', tf_dataset=False):
        # SIZE (128 x 128)
        # super().__init__(self)
        assert split in ['train', 'test'], "Unknown split"

        self.dataset_root = dataset_root
        self.root = os.path.join(self.dataset_root, 'CelebAHQ')
        self.img_channels = 3
        self.fnames = []
        if tf_dataset:
            nameformat = 'img%08d.png'
            train_range = range(0, 27000)
            test_range = range(27000, 30000)
        else:
            nameformat = '%05d.jpg'
            train_range = range(1, 27001)
            test_range = range(27001, 30001)
        if split == 'train':
            for i in train_range:
                self.fnames.append(nameformat % i)
        elif split == 'test':
            for i in test_range:
                self.fnames.append(nameformat % i)

        self.downsampled = downsampled
        self.res = resolution

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        path = os.path.join(self.root, self.fnames[idx])
        img = Image.open(path)
        if self.downsampled:
            width, height = img.size  # Get dimensions

            # s = min(width, height)
            # left = (width - s) / 2
            # top = (height - s) / 2
            # right = (width + s) / 2
            # bottom = (height + s) / 2
            # img = img.crop((left, top, right, bottom))
            img = img.resize((self.res, self.res), resample=Image.BICUBIC)

        return img

def get_mgrid(sidelen, dim=2):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.'''
    if isinstance(sidelen, int):
        sidelen = dim * (sidelen,)

    if dim == 2:
        pixel_coords = np.stack(np.mgrid[:sidelen[0], :sidelen[1]], axis=-1)[None, ...].astype(np.float32)
        pixel_coords[0, :, :, 0] = pixel_coords[0, :, :, 0] / (sidelen[0] - 1)
        pixel_coords[0, :, :, 1] = pixel_coords[0, :, :, 1] / (sidelen[1] - 1)
    elif dim == 3:
        pixel_coords = np.stack(np.mgrid[:sidelen[0], :sidelen[1], :sidelen[2]], axis=-1)[None, ...].astype(np.float32)
        pixel_coords[..., 0] = pixel_coords[..., 0] / max(sidelen[0] - 1, 1)
        pixel_coords[..., 1] = pixel_coords[..., 1] / (sidelen[1] - 1)
        pixel_coords[..., 2] = pixel_coords[..., 2] / (sidelen[2] - 1)
    else:
        raise NotImplementedError('Not implemented for dim=%d' % dim)

    pixel_coords -= 0.5
    pixel_coords *= 2.
    pixel_coords = torch.Tensor(pixel_coords).view(-1, dim)
    return pixel_coords

def get_mgrid_voxel(sidelen):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.'''
    dim = 3
    if isinstance(sidelen, int):
        sidelen = dim * (sidelen,)

    pixel_coords = np.stack(np.mgrid[:sidelen[0], :sidelen[1], :sidelen[2]], axis=-1)[None, ...].astype(np.float32)
    pixel_coords[..., 0] = (pixel_coords[..., 0] + 1) / max(sidelen[0], 1)
    pixel_coords[..., 1] = (pixel_coords[..., 1] + 1) / sidelen[1]
    pixel_coords[..., 2] = (pixel_coords[..., 2] + 1) / sidelen[2]

    pixel_coords -= 0.5
    pixel_coords *= 2.
    pixel_coords = torch.Tensor(pixel_coords).view(-1, dim)
    return pixel_coords



class Implicit2DWrapper(DataLoader):
    def __init__(self, dataset, sidelength=None, istuple=False, ispair=False, checkpoint_path=None, sampling=None, transform_type='baseline', resampling='bicubic', antialias=True):
        self.sidelength = sidelength

        if transform_type == 'baseline':       
            print('use baseline transform')
            if resampling == 'bicubic':
                interp_method = torchvision.transforms.InterpolationMode.BICUBIC
            elif resampling == 'bilinear':
                interp_method = torchvision.transforms.InterpolationMode.BILINEAR

            if len(resampling) > 0:
                self.transform = Compose([
                    Resize(sidelength, antialias=antialias, interpolation=interp_method),
                    ToTensor(),
                    Normalize(torch.Tensor([0.5]), torch.Tensor([0.5]))
                ])
            else:
                self.transform = Compose([
                    ToTensor(),
                    Normalize(torch.Tensor([0.5]), torch.Tensor([0.5]))
                ])
            self.stats = (torch.Tensor([0.5, 0.5, 0.5]), torch.Tensor([0.5, 0.5, 0.5]))
        elif transform_type == 'ffhq':
            assert(False)
            # print('use FFHQ transform')
            # self.transform = transforms.Compose([
            #     transforms.RandomResizedCrop(sidelength, scale=(0.75, 1.0), ratio=(1.0, 1.0), interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
            #     transforms.RandomHorizontalFlip(p=0.5),
            #     ToTensor(),
            #     Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            # ])
        elif transform_type == 'cifar10-stats':
            self.transform = Compose([
                ToTensor(),
                Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
            ])
            self.stats = (torch.Tensor([0.4914, 0.4822, 0.4465]), torch.Tensor([0.2023, 0.1994, 0.2010]))


        self.dataset = dataset
        self.mgrid = get_mgrid(sidelength)
        self.sampling = sampling
        self.istuple = istuple
        self.ispair = ispair

        if hasattr(dataset, 'img_channels'):
            self.img_channels = self.dataset.img_channels
        else:
            self.img_channels = 3
        
        self.checkpoint = checkpoint_path

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        
        if self.istuple:
            data = self.dataset[idx]
            img = self.transform(data[0])
        else:
            img = self.transform(self.dataset[idx])

        img = img.permute(1, 2, 0).view(-1, self.img_channels)

        if self.sampling is not None:
            idcs = np.random.randint(0, int(img.size(0)), size=self.sampling)
            mgrid = self.mgrid[idcs]
            img = img[idcs]
        else:
            mgrid = self.mgrid

        in_dict = {'idx': idx, 'coords': mgrid}
        gt_dict = {'img': img}

        # for contrastive learning
        if self.istuple and self.ispair:
            img2 = self.transform(data[1])
            img2 = img2.permute(1, 2, 0).view(-1, self.img_channels)
            if self.sampling is not None:
                img2 = img2[idcs]

            gt_dict['img2'] = img2

        if self.checkpoint is not None:
            ckpt = torch.load(os.path.join(self.checkpoint, f'model{idx}.pth'), map_location=torch.device('cpu'))
            gt_dict['ckpt'] = ckpt

        return in_dict, gt_dict

    def get_item_small(self, idx):
        img = self.transform(self.dataset[idx])
        spatial_img = img.clone()
        img = img.permute(1, 2, 0).view(-1, self.img_channels)

        gt_dict = {'img': img}

        return spatial_img, img, gt_dict



class INRWeightWrapper(DataLoader):
    def __init__(self, dataset, sidelength=None, istuple=False, checkpoint_path=None, checkpoint_step=0, feed_type='weight', context_tag='set', reduce_sample=False, resampling=None, antialias=None):
        print(checkpoint_path)
        self.reduce_sample = reduce_sample
        self.sidelength = sidelength

        if antialias is not None and resampling is not None:
            if resampling == 'bicubic':
                interp_method = torchvision.transforms.InterpolationMode.BICUBIC
            elif resampling == 'bilinear':
                interp_method = torchvision.transforms.InterpolationMode.BILINEAR

            self.transform = Compose([
                Resize(sidelength, antialias=antialias, interpolation=interp_method),
                ToTensor(),
                Normalize(torch.Tensor([0.5]), torch.Tensor([0.5]))
            ])
        else:
            self.transform = Compose([
                Resize(sidelength),
                ToTensor(),
                Normalize(torch.Tensor([0.5]), torch.Tensor([0.5]))
            ])

        self.dataset = dataset
        self.mgrid = get_mgrid(sidelength)
        self.istuple = istuple

        if hasattr(dataset, 'img_channels'):
            self.img_channels = self.dataset.img_channels
        else:
            self.img_channels = 3
        
        if hasattr(dataset, 'data_type'):
            self.data_type = dataset.data_type
        else:
            self.data_type = 'image'

        self.affine = (None, None)

        self.checkpoint = checkpoint_path
        self.ckpt_step = checkpoint_step
        self.context_tag = context_tag

        self.feed_type = feed_type
        
        if self.feed_type in ['weight', 'dict']:
            if self.ckpt_step == 0:
                stat_path = os.path.join(self.checkpoint, f'mean_std_stats.pth')
            else:
                stat_path = os.path.join(self.checkpoint, f'mean_std_stats_steps{self.ckpt_step}.pth')

            if os.path.exists(stat_path):
                ckpt = torch.load(stat_path, map_location=torch.device('cpu'))
                
                mean, std = [], []
                for k in ckpt.keys():
                    w = ckpt[k]
                    mean.append(w['mean'].view( -1))   
                    std.append(w['std'].view(-1))

                mean = torch.cat(mean, dim=0).unsqueeze(0).contiguous()
                std = torch.cat(std, dim=0).unsqueeze(0).contiguous()
                self.affine = (mean, std)    

                # elif self.feed_type == 'dict':
                #     mean, std = dict(), dict()

                #     for k in ckpt.keys():
                #         w = ckpt[k]
                #         mean[k] = w['mean']
                #         std[k] = w['std']

                # self.affine = (mean, std)
        elif self.feed_type == 'context':
            feat_path = os.path.join(self.checkpoint, 'contexts', f'context-epoch{self.ckpt_step}.pth')
            self.feat = torch.load(feat_path).requires_grad_(False)
            self.affine = (self.feat.mean(dim=0), self.feat.std(dim=0))

        elif self.feed_type == 'context-set':
            if len(self.context_tag) == 0:
                context_tag = ''
            else:
                context_tag = '-'+ self.context_tag
            feat_path = os.path.join(self.checkpoint, 'contexts', f'context{context_tag}-epoch{self.ckpt_step}.pth')
            self.feat = torch.load(feat_path).requires_grad_(False)
            self.affine = (self.feat.mean(dim=[0,1]), self.feat.std(dim=[0,1]))

    def __len__(self):
        if self.dataset is not None:
            return len(self.dataset)
        else:
            return len(self.feat)

    def __getitem__(self, idx):
        if self.reduce_sample > 0:
            idx = idx % self.reduce_sample
        if self.dataset is not None:            
            if self.data_type == 'image':
                if self.istuple:
                    data = self.dataset[idx]
                    img = self.transform(data[0])
                else:
                    img = self.transform(self.dataset[idx])
            else:                
                    img = self.dataset[idx]
            gt_dict = {'img': img}
        else:
            img = None
            gt_dict = {}

        in_dict = {'idx': idx, 'coords': self.mgrid}

        if self.feed_type in ['weight', 'dict']:
            if self.ckpt_step == 0:
                ckpt_path = os.path.join(self.checkpoint, 'checkpoints', f'model{idx}.pth')
            else:
                ckpt_path = os.path.join(self.checkpoint, 'checkpoints', f'steps{self.ckpt_step}', f'model{idx}.pth')

            ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
            W = []
            for k in ckpt.keys():
                w = ckpt[k]
                W.append(w.view(-1))   
            W = torch.cat(W, dim=0).contiguous()
            gt_dict['weights'] = W
        #elif self.feed_type == 'dict':
        #    gt_dict['weights'] = ckpt
        elif self.feed_type == 'context':
            if self.feat is not None:
                gt_dict['weights'] = self.feat[idx, :]
        elif self.feed_type == 'context-set':
            if self.feat is not None:
                rndidx = random.randint(0,self.feat.size(1)-1)
                gt_dict['weights'] = self.feat[idx, rndidx, :]

        return in_dict, gt_dict

# ShapeNet DataLoader is modified from the original code
# https://github.com/yilundu/gem/blob/2778026ac4508f44c7af160e4e157c6fb039f4ce/dataio.py#L718
class ShapeNet(DataLoader):
    def __init__(self, split='train', sampling=None, dataset_root='datasets', simple_output=False, random_scale=False):
        self.dataset_root = dataset_root
        self.sampling = sampling
        self.init_model_bool = False
        self.split = split
        self.simple_output = simple_output
        self.random_scale = random_scale
        self.init_model()
        self.data_type = 'voxel'

    def __len__(self):
        if self.split == "train":
            return 35019
        else:
            return 8762

    def init_model(self):
        split = self.split
        points_path = os.path.join(self.dataset_root, 'shapenet', 'all_vox256_img', 'data_points_int_' + split + '.pth')
        values_path = os.path.join(self.dataset_root, 'shapenet', 'all_vox256_img', 'data_values_' + split + '.pth')

        self.data_points_int = torch.load(points_path).byte()
        self.data_values = torch.load(values_path).byte()


    def __getitem__(self, idx):
        points = (self.data_points_int[idx].float() + 1) / 128 - 1
        occs = self.data_values[idx].float() * 2 -1

        if self.sampling is not None:
            idcs = np.random.randint(0, len(points), size=self.sampling)
            points = points[idcs]
            occs = occs[idcs]

        if self.random_scale:
            points = random.uniform(0.75, 1.25) * points

        if self.simple_output:
            return occs

        else:
            in_dict = {'idx': idx, 'coords': points}
            gt_dict = {'img': occs}

            return in_dict, gt_dict

# ShapeNet DataLoader is modified from the original code
# https://github.com/yilundu/gem/blob/2778026ac4508f44c7af160e4e157c6fb039f4ce/dataio.py#L718
class ShapeNetHDF5(DataLoader):
    def __init__(self, split='train', sampling=None, dataset_root='datasets'):
        self.dataset_root = dataset_root
        self.data_path = os.path.join(self.dataset_root, 'shapenet', 'all_vox256_img', 'all_vox256_img_' + split + '.hdf5')
        self.sampling = sampling
        self.init_model_bool = False
        self.split = split
        self.init_model()

    def __len__(self):
        if self.split == "train":
            return 35019
        else:
            return 8762

    def init_model(self):
        data_path = self.data_path
        data_dict = h5py.File(data_path, 'r')
        self.data_points_int = np.array(data_dict['points_64'][:])
        self.data_points = (self.data_points_int.astype(np.float32) + 1) / 128 - 1

        self.data_values = np.array(data_dict['values_64'][:])
        #self.data_voxels = np.array(data_dict['voxels'][:])
        #self.data_values = self.data_values * 2 - 1 # [-1, 1]

        self.init_model_bool = True

    def __getitem__(self, idx):

        if not self.init_model_bool:
            self.init_model()

        points = torch.from_numpy(self.data_points[idx]).float()
        occs = torch.from_numpy(self.data_values[idx]).float()
        occs = occs * 2 - 1

        if self.sampling is not None:
            idcs = np.random.randint(0, len(points), size=self.sampling)
            points = points[idcs]
            occs = occs[idcs]

        in_dict = {'idx': idx, 'coords': points}
        gt_dict = {'img': occs}

        return in_dict, gt_dict


# ShapeNet DataLoader is modified from the original code
# https://github.com/yilundu/gem/blob/2778026ac4508f44c7af160e4e157c6fb039f4ce/dataio.py#L718
class ShapeNetVoxel(DataLoader):
    def __init__(self, split='train', sampling=None, dataset_root='datasets'):
        self.dataset_root = dataset_root
        self.data_path = os.path.join(dataset_root, 'shapenet', 'all_vox256_img', 'all_vox256_img_' + split + '.pth')
        self.data_voxels = torch.load(self.data_path).byte()
        self.sampling = sampling
        self.split = split
        self.grid = get_mgrid_voxel(64)
        self.affine = (None, None)

    def __len__(self):
        if self.split == "train":
            return 35019
        else:
            return 8762

    def dec2bin(self, x, bits):
        mask = 2 ** torch.arange(bits - 1, -1, -1).to(x.device, x.dtype)
        return x.unsqueeze(-1).bitwise_and(mask).ne(0).byte().flip(-1)
        
    def __getitem__(self, idx):
        points = self.grid.float()

        encoded_voxels = self.data_voxels[idx]
        occs = self.dec2bin(encoded_voxels, 8).view(-1, 1).float()
        #occs = occs * 2 - 1       

        if self.sampling is not None:
            idcs = np.random.randint(0, len(points), size=self.sampling)
            points = points[idcs]
            occs = occs[idcs]

        in_dict = {'idx': idx, 'coords': points}
        gt_dict = {'img': occs}

        return in_dict, gt_dict