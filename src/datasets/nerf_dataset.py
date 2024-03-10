# ------------------------------------------------------------------------------------
# HQ-Transformer
# Copyright (c) 2022 KakaoBrain. All Rights Reserved.
# Licensed under the MIT License [see LICENSE for details]
# ------------------------------------------------------------------------------------

import os
import glob
import imageio
import lmdb
import pickle
from io import BytesIO
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import Compose, ToTensor, Normalize

import numpy as np

import pdb

class SRNDatasets(DataLoader):
    """
    Dataset from SRN (V. Sitzmann et al. 2020)
    PixelNeRF: https://github.com/sxyu/pixel-nerf/blob/master/src/data/SRNDataset.py
    """

    def __init__(
        self, category, opt, split="train", world_scale=1.0, dataset_root='datasets'
    ):
        """
        :param split train | val | test
        :param image_size result image size (resizes if different)
        :param world_scale amount to scale entire world by
        """
        self.debug = opt.debug
        self.zero_to_one = True if opt.rgb_activation == 'sigmoid' else False
        self.subsampled_views = opt.subsampled_views
        image_size=(opt.resolution, opt.resolution)
        
        if category == 'cars':
            path = dataset_root + '/srn_cars/cars'
        else:
            raise NotImplementedError("please category name of SRN Dataset")

        self.base_path = path + "_" + split
        self.dataset_name = os.path.basename(path)

        
        print("Loading SRN dataset", self.base_path, "name:", self.dataset_name)
        self.split = split
        assert os.path.exists(self.base_path)

        # if category == 'chairs' and split == "train":
        #     # Ugly thing from SRN's public dataset
        #     tmp = os.path.join(self.base_path, "chairs_2.0_train")
        #     if os.path.exists(tmp):
        #         self.base_path = tmp

        self.intrins = sorted(
            glob.glob(os.path.join(self.base_path, "*", "intrinsics.txt"))
        )

        self.image_to_tensor = Compose([
            ToTensor(),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        # self.mask_to_tensor = Compose([
        #     ToTensor(),
        #     Normalize((0.0,), (1.0,))
        # ])

        
        self.image_size = image_size
        self.world_scale = world_scale
        self._coord_trans = torch.diag(
            torch.tensor([1, -1, -1, 1], dtype=torch.float32)
        )

        self.z_near = opt.near
        self.z_far = opt.far
        
#        if category == 'cars':
#            self.z_near = 0.8
#            self.z_far = 1.8
#        elif category == 'chairs':
#            self.z_near = 1.25
#            self.z_far = 2.75
        self.lindisp = False

    def __len__(self):
        if self.debug:
            return 1 # for debug (NeRF check)
        return len(self.intrins)
        
    def __getitem__(self, index):
        intrin_path = self.intrins[index]
        dir_path = os.path.dirname(intrin_path)
        rgb_paths = sorted(glob.glob(os.path.join(dir_path, "rgb", "*")))
        pose_paths = sorted(glob.glob(os.path.join(dir_path, "pose", "*")))

        assert len(rgb_paths) == len(pose_paths)

        with open(intrin_path, "r") as intrinfile:
            lines = intrinfile.readlines()
            focal, cx, cy, _ = map(float, lines[0].split())
            height, width = map(int, lines[-1].split())
            
        all_imgs = []
        all_poses = []
        # all_masks = []
        # all_bboxes = []
        for rgb_path, pose_path in zip(rgb_paths, pose_paths):
            img = imageio.imread(rgb_path)[..., :3]
            img_tensor = self.image_to_tensor(img)
            # mask = (img != 255).all(axis=-1)[..., None].astype(np.uint8) * 255
            # mask_tensor = self.mask_to_tensor(mask)

            pose = torch.from_numpy(
                np.loadtxt(pose_path, dtype=np.float32).reshape(4, 4)
            )
            pose = pose @ self._coord_trans

            # rows = np.any(mask, axis=1)
            # cols = np.any(mask, axis=0)
            # rnz = np.where(rows)[0]
            # cnz = np.where(cols)[0]
            # if len(rnz) == 0:
            #     raise RuntimeError(
            #         "ERROR: Bad image at", rgb_path, "please investigate!"
            #     )
            # rmin, rmax = rnz[[0, -1]]
            # cmin, cmax = cnz[[0, -1]]
            # bbox = torch.tensor([cmin, rmin, cmax, rmax], dtype=torch.float32)

            all_imgs.append(img_tensor)
            # all_masks.append(mask_tensor)
            all_poses.append(pose)
            # all_bboxes.append(bbox)

        all_imgs = torch.stack(all_imgs)
        all_poses = torch.stack(all_poses)
        # all_masks = torch.stack(all_masks)
        # all_bboxes = torch.stack(all_bboxes)

        if all_imgs.shape[-2:] != self.image_size:
            scale = self.image_size[0] / all_imgs.shape[-2]
            focal *= scale
            cx *= scale
            cy *= scale
            # all_bboxes *= scale

            all_imgs = F.interpolate(all_imgs, size=self.image_size, mode="area")
            # all_masks = F.interpolate(all_masks, size=self.image_size, mode="area")

        if self.world_scale != 1.0:
            focal *= self.world_scale
            all_poses[:, :3, 3] *= self.world_scale
        focal = torch.tensor(focal, dtype=torch.float32)

        # result = {
        #     "path": dir_path,
        #     "img_id": index,
        #     "focal": focal,
        #     "c": torch.tensor([cx, cy], dtype=torch.float32),
        #     "images": all_imgs,
        #     "masks": all_masks,
        #     "bbox": all_bboxes,
        #     "poses": all_poses,
        # }
        if self.zero_to_one:
            all_imgs = all_imgs * 0.5 + 0.5
        in_dict = {'idx': index,
                'focal': focal,
                'c2w': all_poses}
        out_dict = {'img': all_imgs}

        return in_dict, out_dict


def format_for_lmdb(*args):
    key_parts = []
    for arg in args:
        if isinstance(arg, int):
            arg = str(arg).zfill(7)
        key_parts.append(arg)
    return '-'.join(key_parts).encode('utf-8')


class SRNDatasetsLMDB(DataLoader):
    """
    Dataset from SRN (V. Sitzmann et al. 2020)
    PixelNeRF: https://github.com/sxyu/pixel-nerf/blob/master/src/data/SRNDataset.py
    """

    def __init__(
        self, category, opt, split="train", world_scale=1.0, dataset_root='datasets', zero_to_one=False
    ):
        """
        :param split train | val | test
        :param image_size result image size (resizes if different)
        :param world_scale amount to scale entire world by
        """
        if category != 'cars':
            raise NotImplementedError("please category name of SRN Dataset")

        self.base_path = os.path.join(dataset_root, 'srn_cars_lmdb', "cars_" + split)
        assert os.path.exists(self.base_path)

        self.env = lmdb.open(
            self.base_path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        
        if not self.env:
            raise IOError('Cannot open lmdb dataset', self.base_path)

        print("Loading SRN dataset", self.base_path, "name:")
        with self.env.begin(write=False) as txn:
            self.length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))

        self.split = split

        self.image_to_tensor = Compose([
            ToTensor(),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        self.debug = opt.debug
        self.zero_to_one = True if opt.rgb_activation == 'sigmoid' or zero_to_one else False
        self.subsampled_views = opt.subsampled_views
        self.image_size = (opt.resolution, opt.resolution)
        self.world_scale = world_scale
        self._coord_trans = torch.diag(
            torch.tensor([1, -1, -1, 1], dtype=torch.float32)
        )

        self.z_near = opt.near
        self.z_far = opt.far        
        self.lindisp = False

    def __len__(self):
        if self.debug:
            return 1 # for debug (NeRF check)
        return self.length
        
    def __getitem__(self, index):
        all_imgs = []
        all_poses = []

        with self.env.begin(write=False) as txn:
            intrins_byte = txn.get(format_for_lmdb(index, 'intrinsic'))
            intrins = pickle.loads(intrins_byte)

            focal, cx, cy = intrins[0][0], intrins[0][1], intrins[0][2]

            l_key = format_for_lmdb(index, 'length')
            length_txt = txn.get(l_key)
            for frame_idx in range(0, int(length_txt)):
                #i_key = f'{str(index).zfill(7)}img{str(frame_idx).zfill(7)}'.encode('utf-8')
                #p_key = f'{str(index).zfill(7)}pose{str(frame_idx).zfill(7)}'.encode('utf-8')
                i_key = format_for_lmdb(index, 'img', frame_idx)
                p_key = format_for_lmdb(index, 'pose', frame_idx)

                pose = pickle.loads(txn.get(p_key))
                
                buffer = BytesIO(txn.get(i_key))
                img = Image.open(buffer)
                
                img_tensor = self.image_to_tensor(img)
                

                all_imgs.append(img_tensor)

                all_poses.append(pose)

        all_imgs = torch.stack(all_imgs)
        all_poses = torch.stack(all_poses)

        if all_imgs.shape[-2:] != self.image_size:
            scale = self.image_size[0] / all_imgs.shape[-2]
            focal *= scale
            cx *= scale
            cy *= scale

            all_imgs = F.interpolate(all_imgs, size=self.image_size, mode="area")

        if self.world_scale != 1.0:
            focal *= self.world_scale
            all_poses[:, :3, 3] *= self.world_scale
        focal = torch.tensor(focal, dtype=torch.float32)

        if self.zero_to_one:
            all_imgs = all_imgs * 0.5 + 0.5
        in_dict = {'idx': index,
                   'focal': focal,
                   'c2w': all_poses}
        out_dict = {'img': all_imgs}

        return in_dict, out_dict
