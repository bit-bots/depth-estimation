import logging
from os import listdir
from os.path import splitext
from pathlib import Path

import numpy as np
import cv2
import torch
from torch.utils.data import Dataset


class BasicDataset(Dataset):
    def __init__(self, images_dir: str, masks_dir: str, mask_suffix: str = '', transform=None):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.mask_suffix = mask_suffix
        self.transform = transform

        self.ids = [splitext(file)[0] for file in listdir(images_dir) if not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, img, is_mask):
        if is_mask:
            img = np.clip(img, 0, 10)
        else:
            #Fixed c=4 problem
            w, h, c = img.shape
            if c == 4:
                img = img[:,:,:3]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    @classmethod
    def load(cls, filename):
        ext = splitext(filename)[1]
        if ext in ['.npz', '.npy']:
            return np.load(filename)['arr_0'].T
        else:
            return cv2.imread(str(filename))

    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = list(self.masks_dir.glob(name + self.mask_suffix + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        mask = self.load(mask_file[0])
        img = self.load(img_file[0])
        assert img.shape[0:2] == mask.shape[0:2], \
            f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        out = {
            "image": self.preprocess(img, is_mask=False),
            "depth": self.preprocess(mask, is_mask=True)
        }

        # Apply transforms
        if self.transform is not None:
            out = self.transform(**out)
        
        return out


class TorsoDataset(BasicDataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        super().__init__(images_dir, masks_dir, mask_suffix='_depth_raw', transform=transform)
