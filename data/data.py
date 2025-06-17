import torch.utils.data as Data
import torchvision.transforms as transforms


from PIL import Image, ImageOps
import os.path as osp
import random

from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator


class DataLoaderX(DataLoader):
    def _iter_(self):
        return BackgroundGenerator(super()._iter_())

class Dense_SIRST(Data.Dataset):

    def __init__(self, mode='train'):
        self.base_dir = 'datasets/DenseSIRST/data/SIRSTdevkit'
        self.mean = [0.343, 0.343, 0.343]
        self.std = [0.157, 0.157, 0.157]
        self.base_size = 512
        self.crop_size = 512

        if mode == 'train':
            txtfile = 'trainval_v1.txt'

        elif mode == 'test':
            txtfile = 'test_v1.txt' 

        self.list_dir = osp.join(self.base_dir, 'Splits',txtfile)
        self.imgs_dir = osp.join(self.base_dir, 'PNGImages')
        self.label_dir = osp.join(self.base_dir,'SIRST', 'BinaryMask')

        self.names = []
        with open(self.list_dir, 'r') as f:
            self.names += [line.strip() for line in f.readlines()]

        self.mode = mode
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])

    def __getitem__(self, i):
        name = self.names[i]
        img_path = osp.join(self.imgs_dir, name + '.png')
        label_path = osp.join(self.label_dir, name + '_pixels0.png')

        img = Image.open(img_path).convert('RGB')

        mask = Image.open(label_path).convert('L')

        if self.mode == 'train':
            img, mask = self._sync_transform(img, mask)
        elif self.mode == 'test':
            img, mask = self._testval_sync_transform(img, mask)
        else:
            raise ValueError("Unkown self.mode")

        img, mask = self.transform(img), transforms.ToTensor()(mask)
        return img, mask,name

    def __len__(self):
        return len(self.names)

    def _sync_transform(self, img, mask):
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.FLIP_TOP_BOTTOM)

        if random.random() < 0.5:
            img = img.transpose(Image.ROTATE_90)
            mask = mask.transpose(Image.ROTATE_90)

        crop_size = self.crop_size
        long_size = random.randint(int(self.base_size * 1.0), int(self.base_size * 1.1))
        w, h = img.size
        if h > w:
            oh = long_size
            ow = int(1.0 * w * long_size / h + 0.5)
            short_size = ow
        else:
            ow = long_size
            oh = int(1.0 * h * long_size / w + 0.5)
            short_size = oh
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)

        if short_size < crop_size:
            padh = crop_size - oh if oh < crop_size else 0
            padw = crop_size - ow if ow < crop_size else 0
            padh = padh + 1 if padh % 2 != 0 else padh
            padw = padw + 1 if padw % 2 != 0 else padw
            img = ImageOps.expand(img, border=(int(padw / 2), int(padh / 2), int(padw / 2), int(padh / 2)), fill=0)
            mask = ImageOps.expand(mask, border=(int(padw / 2), int(padh / 2), int(padw / 2), int(padh / 2)), fill=0)

        w, h = img.size
        x1 = random.randint(0, w - crop_size)
        y1 = random.randint(0, h - crop_size)
        img = img.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        mask = mask.crop((x1, y1, x1 + crop_size, y1 + crop_size))

        return img, mask

    def _testval_sync_transform(self, img, mask):
        crop_size = self.crop_size
        img = img.resize((crop_size, crop_size), Image.BILINEAR)
        mask = mask.resize((crop_size, crop_size), Image.NEAREST)

        return img, mask
