# encoding: utf-8

import os.path as osp
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
import torch
import cv2
import torch.nn.functional as F

def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


class ImageDataset(Dataset):
    """Image Person ReID Dataset"""

    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

        # self.att_transform = T.ToTensor()
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid = self.dataset[index]
        img = read_image(img_path)
        # attribute = cv2.imread(attribute)
        # attribute = cv2.resize(attribute, (128, 256), interpolation=cv2.INTER_NEAREST)
        # attribute = torch.tensor(attribute, dtype=torch.int64).unsqueeze(-1)
        # attribute = F.interpolate(attribute,size=[256,128], mode="nearest")
        # attribute = self.att_transform(attribute)

        if self.transform is not None:
            img = self.transform(img)

        return img, pid, camid, img_path

# class train_ImageDataset(Dataset):
#     """Image Person ReID Dataset"""
#
#     def __init__(self, dataset, transform=None):
#         self.dataset = dataset
#         self.transform = transform
#
#         # self.att_transform = T.ToTensor()
#     def __len__(self):
#         return len(self.dataset)
#
#     def __getitem__(self, index):
#         img_path, pid, camid, attribute = self.dataset[index]
#         img = read_image(img_path)
#         attribute = cv2.imread(attribute)
#         attribute = cv2.resize(attribute, (128, 256), interpolation=cv2.INTER_NEAREST)
#         attribute = torch.tensor(attribute, dtype=torch.int64).unsqueeze(-1)
#
#         # attribute = F.interpolate(attribute,size=[256,128], mode="nearest")
#         # attribute = self.att_transform(attribute)
#
#         if self.transform is not None:
#             img = self.transform(img)
#
#         return img, pid, camid, img_path, attribute
#
# class test_ImageDataset(Dataset):
#     """Image Person ReID Dataset"""
#
#     def __init__(self, dataset, transform=None):
#         self.dataset = dataset
#         self.transform = transform
#
#         # self.att_transform = T.ToTensor()
#     def __len__(self):
#         return len(self.dataset)
#
#     def __getitem__(self, index):
#         img_path, pid, camid, attribute = self.dataset[index]
#         img = read_image(img_path)
#         # attribute = cv2.imread(attribute)
#         # attribute = cv2.resize(attribute, (128, 256), interpolation=cv2.INTER_NEAREST)
#         # attribute = torch.tensor(attribute, dtype=torch.int64).unsqueeze(-1)
#
#         # attribute = F.interpolate(attribute,size=[256,128], mode="nearest")
#         # attribute = self.att_transform(attribute)
#
#         if self.transform is not None:
#             img = self.transform(img)
#
#         return img, pid, camid, img_path, attribute