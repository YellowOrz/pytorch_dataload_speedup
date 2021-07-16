#!/usr/bin/python
# -*- coding: utf-8 -*-
# 
# Developed by Shangchen Zhou <shangchenzhou@gmail.com>
'''ref: http://pytorch.org/docs/master/torchvision/transforms.html'''
import time
import cv2
import json
import numpy as np
import os
import io

import torch.utils.data.dataset
from datetime import datetime as dt
from enum import Enum, unique
import numpy as np
import torch
import torchvision.transforms.functional as F
from config import cfg
from PIL import Image
import random


def cuda_norm(x):
    if torch.cuda.is_available():
        x = x.cuda(non_blocking=True)
    x = x / 255.
    return x


class Compose(object):
    """ Composes several co_transforms together.
    For example:
    >>> transforms.Compose([
    >>>     transforms.CenterCrop(10),
    >>>     transforms.ToTensor(),
    >>>  ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, seq_blur, seq_clear, name):
        for t in self.transforms:
            seq_blur, seq_clear = t(seq_blur, seq_clear)
        return seq_blur, seq_clear


class ColorJitter(object):
    def __init__(self, color_adjust_para):
        """brightness [max(0, 1 - brightness), 1 + brightness] or the given [min, max]"""
        """contrast [max(0, 1 - contrast), 1 + contrast] or the given [min, max]"""
        """saturation [max(0, 1 - saturation), 1 + saturation] or the given [min, max]"""
        """hue [-hue, hue] 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5"""
        '''Ajust brightness, contrast, saturation, hue'''
        '''Input: PIL Image, Output: PIL Image'''
        self.brightness, self.contrast, self.saturation, self.hue = color_adjust_para

    def __call__(self, seq_blur, seq_clear):
        seq_blur = [Image.fromarray(np.uint8(img)) for img in seq_blur]
        seq_clear = [Image.fromarray(np.uint8(img)) for img in seq_clear]
        if self.brightness > 0:
            brightness_factor = np.random.uniform(max(0, 1 - self.brightness), 1 + self.brightness)
            seq_blur = [F.adjust_brightness(img, brightness_factor) for img in seq_blur]
            seq_clear = [F.adjust_brightness(img, brightness_factor) for img in seq_clear]

        if self.contrast > 0:
            contrast_factor = np.random.uniform(max(0, 1 - self.contrast), 1 + self.contrast)
            seq_blur = [F.adjust_contrast(img, contrast_factor) for img in seq_blur]
            seq_clear = [F.adjust_contrast(img, contrast_factor) for img in seq_clear]

        if self.saturation > 0:
            saturation_factor = np.random.uniform(max(0, 1 - self.saturation), 1 + self.saturation)
            seq_blur = [F.adjust_saturation(img, saturation_factor) for img in seq_blur]
            seq_clear = [F.adjust_saturation(img, saturation_factor) for img in seq_clear]

        if self.hue > 0:
            hue_factor = np.random.uniform(-self.hue, self.hue)
            seq_blur = [F.adjust_hue(img, hue_factor) for img in seq_blur]
            seq_clear = [F.adjust_hue(img, hue_factor) for img in seq_clear]

        seq_blur = [np.asarray(img) for img in seq_blur]
        seq_clear = [np.asarray(img) for img in seq_clear]

        seq_blur = [img.clip(0, 255) for img in seq_blur]
        seq_clear = [img.clip(0, 255) for img in seq_clear]

        return seq_blur, seq_clear


class RandomColorChannel(object):
    def __call__(self, seq_blur, seq_clear):
        random_order = np.random.permutation(3)

        seq_blur = [img[:, :, random_order] for img in seq_blur]
        seq_clear = [img[:, :, random_order] for img in seq_clear]

        return seq_blur, seq_clear


class RandomGaussianNoise(object):
    def __init__(self, gaussian_para):
        self.mu = gaussian_para[0]
        self.std_var = gaussian_para[1]

    def __call__(self, seq_blur, seq_clear):
        shape = seq_blur[0].shape
        gaussian_noise = np.random.normal(self.mu, self.std_var, shape)
        # only apply to blurry images
        seq_blur = [img + gaussian_noise for img in seq_blur]
        seq_blur = [img.clip(0, 1) for img in seq_blur]

        return seq_blur, seq_clear


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, seq_blur, seq_clear):
        seq_blur = [img / self.std - self.mean for img in seq_blur]
        seq_clear = [img / self.std - self.mean for img in seq_clear]

        return seq_blur, seq_clear


class CenterCrop(object):

    def __init__(self, crop_size):
        """Set the height and weight before and after cropping"""

        self.crop_size_h = crop_size[0]
        self.crop_size_w = crop_size[1]

    def __call__(self, seq_blur, seq_clear):
        input_size_h, input_size_w, _ = seq_blur[0].shape
        x_start = int(round((input_size_w - self.crop_size_w) / 2.))
        y_start = int(round((input_size_h - self.crop_size_h) / 2.))

        seq_blur = [img[y_start: y_start + self.crop_size_h, x_start: x_start + self.crop_size_w] for img in seq_blur]
        seq_clear = [img[y_start: y_start + self.crop_size_h, x_start: x_start + self.crop_size_w] for img in seq_clear]

        return seq_blur, seq_clear


class RandomCrop(object):

    def __init__(self, crop_size):
        """Set the height and weight before and after cropping"""
        self.crop_size_h = crop_size[0]
        self.crop_size_w = crop_size[1]

    def __call__(self, seq_blur, seq_clear):
        input_size_h, input_size_w, _ = seq_blur[0].shape
        x_start = random.randint(0, input_size_w - self.crop_size_w)
        y_start = random.randint(0, input_size_h - self.crop_size_h)

        seq_blur = [img[y_start: y_start + self.crop_size_h, x_start: x_start + self.crop_size_w] for img in seq_blur]
        seq_clear = [img[y_start: y_start + self.crop_size_h, x_start: x_start + self.crop_size_w] for img in seq_clear]

        return seq_blur, seq_clear


class RandomHorizontalFlip(object):
    """Randomly horizontally flips the given PIL.Image with a probability of 0.5 left-right"""

    def __call__(self, seq_blur, seq_clear):
        if random.random() < 0.5:
            '''Change the order of 0 and 1, for keeping the net search direction'''
            seq_blur = [np.copy(np.fliplr(img)) for img in seq_blur]
            seq_clear = [np.copy(np.fliplr(img)) for img in seq_clear]

        return seq_blur, seq_clear


class RandomVerticalFlip(object):
    """Randomly vertically flips the given PIL.Image with a probability of 0.5  up-down"""

    def __call__(self, seq_blur, seq_clear):
        if random.random() < 0.5:
            seq_blur = [np.copy(np.flipud(img)) for img in seq_blur]
            seq_clear = [np.copy(np.flipud(img)) for img in seq_clear]

        return seq_blur, seq_clear


class ToTensor(object):
    """Converts a numpy.ndarray (H x W x C) to a torch.FloatTensor of shape (C x H x W)."""

    def __call__(self, seq_blur, seq_clear):
        seq_blur = [np.transpose(img, (2, 0, 1)) for img in seq_blur]
        seq_clear = [np.transpose(img, (2, 0, 1)) for img in seq_clear]
        # handle numpy array
        seq_blur_tensor = [torch.from_numpy(img).float() for img in seq_blur]
        seq_clear_tensor = [torch.from_numpy(img).float() for img in seq_clear]

        return seq_blur_tensor, seq_clear_tensor


class DatasetType(Enum):
    TRAIN = 0
    TEST = 1


class VideoDataset(torch.utils.data.dataset.Dataset):
    """VideoDataset class used for PyTorch DataLoader"""

    def __init__(self, file_list_with_metadata, transforms=None):
        self.file_list = file_list_with_metadata
        self.transforms = transforms

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        name, seq_blur, seq_clear = self.get_datum(idx)
        seq_blur, seq_clear = self.transforms(seq_blur, seq_clear, name)
        return name, seq_blur, seq_clear

    def get_datum(self, idx):

        name = self.file_list[idx]['name']
        phase = self.file_list[idx]['phase']
        length = self.file_list[idx]['length']
        seq_blur_paths = self.file_list[idx]['seq_blur']
        seq_clear_paths = self.file_list[idx]['seq_clear']
        seq_blur = []
        seq_clear = []
        for i in range(length):
            img_blur = cv2.imread(seq_blur_paths[i]).astype(np.float32)
            img_clear = cv2.imread(seq_clear_paths[i]).astype(np.float32)
            seq_blur.append(img_blur)
            seq_clear.append(img_clear)

        if phase == 'train' and random.random() < 0.5:
            # random reverse
            seq_blur.reverse()
            seq_clear.reverse()

        return name, seq_blur, seq_clear


class VideoDataLoader:
    def __init__(self):
        self.img_blur_path_template = cfg.IMAGE_BLUR_PATH
        self.img_clear_path_template = cfg.IMAGE_CLEAR_PATH

        # Load all files of the dataset
        with io.open(cfg.DATASET_JSON_FILE_PATH, encoding='utf-8') as file:
            self.files_list = json.loads(file.read())

    def get_dataset(self, dataset_type, transforms=None):
        sequences = []
        # Load data for each sequence
        for file in self.files_list:
            if dataset_type == DatasetType.TRAIN and file['phase'] == 'train':
                name = file['name']
                phase = file['phase']
                samples = file['sample']
                sam_len = len(samples)
                seq_len = cfg.SEQ_LENGTH
                seq_num = int(sam_len / seq_len)
                for n in range(seq_num):
                    sequence = self.get_files_of_taxonomy(phase, name, samples[seq_len * n: seq_len * (n + 1)])
                    sequences.extend(sequence)

                if not seq_len % seq_len == 0:
                    sequence = self.get_files_of_taxonomy(phase, name, samples[-seq_len:])
                    sequences.extend(sequence)
                    seq_num += 1

                print('[INFO] %s Collecting files of Taxonomy [Name = %s]' % (dt.now(), name + ': ' + str(seq_num)))

            elif dataset_type == DatasetType.TEST and file['phase'] == 'test':
                name = file['name']
                phase = file['phase']
                samples = file['sample']
                sam_len = len(samples)
                seq_len = cfg.SEQ_LENGTH
                seq_num = int(sam_len / seq_len)
                for n in range(seq_num):
                    sequence = self.get_files_of_taxonomy(phase, name, samples[seq_len * n: seq_len * (n + 1)])
                    sequences.extend(sequence)

                if not seq_len % seq_len == 0:
                    sequence = self.get_files_of_taxonomy(phase, name, samples[-seq_len:])
                    sequences.extend(sequence)
                    seq_num += 1

                print('[INFO] %s Collecting files of Taxonomy [Name = %s]' % (dt.now(), name + ': ' + str(seq_num)))

        print('[INFO] %s Complete collecting files of the dataset for %s. Seq Number: %d.\n' % (
            dt.now(), dataset_type.name, len(sequences)))
        return VideoDataset(sequences, transforms)

    def get_files_of_taxonomy(self, phase, name, samples):
        n_samples = len(samples)
        seq_blur_paths = []
        seq_clear_paths = []
        sequence = []

        for sample_idx, sample_name in enumerate(samples):
            # Get file path of img
            img_blur_path = self.img_blur_path_template % (phase, name, sample_name)
            img_clear_path = self.img_clear_path_template % (phase, name, sample_name)
            if os.path.exists(img_blur_path) and os.path.exists(img_clear_path):
                seq_blur_paths.append(img_blur_path)
                seq_clear_paths.append(img_clear_path)

        if not seq_blur_paths == [] and not seq_clear_paths == []:
            sequence.append({
                'name': name,
                'phase': phase,
                'length': n_samples,
                'seq_blur': seq_blur_paths,
                'seq_clear': seq_clear_paths,
            })
        return sequence

if __name__ == '__main__':
    epoch = 10
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    dataset_loader = VideoDataLoader()

    train_transforms = Compose([RandomCrop(cfg.CROP_IMG_SIZE), ColorJitter(cfg.COLOR_JITTER),
                                Normalize(mean=cfg.MEAN, std=cfg.STD), RandomVerticalFlip(), RandomHorizontalFlip(),
                                RandomColorChannel(), RandomGaussianNoise(cfg.GAUSSIAN), ToTensor()])
    train_data_loader = torch.utils.data.DataLoader(
        dataset=dataset_loader.get_dataset(DatasetType.TRAIN, train_transforms),
        batch_size=5, num_workers=5, pin_memory=True, shuffle=True)
    s = time.time()
    for i in range(epoch):
        t = time.time()
        for seq_idx, (_, seq_blur, seq_clear) in enumerate(train_data_loader):
            if seq_idx == 0:
                print("当前epoch在CPU上耗时{:.2f}\t".format(time.time() - t), end='')
            for batch_idx, [img_blur, img_clear] in enumerate(zip(seq_blur, seq_clear)):
                img_blur = cuda_norm(img_blur)
                img_clear = cuda_norm(img_clear)
        print("单个epoch总耗时{:.2f}".format(time.time() - t))
    print("{}个epoch耗时{:.2f}".format(epoch, time.time() - s))
