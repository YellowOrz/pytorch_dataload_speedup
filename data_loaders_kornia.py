from tqdm import tqdm
from config import cfg
from math import pi
import cv2
import json
import numpy as np
import os
import io
import random
import torch.utils.data.dataset
import kornia as K
from datetime import datetime as dt
from enum import Enum, unique
import torch.nn as nn
import time


def cuda_norm(x):
    if torch.cuda.is_available():
        x = x.cuda(non_blocking=True)
    x = x/255.
    return x


class DatasetType(Enum):
    TRAIN = 0
    TEST = 1


class VideoDatasetKorina(torch.utils.data.dataset.Dataset):
    """VideoDeblurDataset class used for PyTorch DataLoader"""

    def __init__(self, file_list_with_metadata):
        self.file_list = file_list_with_metadata

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        name = self.file_list[idx]['name']
        phase = self.file_list[idx]['phase']
        length = self.file_list[idx]['length']
        seq_blur_paths = self.file_list[idx]['seq_blur']
        seq_clear_paths = self.file_list[idx]['seq_clear']
        seq_blur = []
        seq_clear = []
        if phase in ["train", "resume"]:
            # crop region
            crop_h = cfg.CROP_IMG_SIZE[0]
            crop_w = cfg.CROP_IMG_SIZE[1]
            x1 = random.randint(0, 1280 - crop_w)
            y1 = random.randint(0, 720 - crop_h)
            x2 = x1 + crop_w
            y2 = y1 + crop_h

        for i in range(length):
            img_blur = cv2.imread(seq_blur_paths[i]).transpose(2, 0, 1)
            img_clear = cv2.imread(seq_clear_paths[i]).transpose(2, 0, 1)
            if phase in ["train", "resume"]:
                img_blur = img_blur[:, y1: y2, x1: x2]
                img_clear = img_clear[:, y1: y2, x1: x2]

            seq_blur.append(torch.from_numpy(img_blur).float().cuda())
            seq_clear.append(torch.from_numpy(img_clear).float().cuda())

        if phase == 'train' and random.random() < 0.5:
            # random reverse
            seq_blur.reverse()
            seq_clear.reverse()

        return name, seq_blur, seq_clear


class VideoDataLoaderKorina:
    def __init__(self):
        self.img_blur_path_template = cfg.IMAGE_BLUR_PATH
        self.img_clear_path_template = cfg.IMAGE_CLEAR_PATH

        # Load all files of the dataset
        with io.open(cfg.DATASET_JSON_FILE_PATH, encoding='utf-8') as file:
            self.files_list = json.loads(file.read())

    def get_dataset(self, dataset_type):
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
        return VideoDatasetKorina(sequences)

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
            sequence.append({'name': name,
                             'phase': phase,
                             'length': n_samples,
                             'seq_blur': seq_blur_paths,
                             'seq_clear': seq_clear_paths})
        return sequence


class Transforms(nn.Module):
    def __init__(self):
        super(Transforms, self).__init__()
        self.color_jitter = random.random() < 0.5
        self.brightness = np.random.uniform(-cfg.COLOR_JITTER[0], cfg.COLOR_JITTER[0])  # 亮度通过加法调整，范围[0,1]
        self.contrast = np.random.uniform(max(0, 1 - cfg.COLOR_JITTER[1]), 1)  # 对比度通过乘法调整
        self.saturation = np.random.uniform(max(0, 1 - cfg.COLOR_JITTER[2]), 1)  # 饱和度通过HSV中对S加法调整
        self.hue = np.random.uniform(max(0, 1 - cfg.COLOR_JITTER[3]), 1)  # 色调通过HSV中对h加法调整

        self.hflip = random.random() < 0.5
        self.vflip = random.random() < 0.5
        self.channel_shuffle_order = np.random.permutation(3)
        self.channel_shuffle = False in (self.channel_shuffle_order == np.array([0, 1, 2]))
        self.gaussian_noise = (torch.randn([3] + cfg.CROP_IMG_SIZE).cuda() + cfg.GAUSSIAN[0]) * cfg.GAUSSIAN[1]

    def forward(self, img):
        # Color Jitter
        if self.color_jitter:
            img = K.enhance.adjust_brightness(img, self.brightness)
            img = K.enhance.adjust_contrast(img, self.contrast)
            img = self.adjust_saturation_hue(img, self.saturation, self.hue)

        # 随机翻转
        if self.hflip:
            img = K.geometry.transform.hflip(img)
        if self.vflip:
            img = K.geometry.transform.vflip(img)

        # Channel Shuffle
        if self.channel_shuffle:
            img = img[:, self.channel_shuffle_order, :, :]  # N * C * H * W

        # 高斯噪声
        img = img + self.gaussian_noise
        return img

    def adjust_saturation_hue(self, x, saturation_factor, hue_factor):
        # TODO: 把brightness和contrast也整合进来得了
        x = K.color.rgb_to_hsv(x)

        if isinstance(saturation_factor, float):
            saturation_factor = torch.as_tensor(saturation_factor)
        if isinstance(hue_factor, float):
            hue_factor = torch.as_tensor(hue_factor)

        saturation_factor = saturation_factor.to(x.device).to(x.dtype)
        hue_factor = hue_factor.to(x.device, x.dtype)

        for _ in x.shape[1:]:
            saturation_factor = torch.unsqueeze(saturation_factor, dim=-1)
        for _ in x.shape[1:]:
            hue_factor = torch.unsqueeze(hue_factor, dim=-1)

        h, s, v = torch.chunk(x, chunks=3, dim=-3)

        s = torch.clamp(s * saturation_factor, min=0, max=1)
        divisor = 2 * pi
        h = torch.fmod(h + hue_factor, divisor)

        x = torch.cat([h, s, v], dim=-3)

        x = K.color.hsv_to_rgb(x)
        return x


if __name__ == '__main__':
    epoch = 10
    dataset_loader = VideoDataLoaderKorina()

    train_data_loader = torch.utils.data.DataLoader(dataset=dataset_loader.get_dataset(DatasetType.TRAIN),
                                                    batch_size=5, pin_memory=True, num_workers=5, shuffle=True)
    # test_data_loader = torch.utils.data.DataLoader(dataset=dataset_loader.get_dataset(DatasetType.TEST),
    #                                                batch_size=1, pin_memory=True, num_workers=2, shuffle=False)
    s = time.time()
    for i in range(epoch):
        t = time.time()
        # train
        for seq_idx, (_, seq_blur, seq_clear) in enumerate(train_data_loader):
            if seq_idx == 0:
                print("当前epoch在CPU上耗时{:.2f}\t".format(time.time() - t), end='')
            train_transform = Transforms().cuda()
            train_transform.eval()
            for batch_idx, [img_blur, img_clear] in enumerate(zip(seq_blur, seq_clear)):
                # print(batch_idx, end='')
                with torch.no_grad():
                    img_blur = train_transform(cuda_norm(img_blur))
                    img_clear = train_transform(cuda_norm(img_clear))

        # test
        # for seq_idx, (_, seq_blur, seq_clear) in enumerate(test_data_loader):
        #     for batch_idx, [img_blur, img_clear] in enumerate(zip(seq_blur, seq_clear)):
        #         img_blur = img_blur.cuda()
        #         img_clear = img_clear.cuda()
        print("单个epoch总耗时{:.2f}".format(time.time() - t))
    print("{}个epoch耗时{:.2f}".format(epoch, time.time() - s))
