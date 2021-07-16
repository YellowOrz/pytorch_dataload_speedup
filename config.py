from easydict import EasyDict as edict
import os
import socket

cfg = edict()

cfg.STD = [255.0, 255.0, 255.0]
cfg.MEAN = [0.0, 0.0, 0.0]
cfg.CROP_IMG_SIZE = [320, 448]  # Crop image size: height, width
cfg.GAUSSIAN = [0, 1e-4]  # mu, std_var
cfg.COLOR_JITTER = [0.2, 0.15, 0.3, 0.1]  # brightness, contrast, saturation, hue
cfg.SEQ_LENGTH = 100            # 100,50,20
cfg.DATASET_JSON_FILE_PATH = '../STFAN/datasets/DVD.json'
cfg.DATASET_ROOT = '../STFAN/datasets/DVD'
cfg.IMAGE_BLUR_PATH = os.path.join(cfg.DATASET_ROOT, '%s/%s/input/%s.jpg')
cfg.IMAGE_CLEAR_PATH = os.path.join(cfg.DATASET_ROOT, '%s/%s/GT/%s.jpg')
