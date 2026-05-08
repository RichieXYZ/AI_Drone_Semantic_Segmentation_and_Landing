import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import transforms as T
import cv2
import albumentations as A
import segmentation_models_pytorch as smp

import ssl
ssl._create_default_https_context = ssl._create_stdlib_context


# Set paths to image folder
# ---------------------------------------------
# Insert path to data folder
data_dir = './Data'
n_classes = 23

from DroneDataset import DroneSegmentationDataset


# Image transformations
# ---------------------------------------------
height = 416
width  = 608

t_test = A.Resize(height, width, interpolation=cv2.INTER_NEAREST)
test_set = DroneSegmentationDataset(data_dir, t_test, split="test")

encoder_name = "efficientnet-b0"
model = smp.UnetPlusPlus(encoder_name,
                 encoder_weights='imagenet',
                 classes=23,
                 activation=None,
                 encoder_depth=5,
                 decoder_channels=[256, 128, 64, 32, 16])

print("Loading Model : " + encoder_name + "...")
# Set Path to pretrained model
path = '/Users/riccardo/PycharmProjects/Ai_2024/Drone_segmentation/DroneSegmentation_Gabrieli/Models/Unet++_eff.pt'
model = torch.load(path)

from Utility import show_landing_points

show_landing_points(model, test_set)

