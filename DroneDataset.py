import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision import transforms as T
import torch
from PIL import Image
import cv2
import pandas as pd
import ssl
ssl._create_default_https_context = ssl._create_stdlib_context
import albumentations as A
import os


# 0. Insert path to data folder
# ---------------------------------------------
data_dir = './Data'


# 1. Dataset Class
# ---------------------------------------------
class DroneSegmentationDataset(Dataset):

    # Class constructor
    def __init__(self, data_dir, transform, split):

        # Set data directory
        self.dir = data_dir
        image_path = data_dir + '/original_images'

        # Count the number of images and create a number list
        num_files = len([f for f in os.listdir(image_path) if os.path.isfile(os.path.join(image_path, f))])
        number_list = [f"{i:03}" for i in range(num_files)]

        # Data train/validation/test splitting
        # Fixed proportions 0.75/0.15/0.1

        if split == "train":
            number_list = number_list[:int(0.75 * num_files)]

        elif split == "val":
            number_list = number_list[int(0.75 * num_files):int(0.9 * num_files)]

        else:
            number_list = number_list[int(0.9 * num_files):]

        # Set class variables
        self.index = number_list
        self.split = split
        self.transform = transform

        self.mean = [0.485, 0.456, 0.406]
        self.std  = [0.229, 0.224, 0.225]

    # len method -> return number of elements in the dataset
    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):

        # Access index in the number list
        index = self.index[idx]

        # Set paths to image and mask
        img_path = self.dir + '/original_images/image' + index + '.jpg'
        mask_path = self.dir + '/label_images_semantic/mask' + index + '.png'

        # read the image and mask
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Transform image
        if self.transform is not None:
            aug = self.transform(image=img, mask=mask)
            img = Image.fromarray(aug['image'])
            mask = aug['mask']

        if self.split == "test":
            t = None
        else:
            t = T.Compose([T.ToTensor(), T.Normalize(self.mean, self.std)])

        if t is not None:
            img = t(img)

        mask = torch.from_numpy(mask).long()

        return img, mask


# Image transformations
# ---------------------------------------------
height = 416
width  = 608
t_test = A.Resize(height, width, interpolation=cv2.INTER_NEAREST)

# Create Datasets
# ---------------------------------------------
train_set = DroneSegmentationDataset(data_dir, t_test, split="train")
val_set   = DroneSegmentationDataset(data_dir, t_test, split="val")
test_set  = DroneSegmentationDataset(data_dir, t_test, split="test")

# Show a sample image
# image, mask = test_set.__getitem__(0)
# plt.imshow(image)
# plt.show()


