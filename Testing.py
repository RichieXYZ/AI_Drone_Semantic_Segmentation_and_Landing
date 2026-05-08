import torch
import cv2
import albumentations as A
import segmentation_models_pytorch as smp

import ssl
ssl._create_default_https_context = ssl._create_stdlib_context


class_names = {
    0: 'unlabeled',
    1: 'paved-area',
    2: 'dirt',
    3: 'grass',
    4: 'gravel',
    5: 'water',
    6: 'rocks',
    7: 'pool',
    8: 'vegetation',
    9: 'roof',
    10: 'wall',
    11: 'window',
    12: 'door',
    13: 'fence',
    14: 'fence-pole',
    15: 'person',
    16: 'dog',
    17: 'car',
    18: 'bicycle',
    19: 'tree',
    20: 'bald-tree',
    21: 'ar-marker',
    22: 'obstacle',
}

# Device Setting - Check if GPU is available
# ---------------------------------------------
from Utility import set_device
device = set_device()


# Set paths to image folder
# ---------------------------------------------
data_dir = './Data'
n_classes = 23

from DroneDataset import DroneSegmentationDataset


# Image transformations
# ---------------------------------------------
height = 416
width  = 608

t_test = A.Resize(height, width, interpolation=cv2.INTER_NEAREST)
test_set = DroneSegmentationDataset(data_dir, t_test, split="test")


# Define and Load Best Model
# ---------------------------------------------
encoder = 'efficientnet-b0'

model = smp.UnetPlusPlus(encoder,
                encoder_weights='imagenet',
                classes=23,
                activation=None,
                decoder_attention_type=None,
                encoder_depth=5,
                decoder_channels=[256, 128, 64, 32, 16])

print("Loading Model : ", encoder)
# Set path to pretrained model
path = './Models/Unet++_eff.pt'
model = torch.load(path)

from Utility import show_pred_masks, show_errors
# show_pred_masks(model, test_set, device, encoder)
show_errors(model, test_set, device, encoder)


from Utility import test_predictions
test_acc, test_miou, class_accuracies, class_iou = test_predictions(model, test_set, n_classes=23, device=device)

print('Test Set mIoU ', "{:.2%}".format(test_miou))
print('Test Set Pixel Accuracy ', "{:.2%}".format(test_acc))

for key, value in class_names.items():
    print("Class", key, ": ", value, "\nAccuracy {:.2%}".format(class_accuracies[key]),
          "mIoU {:.2%}".format(class_iou[key]))
