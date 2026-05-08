import matplotlib.pyplot as plt
from PIL import Image
import cv2
import ssl
ssl._create_default_https_context = ssl._create_stdlib_context
from torchvision import transforms as T
import albumentations as A
import glob

# Insert path to data folder
data_dir = './Data'
img_folder  = data_dir + "/original_images"
mask_folder = data_dir + "/RGB_color_image_masks"

image_paths = glob.glob(img_folder + "/*.jpg")
image_paths.sort()

mask_paths = glob.glob(mask_folder + "/*.png")
mask_paths.sort()

height = 416
width  = 608

# Image transformations
# ---------------------------------------------
def image_augmentation(height, width):

    weather_conditions = A.Compose([
                    A.RandomShadow(
                    shadow_roi=(0, 0, 1, 1),  # Region of interest for shadows (x_min, y_min, x_max, y_max)
                    shadow_dimension=5,  # Approximate polygon complexity for the shadow
                    p=0.05),  # Probability of applying the augmentation)

                    A.RandomRain(
                    drop_length=15,
                    drop_width=1,
                    drop_color=(200, 200, 200),
                    blur_value=2,
                    brightness_coefficient=0.8,
                    rain_type="heavy",
                    p=1),

                    A.RandomSunFlare(
                    flare_roi=(0, 0, 1, 1),
                    src_radius=100,
                    src_color=(255, 255, 255),
                    p=0.25),

                    A.RandomSnow(
                    brightness_coeff=2.5,
                    p=0.05),

                    A.RandomFog(
                    alpha_coef=0.1,
                    p=0.05)])

    damage_adaptation = A.Compose([A.Defocus(alias_blur=(0.1, 0.5),
                                             p=0.05),

                                   # pixel_dropout
                                   A.PixelDropout(dropout_prob=0.01, p=0.05)])

    spatial_transforms = A.Compose([A.HorizontalFlip(p=0.5),
                                    A.VerticalFlip(p=0.5),
                                    A.Affine(p=0.35,
                                        interpolation=cv2.INTER_LINEAR),
                                    A.Perspective(
                                        scale=(0.05, 0.25),
                                        interpolation=cv2.INTER_LINEAR,
                                        p=0.5)])

    t_train = A.Compose([A.Resize(height, width, interpolation=cv2.INTER_NEAREST),

                         spatial_transforms,
                         damage_adaptation,
                         weather_conditions,

                         A.GridDistortion(p=0.05),
                         A.RandomBrightnessContrast((0, 0.5), (0, 0.5)),
                         ])

    return t_train


# Show an example image
# ---------------------------------------------
import random
image_idx = random.randint(1, 400)

img = cv2.imread(image_paths[image_idx])
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
mask = cv2.imread(mask_paths[image_idx])

# Plots
def show_augmentation(img, mask):
    fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(20,10))
    ax1.imshow(img)
    ax1.set_title('Picture')

    ax2.imshow(mask, cmap='viridis')
    ax2.set_title('Ground truth')
    ax2.set_axis_off()

    ax3.imshow(img)
    ax3.imshow(mask, alpha=0.5, cmap='viridis')
    ax3.set_title('Picture with Mask Applied')

    plt.show()

    # Image transformations
    # ---------------------------------------------
    height = 416
    width  = 608

    t_train = image_augmentation(height,width)
    aug = t_train(image=img, mask=mask)
    img1 = Image.fromarray(aug['image'])
    mask1 = aug['mask']

    fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(20,10))
    ax1.imshow(img1)
    ax1.set_title('Picture')
    ax1.set_axis_off()

    ax2.imshow(mask1, cmap='viridis')
    ax2.set_title('Ground truth')
    ax2.set_axis_off()

    ax3.imshow(img1)
    ax3.imshow(mask1, alpha=0.5, cmap='viridis')
    ax3.set_title('Picture with Mask Applied')
    ax3.set_axis_off()

    plt.show()


def show_comparison(img, mask):

    # Image transformations
    # ---------------------------------------------
    height = 416
    width  = 608

    t_train = image_augmentation(height,width)
    aug = t_train(image=img, mask=mask)
    img1 = Image.fromarray(aug['image'])

    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(20,10))
    ax1.imshow(img)
    ax1.set_title('Image')
    ax1.set_axis_off()

    ax2.imshow(img1)
    ax2.set_title('Augmented')
    ax2.set_axis_off()

    plt.show()


show_comparison(img, mask)
