import matplotlib.pyplot as plt
import torch
import numpy as np
import torch.nn.functional as F
from torchvision import transforms as T
from tqdm import tqdm
import pandas as pd
import os
import math
import cv2


# Training
def set_device():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print("Device : ", device)

    return device

# Get learning rate for each epoch
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def plot_results(history, model_name):
    # Create a 2x2 subplot grid
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))  # (rows, cols)
    plt.title(model_name)

    # Plot data
    axes[0, 0].plot(history["test_losses"], label='val', marker='o')
    axes[0, 0].plot(history["train_losses"], label='train', marker='o')
    axes[0, 0].set_title(model_name + ' Loss')
    axes[0, 0].set_xlabel('epoch')
    axes[0, 0].set_ylabel('loss')
    axes[0, 0].legend()
    axes[0, 0].grid()

    axes[0, 1].plot(history["val_iou"], label='val', marker='o')
    axes[0, 1].plot(history["train_iou"], label='train', marker='o')
    axes[0, 1].set_title(model_name + ' IoU')
    axes[0, 1].set_xlabel('epoch')
    axes[0, 1].set_ylabel('IoU')
    axes[0, 1].legend()
    axes[0, 1].grid()

    axes[1, 0].plot(history["lrs"], label='val', marker='o')
    axes[1, 0].set_title(model_name + ' LR ')
    axes[1, 0].set_xlabel('epoch')
    axes[1, 0].set_ylabel('LR')
    axes[1, 0].legend()
    axes[1, 0].grid()

    axes[1, 1].plot(history["val_acc"], label='val', marker='o')
    axes[1, 1].plot(history["train_acc"], label='train', marker='o')
    axes[1, 1].set_title(model_name + ' Accuracy')
    axes[1, 1].set_xlabel('epoch')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].legend()
    axes[1, 1].grid()

    # Create directory if it doesn't exist
    saving_dir = "Results/" + model_name
    if not os.path.exists("Results"):
        os.makedirs("Results")
    if not os.path.exists(saving_dir):
        os.makedirs(saving_dir)

    # Save Plot
    plt.tight_layout()
    plt.savefig(saving_dir + "/Training_Results.png")

    # Save numerical result as dataframe
    print(history)
    df = pd.DataFrame(history)
    df.to_csv(saving_dir+"/results.csv", index=False)


# Plot each list on the same plot
def plot_losses(Losses, encoder_name):
    plt.figure()
    for data in Losses:
        plt.plot(data, marker='o')

    # Adding labels and title
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.title("Plot of List of Lists")
    plt.legend([encoder_name[i] for i in range(len(Losses))])  # Adding a legend

    # Show the plot
    plt.show()
    plt.savefig("Results/Training_Results.png")

# Testing
def compute_accuracy_metrics(output, mask, n_classes = 23):

    with torch.no_grad():

        prediction_mask = torch.argmax(F.softmax(output, dim=1), dim=1)

        # Compute Accuracy
        correct = torch.eq(prediction_mask, mask).int()
        accuracy = float(correct.sum()) / float(correct.numel())

        # Compute mIoU
        pred_mask = prediction_mask.contiguous().view(-1)
        mask = mask.contiguous().view(-1)

        smooth = 1e-10
        iou_per_class = []
        for clas in range(0, n_classes):  # loop per pixel class
            true_class = pred_mask == clas
            true_label = mask == clas

            if true_label.long().sum().item() == 0 and true_label.long().sum().item() == 0:
                    iou_per_class.append(np.nan)
            else:
                intersect = torch.logical_and(true_class, true_label).sum().float().item()
                union = torch.logical_or(true_class, true_label).sum().float().item()

                iou = (intersect + smooth) / (union + smooth)
                iou_per_class.append(iou)

                mIoU = np.nanmean(iou_per_class)

        return accuracy, mIoU

def predict_mask_metrics(model, image, mask, device):
    model.eval()
    model.to(device)

    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    t = T.Compose([T.ToTensor(), T.Normalize(mean, std)])

    image = t(image)
    image = image.to(device)
    mask  = mask.to(device)

    with torch.no_grad():
        image = image.unsqueeze(0)
        mask  = mask.unsqueeze(0)

        output = model(image)

        acc, score = compute_accuracy_metrics(output, mask)

        masked = torch.argmax(output, dim=1)
        masked = masked.cpu().squeeze(0)

    return masked, acc, score


def class_accuracy(pred_mask, mask, n_classes = 23):

    accuracies = []
    iou_per_class = []
    smooth = 1e-10

    for i in range(n_classes):
        class_pred = pred_mask == i
        class_mask = mask == i

        # Calculate True Positives (TP)
        TP = torch.sum((class_pred == 1) & (class_mask == 1)).item()
        # Calculate False Negatives (FN)
        FN = torch.sum((class_pred == 0) & (class_mask == 1)).item()

        if (TP + FN) == 0:
            accuracy = 1.0  # If there are no true positives or false negatives, accuracy is 1
        else:
            accuracy = TP / (TP + FN)

        if class_mask.long().sum().item() == 0 and class_pred.long().sum().item() == 0:  # no exist label in this loop
                iou_per_class.append(np.nan)
                accuracies.append(np.nan)
        else:
            intersect = torch.logical_and(class_pred, class_mask).sum().float().item()
            union = torch.logical_or(class_pred, class_mask).sum().float().item()

            iou = (intersect + smooth) / (union + smooth)
            iou_per_class.append(iou)
            accuracies.append(accuracy)

    return accuracies, iou_per_class


def test_predictions(model, test_set, n_classes, device):

    accuracy = []
    score_iou = []
    total_class_acc = [0]*n_classes
    class_iou = [0]*n_classes

    class_elements = [0]*n_classes

    # Iterate over all images in the test set
    for i in tqdm(range(len(test_set))):
        img, mask = test_set[i]

        # Predictions
        pred_mask, acc, score = predict_mask_metrics(model, img, mask, device)

        accuracies, ious = class_accuracy(pred_mask, mask)

        for j in range(len(accuracies)):

            if not math.isnan(accuracies[j]):
                total_class_acc[j] += accuracies[j]

                class_iou[j] += ious[j]
                class_elements[j] += 1

        accuracy.append(acc)
        score_iou.append(score)

    total_class_acc = [total_class_acc[i] / class_elements[i] for i in range(len(total_class_acc))]
    class_iou = [class_iou[i] / class_elements[i] for i in range(len(class_iou))]

    return np.mean(accuracy), np.mean(score_iou), total_class_acc, class_iou

def show_pred_masks(model, test_set, device, name):

    for i in range(len(test_set)):
        image, mask = test_set[i]
        pred_mask, acc, score = predict_mask_metrics(model, image, mask, device)

        fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(15,5))
        ax1.imshow(image)
        ax1.set_title('Picture')

        ax2.imshow(mask, cmap="plasma")
        ax2.set_title('Ground truth')
        ax2.set_axis_off()

        ax3.imshow(pred_mask, cmap="plasma")
        ax3.set_title(name + ' | mIoU {:.3f}'.format(score))
        ax3.set_axis_off()
        plt.show()

    return


def show_errors(model, test_set, device, name):

    for i in range(len(test_set)):
        image, mask = test_set[i]
        pred_mask, acc, score = predict_mask_metrics(model, image, mask, device)

        comparison = (mask == pred_mask).to(dtype=torch.uint8)

        fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(15,5))
        ax1.imshow(mask, cmap="plasma")
        ax1.set_title('Ground Truth')
        ax1.set_axis_off()

        ax2.imshow(pred_mask, cmap="plasma")
        ax2.set_title('Prediction' + ' | mIoU {:.3f}'.format(score))
        ax2.set_axis_off()

        ax3.imshow(comparison.numpy(), cmap="gray")
        ax3.set_title("Errors, (1: Match, 0: Difference)")
        ax3.set_axis_off()
        plt.show()

    return

def find_largest_inscribed_circle(image, channel_index):

    max_radius = 0
    center = (0, 0)

    for i in range(len(channel_index)):
        # Select the specific channel
        channel = image[:, :, channel_index[i]]

        # Convert the channel to binary image
        _, binary = cv2.threshold(channel, 0.5, 255, cv2.THRESH_BINARY)

        # Create a binary mask for the image borders
        binary[:, 1] = 0
        binary[:, -1] = 0
        binary[1, :] = 0
        binary[-1, :] = 0

        # Perform distance transform
        dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)

        # Find the maximum value and its location in the distance transform
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(dist_transform)
        radius = int(max_val)
        # max_loc is the center and max_val is the radius of the largest inscribed circle

        if radius > max_radius:
            center = max_loc
            max_radius = radius

    return center, max_radius

def landing_point_prediction(mask):
    # Number of classes
    num_classes = 24

    # Initialize an empty array for the one-hot encoded map
    one_hot_map = np.zeros((mask.shape[0], mask.shape[1], num_classes), dtype=np.uint8)

    # Populate the one-hot encoded map
    for c in range(num_classes):
        one_hot_map[:, :, c] = (mask == c).astype(np.uint8)

    # Assume the desired channel index is known (e.g., 0 for the first channel)
    channel_index = [1, 3]  # Land on grass or paved area
    # Find the largest inscribed circle in the selected channel
    center, radius = find_largest_inscribed_circle(one_hot_map, channel_index)

    return center, radius

def predict_mask(model, image, device):
    model.eval()
    model.to(device)

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    t = T.Compose([T.ToTensor(), T.Normalize(mean, std)])

    image = t(image)
    image = image.to(device)

    with torch.no_grad():
        image = image.unsqueeze(0)
        output = model(image)
        mask = torch.argmax(output, dim=1)
        mask = mask.cpu().squeeze(0)

    return mask

def show_landing_points(model, test_set):

    device = set_device()

    for i in range(len(test_set)):
        image, _ = test_set[i]
        pred_mask = predict_mask(model, image, device)

        c, r = landing_point_prediction(np.asarray(pred_mask))

        circle1 = plt.Circle(c, r, color='yellow', fill=False)

        fig, (ax1, ax2) = plt.subplots(1,2, figsize=(15,5))
        ax1.imshow(image)
        ax1.set_title('Picture')

        ax2.imshow(pred_mask, cmap='plasma')
        ax2.set_title('UNet++ Prediction')
        ax2.set_axis_off()
        ax2.add_patch(circle1)
        ax2.scatter(x=c[0], y=c[1], c='r')
        plt.show()

    return