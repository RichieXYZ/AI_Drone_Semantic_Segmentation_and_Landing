import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import cv2
import albumentations as A
import time
from tqdm import tqdm
import segmentation_models_pytorch as smp
import ssl
from Utility import compute_accuracy_metrics
from Utility import get_lr
from collections import defaultdict
from Utility import plot_results

ssl._create_default_https_context = ssl._create_stdlib_context

# 0. Device Setting - Check if GPU is available
# ---------------------------------------------
from Utility import set_device

device = set_device()

# Set paths to image folder
# ---------------------------------------------
data_dir = './Data'
n_classes = 23
height = 416
width = 608

# 2. Image Augmentations
# ---------------------------------------------
from ImageAugmentation import image_augmentation
from DroneDataset import DroneSegmentationDataset

t_train = image_augmentation(height, width)
t_val = A.Resize(height, width, interpolation=cv2.INTER_NEAREST)

train_set = DroneSegmentationDataset(data_dir, t_train, split="train")
val_set = DroneSegmentationDataset(data_dir, t_val, split="val")

# Create Dataloaders
# ---------------------------------------------
batch_size = 3
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)


def Training_Function(model, train_loader, val_loader, criterion, optimizer, scheduler):
    # Create history dictionary
    history = defaultdict(list)

    # Initialize variables
    min_loss = np.inf
    patience = 0

    best_model = model
    model.to(device)

    # training + validation loop
    epoch = 0
    fit_time = time.time()  # Time of training
    while True:

        running_loss = 0
        iou_score = 0
        accuracy_score = 0

        model.train()
        since = time.time()
        for i, data in enumerate(tqdm(train_loader)):
            image, mask = data
            image = image.to(device)
            mask = mask.to(device)

            # forward
            output = model(image)
            loss = criterion(output, mask)

            # evaluation metrics
            accuracy, mIoU = compute_accuracy_metrics(output, mask)
            iou_score += mIoU
            accuracy_score += accuracy
            running_loss += loss.item()

            # backward
            loss.backward()
            optimizer.step()  # update weight
            optimizer.zero_grad()  # reset gradient

        model.eval()
        test_loss = 0
        test_accuracy = 0
        val_iou_score = 0

        # validation loop
        with torch.no_grad():
            for i, data in enumerate(tqdm(val_loader)):
                # reshape to 9 patches from single image, delete batch size
                image, mask = data

                image = image.to(device)
                mask = mask.to(device)

                output = model(image)

                # evaluation metrics
                accuracy, mIoU = compute_accuracy_metrics(output, mask)
                val_iou_score += mIoU
                test_accuracy += accuracy

                # loss
                loss = criterion(output, mask)
                test_loss += loss.item()

        cur_val_loss = test_loss / len(val_loader)

        # calculation of mean for each batch
        history["lrs"].append(get_lr(optimizer))
        history["train_losses"].append(running_loss / len(train_loader))
        history["test_losses"].append(cur_val_loss)

        if cur_val_loss < min_loss:
            print('Loss Decreasing.. {:.3f} >> {:.3f} '.format(min_loss, cur_val_loss))
            min_loss = cur_val_loss
            best_model = model
            patience = 0
        else:
            patience += 1
            print(f'Loss Not Decrease for {patience} time')

            # Ending Condition
            if patience == 10:
                print('Loss not decrease for {} times, Stop Training' 'Loss Decreasing..  '.format(patience))
                break

        scheduler.step(cur_val_loss)

        # Accuracy Metrics
        history["val_iou"].append(val_iou_score / len(val_loader))
        history["train_iou"].append(iou_score / len(train_loader))
        history["train_acc"].append(accuracy_score / len(train_loader))
        history["val_acc"].append(test_accuracy / len(val_loader))

        print("\nEpoch:{}..".format(epoch + 1),
              "\nTrain Loss: {:.3f}..".format(running_loss / len(train_loader)),
              "\nVal Loss: {:.3f}..".format(test_loss / len(val_loader)),
              "\nTrain mIoU:{:.3f}..".format(iou_score / len(train_loader)),
              "\nVal mIoU: {:.3f}..".format(val_iou_score / len(val_loader)),
              "\nTrain Acc:{:.3f}..".format(accuracy_score / len(train_loader)),
              "\nVal Acc:{:.3f}..".format(test_accuracy / len(val_loader)),
              "\nTime: {:.2f}m".format((time.time() - since) / 60))

        epoch += 1

    print('Total time: {:.2f} m'.format((time.time() - fit_time) / 60))

    return history, best_model, min_loss


def Exploratory_Training(n_epochs, model, train_loader, val_loader, criterion, optimizer, scheduler):
    # Create history dictionary
    history = defaultdict(list)

    # Initialize variables
    min_loss = np.inf
    best_model = model
    model.to(device)

    # training + validation loop
    epoch = 0
    fit_time = time.time()  # Time of training
    while epoch < n_epochs:

        running_loss = 0
        iou_score = 0
        accuracy_score = 0

        model.train()
        since = time.time()
        for i, data in enumerate(tqdm(train_loader)):
            image, mask = data
            image = image.to(device)
            mask = mask.to(device)

            # forward
            output = model(image)
            loss = criterion(output, mask)

            # evaluation metrics
            accuracy, mIoU = compute_accuracy_metrics(output, mask)
            iou_score += mIoU
            accuracy_score += accuracy
            running_loss += loss.item()

            # backward
            loss.backward()
            optimizer.step()  # update weight
            optimizer.zero_grad()  # reset gradient

            # step the learning rate

        model.eval()
        test_loss = 0
        test_accuracy = 0
        val_iou_score = 0

        # validation loop
        with torch.no_grad():
            for i, data in enumerate(tqdm(val_loader)):
                # reshape to 9 patches from single image, delete batch size
                image, mask = data

                image = image.to(device)
                mask = mask.to(device)

                output = model(image)

                # evaluation metrics
                accuracy, mIoU = compute_accuracy_metrics(output, mask)
                val_iou_score += mIoU
                test_accuracy += accuracy

                # loss
                loss = criterion(output, mask)
                test_loss += loss.item()

        cur_val_loss = test_loss / len(val_loader)

        # calculation of mean for each batch
        history["lrs"].append(get_lr(optimizer))
        history["train_losses"].append(running_loss / len(train_loader))
        history["test_losses"].append(cur_val_loss)

        if cur_val_loss < min_loss:
            print('Loss Decreasing.. {:.3f} >> {:.3f} '.format(min_loss, cur_val_loss))
            min_loss = cur_val_loss
            best_model = model

        scheduler.step(cur_val_loss)

        # Accuracy Metrics
        history["val_iou"].append(val_iou_score / len(val_loader))
        history["train_iou"].append(iou_score / len(train_loader))
        history["train_acc"].append(accuracy_score / len(train_loader))
        history["val_acc"].append(test_accuracy / len(val_loader))

        print("Epoch:{}..".format(epoch + 1),
              "Train Loss: {:.3f}..".format(running_loss / len(train_loader)),
              "Val Loss: {:.3f}..".format(test_loss / len(val_loader)),
              "Train mIoU:{:.3f}..".format(iou_score / len(train_loader)),
              "Val mIoU: {:.3f}..".format(val_iou_score / len(val_loader)),
              "Train Acc:{:.3f}..".format(accuracy_score / len(train_loader)),
              "Val Acc:{:.3f}..".format(test_accuracy / len(val_loader)),
              "Time: {:.2f}m".format((time.time() - since) / 60))

        epoch += 1

    print('Total time: {:.2f} m'.format((time.time() - fit_time) / 60))

    return history, best_model, min_loss


# 3. Run exploratory training to find the best encoder
# ----------------------------------------------------
encoder_name = ['mobilenet_v2', 'resnet50', 'efficientnet-b0', 'densenet121']


def encoder_comparison(encoder_name, n_epochs):
    Losses = []

    for i in range(len(encoder_name)):
        best_loss = np.inf
        model = smp.Unet(encoder_name[i],
                         encoder_weights='imagenet',
                         classes=23,
                         activation=None,
                         encoder_depth=5,
                         decoder_channels=[256, 128, 64, 32, 16])

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           factor=0.75,
                                                           patience=1,
                                                           cooldown=1,
                                                           min_lr=0)

        exp_sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=0.0001)

        print("Training : ", encoder_name[i])
        history, best_model, loss = Exploratory_Training(n_epochs, model, train_loader, val_loader, criterion,
                                                         optimizer, exp_sched)

        if loss < best_loss:
            best_loss = loss
            final_model = best_model
            best_encoder = name

        name = encoder_name[i]
        plot_results(history, name)
        Losses.append(history["test_losses"])
        torch.save(best_model, 'Results/' + name + '/' + name + '_best.pt')

    torch.save(final_model, 'Results/' + best_encoder + '_final.pt')

    return Losses


# Losses = encoder_comparison(encoder_name)
from Utility import plot_losses

# plot_losses(Losses, encoder_name)

best_encoder = 'efficientnet-b0'
model = smp.UnetPlusPlus(best_encoder,
                         encoder_weights='imagenet',
                         classes=23,
                         activation=None,
                         decoder_attention_type=None,
                         encoder_depth=5,
                         decoder_channels=[256, 128, 64, 32, 16])

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                   factor=0.65,
                                                   patience=2,
                                                   cooldown=1,
                                                   min_lr=0)

print("Training : ", best_encoder, " Unet++")
history, best_model, loss = Training_Function(model, train_loader, val_loader, criterion, optimizer, sched)
torch.save(best_model, 'Models/Unet++_' + best_encoder + '.pt')
plot_results(history, best_encoder)
