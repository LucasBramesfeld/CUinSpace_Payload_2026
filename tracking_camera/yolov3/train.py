import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from tqdm import tqdm
from utils import save_checkpoint, load_checkpoint
from dataset import Dataset
from model import YOLOv3, YOLOvR1, YOLOvR2
from loss import YOLOLoss
    
# Define the train function to train the model
def training_loop(loader, model, optimizer, loss_fn, scaler, scaled_anchors):
    # Creating a progress bar
    progress_bar = tqdm(loader, leave=True)

    # Initializing a list to store the losses
    losses = []

    # Iterating over the training data
    for _, (x, y) in enumerate(progress_bar):
        x = x.to(device)
        y0, y1, y2 = (
            y[0].to(device),
            y[1].to(device),
            y[2].to(device),
        )

        with torch.cuda.amp.autocast():
            # Getting the model predictions
            outputs = model(x)
            # Calculating the loss at each scale
            loss = (
                  loss_fn(outputs[0], y0, scaled_anchors[0])
                + loss_fn(outputs[1], y1, scaled_anchors[1])
                + loss_fn(outputs[2], y2, scaled_anchors[2])
            )

        # Add the loss to the list
        losses.append(loss.item())

        # Reset gradients
        optimizer.zero_grad()

        # Backpropagate the loss
        scaler.scale(loss).backward()

        # Optimization step
        scaler.step(optimizer)

        # Update the scaler for next iteration
        scaler.update()

        # update progress bar with loss
        mean_loss = sum(losses) / len(losses)
        progress_bar.set_postfix(loss=mean_loss)

if __name__ == '__main__':
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load and save model variable
    load_model = False
    save_model = True

    # model checkpoint file name
    checkpoint_file = "C:/Users/lucas_6hii5cu/Documents/datasets/tracking_camera/rocketv2.pth.tar"

    # Anchor boxes for each feature map scaled between 0 and 1
    # 3 feature maps at 3 different scales based on YOLOv3 paper
    ANCHORS = [
        [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
        [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
        [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],
    ]

    # Batch size for training
    batch_size = 4

    # Learning rate for training
    leanring_rate = 1e-5

    # Number of epochs for training
    epochs = 50

    # Image size
    image_size = 640

    # Grid cell sizes
    s = [image_size // 32, image_size // 16, image_size // 8]

    # Creating the model from YOLOv3 class
    model = YOLOvR2().to(device)

    # Defining the optimizer
    optimizer = optim.Adam(model.parameters(), lr = leanring_rate)

    if load_model:
        load_checkpoint(checkpoint_file, model, optimizer, leanring_rate, device=device)

    # Defining the loss function
    loss_fn = YOLOLoss()

    # Defining the scaler for mixed precision training
    scaler = torch.cuda.amp.GradScaler()

    # Creating a dataset object
    train_dataset = Dataset(
        csv_file="C:/Users/lucas_6hii5cu/Documents/datasets/tracking_camera/data/train.csv",
        image_dir="C:/Users/lucas_6hii5cu/Documents/datasets/tracking_camera/data/images/",
        label_dir="C:/Users/lucas_6hii5cu/Documents/datasets/tracking_camera/data/labels/",
        grid_sizes=[13, 26, 52],
        anchors=ANCHORS,
        transform=None
    )
    '''train_dataset = Dataset(
        csv_file="C:/Users/lucas/Downloads/test_yolo_data/train.csv",
        image_dir="C:/Users/lucas/Downloads/test_yolo_data/images/",
        label_dir="C:/Users/lucas/Downloads/test_yolo_data/labels/",
        grid_sizes=[13, 26, 52],
        anchors=ANCHORS,
        transform=None
    )'''

    # Defining the train data loader
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size = batch_size,
        num_workers = 8,
        shuffle = True,
        pin_memory = True,
    )

    # Scaling the anchors
    scaled_anchors = (
        torch.tensor(ANCHORS) * 
        torch.tensor(s).unsqueeze(1).unsqueeze(1).repeat(1,3,2)
    ).to(device)

    # Training the model
    for e in range(1, epochs+1):
        print("Epoch:", e)
        training_loop(train_loader, model, optimizer, loss_fn, scaler, scaled_anchors)

        # Saving the model
        if save_model:
            save_checkpoint(model, optimizer, filename=checkpoint_file)