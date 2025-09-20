import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# import your model + dataset
from dataset import BoundingBoxDataset  # your dataset file
from bounding_box_CNN import BoundingBoxCNN        # your CNN file
from label_image_veiwer import compare_bounding_box

def iou_loss(preds, targets, eps=1e-6):
    """
    preds and targets: [batch_size, 4] in (x_min, y_min, x_max, y_max) format
    Returns: 1 - IoU for each box
    """
    # Intersection coordinates
    x1 = torch.max(preds[:, 0], targets[:, 0])
    y1 = torch.max(preds[:, 1], targets[:, 1])
    x2 = torch.min(preds[:, 2], targets[:, 2])
    y2 = torch.min(preds[:, 3], targets[:, 3])

    # Intersection area
    inter_area = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    # Union area
    pred_area = (preds[:, 2] - preds[:, 0]) * (preds[:, 3] - preds[:, 1])
    target_area = (targets[:, 2] - targets[:, 0]) * (targets[:, 3] - targets[:, 1])
    union_area = pred_area + target_area - inter_area + eps

    # IoU
    iou = inter_area / union_area

    # IoU loss
    return -torch.log(iou + eps)

def train(model, dataloader, epochs=10, lr=1e-3, device="cuda",lambda_iou=1):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.SmoothL1Loss()  # good for bounding box regression

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        # Wrap dataloader with tqdm
        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)

        for imgs, targets in loop:
            imgs, targets = imgs.to(device), targets.to(device)

            # forward
            outputs = model(imgs)

            # loss
            # Regression loss
            loss_reg = criterion(outputs, targets)
            
            # IoU loss
            #loss_iou = iou_loss(outputs, targets).mean()  # mean over batch

            # Combined loss
            loss = loss_reg #+ lambda_iou * loss_iou

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # update tqdm description with current loss
            loop.set_postfix(loss=running_loss / (loop.n + 1))

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(dataloader):.4f}")

    return model


if __name__ == "__main__":
    # dataset + dataloader
    dataset = BoundingBoxDataset("dataset/images", target_size=512)  # resize/pad to 128x128
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # model
    model = BoundingBoxCNN(in_channels=3)
    #model = SimpleYOLO()

    # train
    trained_model = train(model, dataloader, epochs=1, lr=1e-3, device="cuda" if torch.cuda.is_available() else "cpu")

    # test prediction
    while True:
        imgs, targets = next(iter(dataloader))
        preds = model(imgs)  # normalized outputs
        
        # ---- Convert first image for OpenCV ----
        img = imgs[0].permute(1, 2, 0).numpy()   # [H,W,C], still RGB
        img = (img * 255).astype("uint8")        # scale back to 0-255
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # OpenCV expects BGR

        # ---- Draw predicted box ----
        compare_bounding_box(img.copy(), targets[0].tolist(), preds[0].tolist())  
