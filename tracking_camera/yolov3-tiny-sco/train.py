import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchvision import transforms
from tqdm import tqdm
from utils import save_checkpoint, load_checkpoint, target_from_box

# Define the train function to train the model
def training_loop(model, dataloader, optimizer, loss_fn, device, S=20):
    progress_bar = tqdm(dataloader, leave=True)
    losses = []

    for batch_idx, (images, boxes) in enumerate(progress_bar):
        images = images.to(device)
        targets = torch.zeros((images.size(0), S, S, 5)).to(device)

        # Create target tensor
        for i, box in enumerate(boxes):
            if box.numel() == 0:
                continue
            targets[i] = target_from_box(box, S=S).to(device)

        # Forward
        predictions = model(images).permute(0, 2, 3, 1) # [B, S, S, 5]
        loss, box_loss, object_loss, no_object_loss = loss_fn(predictions, targets)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        # update progress bar with loss
        mean_loss = sum(losses) / len(losses)
        #progress_bar.set_description(f"Loss: {mean_loss:.4f}")
        progress_bar.set_description(f"Loss: {(box_loss, object_loss, no_object_loss)}")

    avg_loss = sum(losses) / len(dataloader)
    return avg_loss

if __name__ == "__main__":
    import os
    from dataset import Dataset
    from model import YOLOv2
    from loss import YOLOLoss
   
    IMAGE_SIZE = 640
    BATCH_SIZE = 4
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 100
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    LOAD_MODEL = True
    SAVE_MODEL = True

    IMAGES_DIR = "C:/Users/lucas_6hii5cu/Documents/datasets/tracking_camera/data/images"
    LABELS_DIR = "C:/Users/lucas_6hii5cu/Documents/datasets/tracking_camera/data/labels"
    MODEL_DIR = "C:/Users/lucas_6hii5cu/Documents/datasets/tracking_camera/RTv3"

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])

    dataset = Dataset(IMAGES_DIR, LABELS_DIR, image_size=IMAGE_SIZE, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = YOLOv2().to(DEVICE)
    loss_fn = YOLOLoss()
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    if LOAD_MODEL:
        load_checkpoint(MODEL_DIR, model, optimizer, LEARNING_RATE, device=DEVICE)

    # Training loop
    for epoch in range(NUM_EPOCHS):
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}]")
        avg_loss = training_loop(model, dataloader, optimizer, loss_fn, DEVICE)
        print(f"Average Loss: {avg_loss:.4f}")

        if SAVE_MODEL:
            save_checkpoint(model, optimizer, filename=MODEL_DIR)

    print("Training completed!")