import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchvision import transforms
from tqdm import tqdm
from utils import save_checkpoint, load_checkpoint, target_from_box, plot_image

if __name__ == '__main__':
    import os
    from dataset import Dataset  # Your Dataset class
    from model import YOLOv2      # Your model class
    from loss import YOLOLoss     # Your loss class

    # Hyperparameters
    IMAGE_SIZE = 640
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Paths
    IMAGES_DIR = "C:/Users/lucas_6hii5cu/Documents/datasets/tracking_camera/data/images"
    LABELS_DIR = "C:/Users/lucas_6hii5cu/Documents/datasets/tracking_camera/data/labels"
    MODEL_DIR = "C:/Users/lucas_6hii5cu/Documents/datasets/tracking_camera/RTv1"

    # Transform
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])

    # Dataset and Dataloader
    dataset = Dataset(IMAGES_DIR, LABELS_DIR, image_size=IMAGE_SIZE, transform=transform)
    test_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Model, Loss, Optimizer
    model = YOLOv2().to(DEVICE)
    loss_fn = YOLOLoss()
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    load_checkpoint(MODEL_DIR, model, optimizer, LEARNING_RATE, device=DEVICE)

    # Getting a sample image from the test data loader
    x, y = next(iter(test_loader))
    x = x.to(DEVICE)

    model.eval()
    with torch.no_grad():
        output = model(x).permute(0, 2, 3, 1)  # [B, H, W, C]
        bbox = []

        B, H, W, C = output.shape
        cell_size_x = 1.0 / W
        cell_size_y = 1.0 / H

        for result in output:
            # Find the cell with the highest objectness
            best_idx = torch.argmax(result[..., 0])
            row = best_idx // W
            col = best_idx % W

            # Select the best box
            box = result[row, col].clone()  # shape: [C] -> [objectness, x, y, w, h]

            # Convert to image-relative coordinates
            box_x = (col + box[1]) / W
            box_y = (row + box[2]) / H
            box_w = box[3]  # assume already normalized to [0,1] relative to image width
            box_h = box[4]  # assume already normalized to [0,1] relative to image height

            bbox.append(torch.tensor([box_x, box_y, box_w, box_h]))

        for i in range(BATCH_SIZE):
            plot_image(x[i].cpu().permute(1, 2, 0), bbox[i].cpu())