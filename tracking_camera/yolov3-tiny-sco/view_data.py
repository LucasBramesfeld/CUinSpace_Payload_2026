import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchvision import transforms
from tqdm import tqdm
from utils import save_checkpoint, load_checkpoint, target_from_box, plot_image

if __name__ == '__main__':
    from dataset import Dataset  # Your Dataset class

    # Hyperparameters
    IMAGE_SIZE = 640
    BATCH_SIZE = 8
    LEARNING_RATE = 1e-4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Paths
    IMAGES_DIR = "C:/Users/lucas_6hii5cu/Documents/datasets/tracking_camera/data/images"
    LABELS_DIR = "C:/Users/lucas_6hii5cu/Documents/datasets/tracking_camera/data/labels"
    MODEL_DIR = "C:/Users/lucas_6hii5cu/Documents/datasets/tracking_camera/yolov2_test"

    # Transform
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])

    # Dataset and Dataloader
    dataset = Dataset(IMAGES_DIR, LABELS_DIR, image_size=IMAGE_SIZE, transform=transform)
    test_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    x, y = next(iter(test_loader))
    

    for i in range(BATCH_SIZE):
        plot_image(x[i].permute(1,2,0), y[i])