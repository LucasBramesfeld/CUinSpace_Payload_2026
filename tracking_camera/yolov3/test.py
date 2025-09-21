import torch
import torch.optim as optim


from dataset import Dataset
from model import YOLOv3, YOLOvR1
from utils import convert_cells_to_bboxes, nms, load_checkpoint, plot_image
from loss import YOLOLoss

if __name__ == '__main__':
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load and save model variable
    load_model = False
    save_model = True

    # model checkpoint file name
    checkpoint_file = "C:/Users/lucas_6hii5cu/Documents/datasets/tracking_camera/rocketv1.pth.tar"

    # Anchor boxes for each feature map scaled between 0 and 1
    # 3 feature maps at 3 different scales based on YOLOv3 paper
    ANCHORS = [
        [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
        [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
        [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],
    ]

    # Batch size for training
    batch_size = 8

    # Learning rate for training
    leanring_rate = 1e-5

    # Number of epochs for training
    epochs = 1

    # Image size
    image_size = 640

    # Grid cell sizes
    s = [image_size // 32, image_size // 16, image_size // 8]

    # Class labels
    class_labels = [
        "rocket", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
        "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
        "pottedplant", "sheep", "sofa", "train", "tvmonitor"
    ]


    # Taking a sample image and testing the model

    # Setting the load_model to True
    load_model = True

    # Defining the model, optimizer, loss function and scaler
    model = YOLOvR1().to(device)
    optimizer = optim.Adam(model.parameters(), lr = leanring_rate)
    loss_fn = YOLOLoss()
    scaler = torch.cuda.amp.GradScaler()

    # Loading the checkpoint
    if load_model:
        load_checkpoint(checkpoint_file, model, optimizer, leanring_rate, device=device)

    # Defining the test dataset and data loader
    test_dataset = Dataset(
        csv_file="C:/Users/lucas_6hii5cu/Documents/datasets/tracking_camera/data/train.csv",
        image_dir="C:/Users/lucas_6hii5cu/Documents/datasets/tracking_camera/data/images/",
        label_dir="C:/Users/lucas_6hii5cu/Documents/datasets/tracking_camera/data/labels/",
        anchors=ANCHORS,
        transform=None
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size = 16,
        num_workers = 2,
        shuffle = True,
    )

    # Getting a sample image from the test data loader
    x, y = next(iter(test_loader))
    x = x.to(device)

    model.eval()
    with torch.no_grad():
        # Getting the model predictions
        output = model(x)
        # Getting the bounding boxes from the predictions
        bboxes = [[] for _ in range(x.shape[0])]
        anchors = (
                torch.tensor(ANCHORS)
                    * torch.tensor(s).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
                ).to(device)

        # Getting bounding boxes for each scale
        for i in range(3):
            batch_size, A, S, _, _ = output[i].shape
            anchor = anchors[i]
            boxes_scale_i = convert_cells_to_bboxes(
                                output[i], anchor, s=S, is_predictions=True
                            )
            for idx, (box) in enumerate(boxes_scale_i):
                bboxes[idx] += box
    model.train()

    # Plotting the image with bounding boxes for each image in the batch
    for i in range(batch_size):
        # Applying non-max suppression to remove overlapping bounding boxes
        nms_boxes = nms(bboxes[i], iou_threshold=0.5, threshold=0.6)
        # Plotting the image with bounding boxes
        plot_image(x[i].permute(1,2,0).detach().cpu(), nms_boxes, class_labels)