import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Defining a function to calculate Intersection over Union (IoU)
def iou(box1, box2, is_pred=True):
    if is_pred:
        # IoU score for prediction and label
        # box1 (prediction) and box2 (label) are both in [x, y, width, height] format
        
        # Box coordinates of prediction
        b1_x1 = box1[..., 0:1] - box1[..., 2:3] / 2
        b1_y1 = box1[..., 1:2] - box1[..., 3:4] / 2
        b1_x2 = box1[..., 0:1] + box1[..., 2:3] / 2
        b1_y2 = box1[..., 1:2] + box1[..., 3:4] / 2

        # Box coordinates of ground truth
        b2_x1 = box2[..., 0:1] - box2[..., 2:3] / 2
        b2_y1 = box2[..., 1:2] - box2[..., 3:4] / 2
        b2_x2 = box2[..., 0:1] + box2[..., 2:3] / 2
        b2_y2 = box2[..., 1:2] + box2[..., 3:4] / 2

        # Get the coordinates of the intersection rectangle
        x1 = torch.max(b1_x1, b2_x1)
        y1 = torch.max(b1_y1, b2_y1)
        x2 = torch.min(b1_x2, b2_x2)
        y2 = torch.min(b1_y2, b2_y2)
        # Make sure the intersection is at least 0
        intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

        # Calculate the union area
        box1_area = abs((b1_x2 - b1_x1) * (b1_y2 - b1_y1))
        box2_area = abs((b2_x2 - b2_x1) * (b2_y2 - b2_y1))
        union = box1_area + box2_area - intersection

        # Calculate the IoU score
        epsilon = 1e-6
        iou_score = intersection / (union + epsilon)

        # Return IoU score
        return iou_score
    
    else:
        # IoU score based on width and height of bounding boxes
        
        # Calculate intersection area
        intersection_area = torch.min(box1[..., 0], box2[..., 0]) * \
                            torch.min(box1[..., 1], box2[..., 1])

        # Calculate union area
        box1_area = box1[..., 0] * box1[..., 1]
        box2_area = box2[..., 0] * box2[..., 1]
        union_area = box1_area + box2_area - intersection_area

        # Calculate IoU score
        iou_score = intersection_area / union_area

        # Return IoU score
        return iou_score

def target_from_box(box, S=20): # Converts box label to tensor for loss
    target = torch.zeros(S, S, 5)
    
    obj, x_c, y_c, w, h = box
    
    # Determine which cell this box falls into
    i = int(y_c * S)
    j = int(x_c * S)
    
    # Convert x, y to be relative to the cell
    x_cell = x_c * S - j
    y_cell = y_c * S - i
    
    target[i, j, 0] = 1# objectness
    target[i, j, 1:5] = torch.tensor([x_cell, y_cell, w, h])
    
    return target

# Function to plot images with bounding boxes and class labels
def plot_image(image, box, label="rocket", color=(0,1,0)):
    # Reading the image with OpenCV
    img = np.array(image)
    # Getting the height and width of the image
    h, w, _ = img.shape

    # Create figure and axes
    fig, ax = plt.subplots(1)

    # Add image to plot
    ax.imshow(img)

    # Get the upper left corner coordinates
    upper_left_x = box[0] - box[2] / 2
    upper_left_y = box[1] - box[3] / 2

    # Create a Rectangle patch with the bounding box
    rect = patches.Rectangle(
        (upper_left_x * w, upper_left_y * h),
        box[2] * w,
        box[3] * h,
        linewidth=1,
        edgecolor=color,
        facecolor="none",
    )
    
    # Add the patch to the Axes
    ax.add_patch(rect)
    
    # Add class name to the patch
    plt.text(
        upper_left_x * w,
        upper_left_y * h,
        s=label,
        color="white",
        verticalalignment="top",
        bbox={"color": color, "pad": 0},
    )

    # Display the plot
    plt.show()

# Function to save checkpoint
def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("==> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)

# Function to load checkpoint
def load_checkpoint(checkpoint_file, model, optimizer, lr, device='cpu'):
    print("==> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr