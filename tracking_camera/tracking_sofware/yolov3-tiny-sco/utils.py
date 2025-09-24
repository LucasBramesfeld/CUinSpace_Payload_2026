import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def target_from_box(box, S=20): # Converts box label to tensor for loss
    target = torch.zeros(S, S, 5)
    
    x_c, y_c, w, h = box
    
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