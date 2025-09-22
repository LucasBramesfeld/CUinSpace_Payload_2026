import cv2
import torch
from torchvision import transforms
from model import YOLOv2
from loss import YOLOLoss
from utils import load_checkpoint, plot_image
from PIL import Image

IMAGE_SIZE = 640
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = YOLOv2().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters())
MODEL_DIR = "C:/Users/lucas_6hii5cu/Documents/datasets/tracking_camera/RTv3"
load_checkpoint(MODEL_DIR, model, optimizer, LEARNING_RATE, device=DEVICE)

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])

video_path = "C:/Users/lucas_6hii5cu/Documents/datasets/tracking_camera/rocket_videos/IREC_2017_compilation.mp4"
cap = cv2.VideoCapture(video_path)

model.eval()
with torch.no_grad():
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess frame
        frame_resized = cv2.resize(frame, (IMAGE_SIZE, IMAGE_SIZE))
        frame_pil = Image.fromarray(cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB))  
        img_tensor = transform(frame_pil).unsqueeze(0).to(DEVICE)

        output = model(img_tensor).permute(0, 2, 3, 1)  # [B, H, W, C]

        B, H, W, C = output.shape
        cell_size_x = 1.0 / W
        cell_size_y = 1.0 / H

        bbox = []
        for result in output:
            best_idx = torch.argmax(result[..., 0])  # highest confidence
            row = best_idx // W
            col = best_idx % W

            box = result[row, col].clone()
            box_x = (col + box[1]) / W
            box_y = (row + box[2]) / H
            box_w = box[3]
            box_h = box[4]
            bbox.append([box_x, box_y, box_w, box_h])

        H, W, _ = frame.shape

        # Draw bounding box
        for (bx, by, bw, bh) in bbox:
            x1 = int((bx - bw / 2) * W)
            y1 = int((by - bh / 2) * H)
            x2 = int((bx + bw / 2) * W)
            y2 = int((by + bh / 2) * H)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

        cv2.imshow("YOLOv3-tiny - RTv3", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
