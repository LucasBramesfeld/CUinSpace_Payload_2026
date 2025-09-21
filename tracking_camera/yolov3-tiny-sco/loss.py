import torch
import torch.nn as nn
from utils import iou 

class YOLOLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.sigmoid = nn.Sigmoid()

    def forward(self, pred, target, lambda_coord=2, lambda_noobj=1, lambda_iou=2):
        """
        pred: [batch, H, W, 5]  -> [objectness, x, y, w, h]
        target: same shape
        """
        # Masks for cells with and without objects
        obj = target[..., 0] == 1
        no_obj = target[..., 0] == 0

        # No-object loss
        no_object_loss = self.bce(pred[..., 0:1][no_obj], target[..., 0:1][no_obj])

        if obj.any():
            # Predicted boxes and target boxes for cells containing objects
            pred_boxes = pred[..., 1:5][obj]
            target_boxes = target[..., 1:5][obj]

            # IoU loss
            iou_score = iou(pred_boxes, target_boxes)
            iou_loss = (1 - iou_score).mean()

            # Optional MSE for centers and sizes
            center_loss = self.mse(pred_boxes[:, :2], target_boxes[:, :2])
            size_loss = self.mse(pred_boxes[:, 2:4], target_boxes[:, 2:4])
            box_loss = lambda_iou * iou_loss + center_loss + size_loss

            # Objectness loss
            object_loss = self.bce(pred[..., 0:1][obj], target[..., 0:1][obj])
        else:
            box_loss = torch.tensor(0.0, device=pred.device)
            object_loss = torch.tensor(0.0, device=pred.device)

        # Total loss
        loss = lambda_coord * box_loss + object_loss + lambda_noobj * no_object_loss
        return loss
