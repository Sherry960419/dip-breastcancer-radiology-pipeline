import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from config import OUTPUT_DIR

def load_les_polygon_mask(les_path: str, image_shape) -> np.ndarray:
    """Decode TCGA .les polygon → 0/1 GT mask."""
    data = np.fromfile(str(les_path), dtype=np.uint16)
    data = data[data != 0]  # remove paddings

    if len(data) % 2 != 0:
        data = data[:-1]

    pts = data.reshape((-1, 2)).astype(np.int32)

    h, w = image_shape
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 1)
    return mask

def compute_dice_iou(pred_mask: np.ndarray, gt_mask: np.ndarray):
    pred_bin = (pred_mask > 0).astype(np.uint8)
    gt_bin = (gt_mask > 0).astype(np.uint8)

    inter = np.sum(pred_bin * gt_bin)
    dice = 2 * inter / (np.sum(pred_bin) + np.sum(gt_bin) + 1e-8)

    union = np.sum((pred_bin + gt_bin) > 0)
    iou = inter / (union + 1e-8)
    return float(dice), float(iou)

def visualize_pred_vs_gt(img_gray, pred, gt, vis_path):
    """Overlay pred (RED) & GT (GREEN) contours on original image."""
    H, W = img_gray.shape
    canvas = np.stack([img_gray]*3, axis=-1)

    # compute contours
    pred_cont = (pred - cv2.erode(pred, None)) > 0
    gt_cont   = (gt   - cv2.erode(gt, None)) > 0

    # apply colors
    canvas[pred_cont] = [255, 0, 0]   # red
    canvas[gt_cont]   = [0, 255, 0]   # green

    plt.figure(figsize=(6,6))
    plt.imshow(canvas)
    plt.axis("off")
    plt.savefig(vis_path, dpi=200, bbox_inches="tight")
    plt.close()

def run_step(context: dict) -> dict:
    """
    Evaluate Dice / IoU and save visualization figure.
    """
    pred_mask = context.get("pred_mask")
    gt_mask_path = context.get("gt_mask_path")
    png_path = context.get("png_path")

    metrics = {}

    if pred_mask is None:
        print("[Step3] No pred_mask in context → skip.")
        return {"metrics": metrics}

    if gt_mask_path is None:
        print("[Step3] No GT .les file → skip DICE/IoU.")
        return {"metrics": metrics}

    print(f"[Step3] Evaluating Dice/IoU with GT: {gt_mask_path}")

    # load GT
    h, w = pred_mask.shape[:2]
    gt_mask = load_les_polygon_mask(gt_mask_path, (h, w))

    # compute Dice & IoU
    dice, iou = compute_dice_iou(pred_mask, gt_mask)
    metrics = {"dice": dice, "iou": iou}
    print(f"[Step3] Dice = {dice:.4f}, IoU = {iou:.4f}")

    # visualization
    if png_path:
        img = cv2.imread(png_path, cv2.IMREAD_GRAYSCALE)
        vis_path = OUTPUT_DIR / f"{Path(png_path).stem}_pred_vs_gt.png"
        visualize_pred_vs_gt(img, pred_mask, gt_mask, vis_path)
        print(f"[Step3] Saved visualization: {vis_path}")

        context["eval_vis_path"] = str(vis_path)

    return {"metrics": metrics, "eval_vis_path": context.get("eval_vis_path")}
