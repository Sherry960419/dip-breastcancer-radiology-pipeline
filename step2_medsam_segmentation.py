from pathlib import Path
from config import OUTPUT_DIR, MEDSAM_CHECKPOINT
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
from segment_anything import sam_model_registry, SamPredictor


def preprocess_image_breast(bgr_img):
    """
    Basic preprocessing for breast MR:
    - convert to gray
    - normalize intensity to [0, 255]
    - apply CLAHE (local contrast enhancement)
    - light Gaussian blur
    Returns:
        pre_rgb: 3-channel RGB image for SAM
        pre_gray: single-channel preprocessed gray for seed selection
    """
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)

    # Normalize to full 0-255 range
    gray_norm = cv2.normalize(gray, None, alpha=0, beta=255,
                              norm_type=cv2.NORM_MINMAX)
    gray_norm = gray_norm.astype(np.uint8)

    # CLAHE to enhance local contrast around lesions
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_clahe = clahe.apply(gray_norm)

    # Light smoothing to reduce noise
    gray_blur = cv2.GaussianBlur(gray_clahe, (3, 3), 0)

    # SAM expects 3-channel RGB
    pre_rgb = cv2.cvtColor(gray_blur, cv2.COLOR_GRAY2RGB)

    return pre_rgb, gray_blur


def get_intensity_seed(gray_img, top_k_ratio=0.01):
    """
    Select high-intensity pixels as foreground seed points.
    Returns an (N, 2) array of (x, y) points.
    """
    flat = gray_img.flatten()
    n_pixels = len(flat)
    k = max(1, int(n_pixels * top_k_ratio))

    # indices of k brightest pixels
    top_idx = np.argpartition(flat, -k)[-k:]
    ys, xs = np.divmod(top_idx, gray_img.shape[1])
    pts = np.column_stack([xs, ys])
    return pts


def build_bbox_from_points(points, pad=25, shape=None):
    """
    Build a loose bounding box around the given points.
    shape: (H, W) of the image.
    """
    if points is None or len(points) == 0:
        return None

    xs, ys = points[:, 0], points[:, 1]
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    if shape is not None:
        H, W = shape
        x_min = max(0, x_min - pad)
        y_min = max(0, y_min - pad)
        x_max = min(W - 1, x_max + pad)
        y_max = min(H - 1, y_max + pad)
    else:
        x_min -= pad
        y_min -= pad
        x_max += pad
        y_max += pad

    return np.array([x_min, y_min, x_max, y_max], dtype=np.int32)


def postprocess_mask(mask):
    """
    Keep only the largest connected component as final lesion.
    """
    mask_bin = (mask > 0).astype(np.uint8)
    num_labels, labels = cv2.connectedComponents(mask_bin)

    if num_labels <= 1:
        return mask_bin

    max_area = 0
    best_id = 0
    for lbl in range(1, num_labels):
        area = np.sum(labels == lbl)
        if area > max_area:
            max_area = area
            best_id = lbl

    return (labels == best_id).astype(np.uint8)


def classify_shape_from_mask(mask):
    """
    Very simple shape classifier based on contour circularity.
    Returns "Round-Oval" or "Irregular" or "Unknown".
    """
    mask_bin = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return "Unknown"

    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    peri = cv2.arcLength(cnt, closed=True)
    if peri == 0:
        return "Unknown"

    circularity = 4.0 * np.pi * area / (peri ** 2 + 1e-8)
    if circularity > 0.75:
        return "Round-Oval"
    else:
        return "Irregular"


def run_step(context: dict) -> dict:
    """
    Enhanced MedSAM segmentation step:
      - DICOM was already converted to PNG in Step1
      - Here we:
          * preprocess image (CLAHE + blur)
          * run MedSAM with intensity-based seeds + bounding box
          * post-process mask (largest component)
          * classify tumor shape
          * save mask & overlay
    """
    png_path = Path(context["png_path"])
    patient_id = context.get("patient_id", png_path.stem)

    print("[Step2] Running MedSAM segmentation...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Step2] Using device: {device}")

    # Load MedSAM model checkpoint
    sam = sam_model_registry["vit_b"]()
    state = torch.load(str(MEDSAM_CHECKPOINT), map_location=device)
    sam.load_state_dict(state)
    sam.to(device)

    predictor = SamPredictor(sam)

    # --- Load original PNG image ---
    img_bgr = cv2.imread(str(png_path))
    if img_bgr is None:
        raise RuntimeError(f"[Step2] Failed to read PNG: {png_path}")
    img_rgb_orig = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # --- Preprocess image for SAM ---
    img_rgb_pre, img_gray_pre = preprocess_image_breast(img_bgr)
    H, W = img_gray_pre.shape

    # --- Intensity-based seed points ---
    seed_points = get_intensity_seed(img_gray_pre, top_k_ratio=0.01)
    if seed_points.shape[0] > 10:
        fg_points = seed_points[:10]
    else:
        fg_points = seed_points
    point_labels = np.ones(len(fg_points), dtype=np.int32)

    # --- Auto bounding box from seeds ---
    bbox = build_bbox_from_points(seed_points, pad=30, shape=(H, W))
    if bbox is None:
        # Fallback: center box
        cx, cy = W // 2, H // 2
        bbox = np.array([cx - 32, cy - 32, cx + 32, cy + 32], dtype=np.int32)

    # --- Run SAM on preprocessed image ---
    predictor.set_image(img_rgb_pre)
    masks, scores, _ = predictor.predict(
        box=bbox,
        point_coords=fg_points,
        point_labels=point_labels,
        multimask_output=False,
    )

    raw_mask = masks[0]
    mask = postprocess_mask(raw_mask)
    mask_score = float(scores[0])

    # --- Shape classification from mask ---
    shape = classify_shape_from_mask(mask)

    # --- Save mask as PNG ---
    mask_path = OUTPUT_DIR / f"{patient_id}_mask.png"
    plt.imsave(mask_path, mask, cmap="gray")

    # --- Save overlay on ORIGINAL RGB image ---
    overlay = img_rgb_orig.copy()
    overlay[mask > 0] = [255, 0, 0]
    overlay_path = OUTPUT_DIR / f"{patient_id}_overlay.png"
    plt.imsave(overlay_path, overlay)

    print(f"[Step2] Mask score = {mask_score:.4f}, shape = {shape}")
    print(f"[Step2] Saved mask: {mask_path}")
    print(f"[Step2] Saved overlay: {overlay_path}")

    # Update context for later steps
    context["mask_path"] = str(mask_path)
    context["overlay_path"] = str(overlay_path)
    context["shape"] = shape

    return {
        "pred_mask": mask,
        "mask_score": mask_score,
        "mask_path": str(mask_path),
        "overlay_path": str(overlay_path),
        "shape": shape,
    }
