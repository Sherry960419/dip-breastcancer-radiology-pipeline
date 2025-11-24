from pathlib import Path
import numpy as np
import pydicom
import cv2

from config import OUTPUT_DIR

def run_step(context: dict) -> dict:
    """
    Read DICOM → normalize → save as standard 8-bit PNG (0–255).
    This ensures OpenCV can read it in Step2.
    """
    dicom_path = Path(context["dicom_path"])
    patient_id = context.get("patient_id", dicom_path.stem)

    ds = pydicom.dcmread(str(dicom_path))
    arr = ds.pixel_array.astype(float)

    # Normalize to 0–255
    arr -= arr.min()
    arr /= (arr.max() + 1e-8)
    arr_uint8 = (arr * 255).astype(np.uint8)

    # If grayscale, make 3-channel for SAM compatibility
    img_rgb = cv2.cvtColor(arr_uint8, cv2.COLOR_GRAY2RGB)

    png_path = OUTPUT_DIR / f"{patient_id}.png"
    cv2.imwrite(str(png_path), img_rgb)

    print(f"[Step1] Saved PNG: {png_path}")

    return {
        "png_path": str(png_path),
        "dicom_ds": ds,
    }
