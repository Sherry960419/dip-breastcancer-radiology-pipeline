from pathlib import Path
import pandas as pd
from config import LESION_DIR, DATA_DIR

def run_step(context: dict) -> dict:
    """
    Step 0: Prepare case by:
    1) extracting patient ID from DICOM filename
    2) finding GT .les mask (if exists)
    3) loading clinical info (age, subtype, stage code)
    """

    dicom_path = Path(context["dicom_path"]).resolve()
    patient_id = dicom_path.stem

    # ---------- GT lesion ----------
    gt_path = LESION_DIR / f"{patient_id}.les"
    if gt_path.exists():
        gt_mask_path = str(gt_path)
        print(f"[Step0] Found GT lesion file: {gt_mask_path}")
    else:
        gt_mask_path = None
        print(f"[Step0] No GT lesion (.les) found for {patient_id}.")

    # ---------- Load TSV ----------
    tsv_path = DATA_DIR / "brca_tcga_pan_can_atlas_2018_clinical_data.tsv"

    df = pd.read_csv(tsv_path, sep="\t", low_memory=False)
    print(f"[Step0] Clinical TSV loaded ({len(df)} rows).")

    # Find row for this patient
    row = df[df["Patient ID"] == patient_id]

    if row.empty:
        print(f"[Step0] WARNING: Patient {patient_id} not found")
        age = None
        subtype = None
        stage_code = None
    else:
        row = row.iloc[0]

        age = row["Diagnosis Age"]
        subtype = row["Subtype"]

        # Combine 3 columns to form final stage code
        t_stage = str(row["American Joint Committee on Cancer Tumor Stage Code"])
        n_stage = str(row["Neoplasm Disease Lymph Node Stage American Joint Committee on Cancer Code"])
        m_stage = str(row["American Joint Committee on Cancer Metastasis Stage Code"])

        # correct format e.g. T1C_N0_M0
        stage_code = f"{t_stage}_{n_stage}_{m_stage}"

        print(f"[Step0] Clinical info — Age: {age}, Subtype: {subtype}, Stage: {stage_code}")

    return {
        "dicom_path": str(dicom_path),
        "patient_id": patient_id,
        "gt_mask_path": gt_mask_path,
        "age": age,
        "subtype": subtype,
        "stage_code": stage_code
    }
