from pathlib import Path
import os

# Base directory for the DIP project
BASE_DIR = Path(__file__).resolve().parent

# Data directories
DATA_DIR = BASE_DIR / "data"
DICOM_DIR = DATA_DIR / "dicoms"
LESION_DIR = DATA_DIR / "lesions"

# Outputs and model directories
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_DIR = BASE_DIR / "models"

for d in [DATA_DIR, DICOM_DIR, LESION_DIR, OUTPUT_DIR, MODEL_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# MedSAM checkpoint
MEDSAM_CHECKPOINT = MODEL_DIR / "medsam_vit_b.pth"

# Clinical TSV from TCGA (pan-can atlas)
CLINICAL_TSV = DATA_DIR / "brca_tcga_pan_can_atlas_2018_clinical_data.tsv"

# PubMed email (required by NCBI Entrez)
PUBMED_EMAIL = "your_email@domain.com"

# Gemini API Key (must be set in environment)
GENAI_API_KEY = os.environ.get("GOOGLE_API_KEY")

if __name__ == "__main__":
    print("BASE_DIR    :", BASE_DIR)
    print("DICOM_DIR   :", DICOM_DIR)
    print("LESION_DIR  :", LESION_DIR)
    print("OUTPUT_DIR  :", OUTPUT_DIR)
    print("MODEL_DIR   :", MODEL_DIR)
    print("MEDSAM_CHECKPOINT :", MEDSAM_CHECKPOINT)
    print("CLINICAL_TSV:", CLINICAL_TSV)
    print("GENAI_API_KEY is None? ", GENAI_API_KEY is None)
