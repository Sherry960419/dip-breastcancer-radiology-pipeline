# DIP Breast Cancer Radiology AI Pipeline

Automated pipeline for breast cancer DICOM preprocessing, MedSAM segmentation, Dice/IoU evaluation, clinical data extraction, PubMed retrieval, and Gemini-based radiology report generation.

---

## 1. Project Overview

This project builds an end-to-end automated radiology workflow:

- DICOM → PNG preprocessing  
- MedSAM segmentation (with preprocessing-enhanced prompts)  
- Dice/IoU evaluation with visualization  
- Clinical information extraction from TCGA PanCanAtlas TSV  
- PubMed top-20 literature search  
- Gemini 2.5 Flash Vision radiology report generation  

The workflow is modular (Step0–Step7) and designed for easy debugging and extension.

---

## 2. Folder Structure

```
DIP Project/
│── config.py
│── step0_prepare_case.py
│── step1_dicom_to_png.py
│── step2_medsam_segmentation.py
│── step3_evaluation.py
│── step4_literature_search.py
│── step5_build_prompt.py
│── step6_gemini_summary.py
│── step7_run_agent.py
│
├── data/
│     ├── dicoms/
│     ├── lesions/
│     └── brca_tcga_pan_can_atlas_2018_clinical_data.tsv
│
└── outputs/
```

---

## 3. Installation

### 3.1 Create and activate Conda environment

```bash
conda create -n dip_env python=3.10
conda activate dip_env
```

### 3.2 Install dependencies

```bash
pip install pydicom opencv-python numpy matplotlib langchain-core biopython google-generativeai
```

### 3.3 MedSAM checkpoint

Download or place the MedSAM checkpoint here:

```
models/medsam_vit_b.pth
```

---

## 4. How to Run the Pipeline

### 4.1 Run entire workflow

```bash
python step7_run_agent.py
```

### 4.2 Test individual steps  
Useful for debugging segmentation or evaluation.

```bash
python step2_medsam_segmentation.py
python step3_evaluation.py
```

---

## 5. Clinical TSV Requirements

The TSV file must contain at least the columns:

- `Patient ID`  
- `Diagnosis Age`  
- `Subtype`  
- `American Joint Committee on Cancer Tumor Stage Code`  
- `Neoplasm Disease Lymph Node Stage American Joint Committee on Cancer Code`  
- `American Joint Committee on Cancer Metastasis Stage Code`

These fields are used to extract:

- Age  
- Subtype  
- Stage Code (combined T + N + M)

---

## 6. PubMed Search

The workflow retrieves the **top 20 most relevant papers** based on:

```
breast cancer + predicted tumor shape
```

### 6.1 Set PubMed email

Update in `config.py`:

```python
PUBMED_EMAIL = "your_email@example.com"
```

---

## 7. Output Files

The workflow generates:

- Preprocessed PNG  
- Segmentation mask  
- Overlay image  
- Dice/IoU comparison visualization  
- Final Gemini-generated radiology report

Reports are saved under:

```
outputs/{patient_id}_summary.txt
```

---

## 8. Features

- Modular multi-step design for easy debugging  
- Preprocessing-enhanced MedSAM segmentation  
- Automatic tumor shape classification  
- PubMed integration (BioPython Entrez)  
- Large language model (Gemini Vision) for final clinical report  
- Supports multi-patient batch processing  

---

## 9. Contact

For questions, please open an Issue or contact the project maintainer.

