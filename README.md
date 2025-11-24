# Breast Cancer Radiology AI Agent Workflow (DIP Project)

This repository contains an end-to-end AI Agent workflow for **breast cancer radiology analysis**, integrating medical image preprocessing, MedSAM segmentation, Dice/IoU evaluation, TCGA clinical data extraction, PubMed literature retrieval, and Gemini Vision report generation.

The design follows a modular multi-step architecture inspired by modern AI agent systems (LangChain RunnableSequence), ensuring reproducibility, interpretability, and component-level debug capability.

---

## 1. Introduction

Accurate breast cancer assessment requires radiology–clinical–genomic integration.  
Traditional imaging workflows often lack automation, consistency, and structured reporting.

This project proposes a **DICOM-to-Report AI Agent**, consisting of:

- Automated DICOM preprocessing  
- Preprocessing-enhanced MedSAM segmentation  
- Dice/IoU quantitative evaluation  
- Clinical info extraction from TCGA PanCanAtlas  
- PubMed top-20 literature retrieval based on tumor morphology  
- Gemini 2.5 Flash Vision–powered radiology report  

The full workflow is suitable for:

- Research  
- Radiogenomics exploratory analysis  
- AI-assisted structured radiology reporting  
- Medical imaging coursework projects  

---

## 2. Project Objectives

The goals of this project include:

### 🔹 2.1 Image Processing & Segmentation
- Convert DICOM to PNG with intensity normalization  
- Preprocess PNG before segmentation  
- Apply MedSAM with:
  - intensity-based seed point selection  
  - adaptive bounding box  
  - largest-component postprocessing  
- Predict tumor shape (Round/Oval vs Irregular)

### 🔹 2.2 Evaluation & Visualization
- Decode TCGA `.les` polygon files  
- Compute Dice & IoU  
- Generate overlay comparison visualization  

### 🔹 2.3 Clinical Data Integration
Extract from TCGA TSV:
- Diagnosis Age  
- Subtype  
- TNM Stage Code  
- Validate patient–row matching  

### 🔹 2.4 Literature Retrieval
- Retrieve **Top 20 most relevant PubMed papers**  
- Extract titles, journals, years  
- Provide contextual understanding of tumor morphology  

### 🔹 2.5 Gemini Radiology Report Generation
Use Gemini Vision to produce:
- Findings based on PNG + segmentation  
- Integrated literature insights  
- Personalized clinical next steps  
- Uncertainty & limitations  

---

## 3. Dataset

This workflow relies on:

### **3.1 TCGA-BRCA Clinical Data**
```
brca_tcga_pan_can_atlas_2018_clinical_data.tsv
```
Containing:
- Patient ID  
- Diagnosis Age  
- Subtype  
- Tumor/Lymph Node/Metastasis stage codes  

### **3.2 DICOM Images**
Place under:
```
data/dicoms/
```

### **3.3 Ground Truth Lesions (.les)**
Polygon masks for Dice/IoU evaluation:
```
data/lesions/
```

### **3.4 MedSAM Model Checkpoint**
```
models/medsam_vit_b.pth
```
(Provided by the MedSAM authors—include citation below)

---

## 4. Methods

### **4.1 Image Preprocessing**
- DICOM → normalized PNG  
- Optional brightness / contrast adjustment  
- Grayscale conversion for intensity-based seed extraction  

### **4.2 MedSAM Segmentation (Enhanced)**
Segmentation uses:
- **Intensity-based seed points** (top 0.5% bright pixels)  
- **Auto bounding box**  
- **Largest connected component extraction**  
- **Tumor shape inference**  

### **4.3 Quantitative Evaluation**
- Load `.les` polygon  
- Fill into binary mask  
- Compute:
  - Dice  
  - IoU  
- Save comparison visualization  

### **4.4 PubMed Retrieval**
Query:
```
"breast cancer" + predicted tumor shape
```
Retrieve:
- Top 20 relevant papers  
- Titles / journals / years  

### **4.5 Gemini Radiology Report**
Final radiology-style report includes:
- Patient basic clinical info  
- Findings on PNG & mask  
- Literature synthesis  
- Clinically meaningful next steps  
- Limitations  

---

## 5. Repository Structure

```
DIP Project/
│
├── agent/
│   └── step7_run_agent.py               # Main orchestrator workflow
│
├── tools/
│   ├── step0_prepare_case.py            # Extract patient ID + clinical info
│   ├── step1_dicom_to_png.py            # DICOM → PNG
│   ├── step2_medsam_segmentation.py     # Enhanced MedSAM segmentation
│   ├── step3_evaluation.py              # Dice/IoU + visualization
│   ├── step4_literature_search.py       # PubMed retrieval
│   ├── step5_build_prompt.py            # Build Gemini prompt
│   └── step6_gemini_summary.py          # Call Gemini Vision API
│
├── config.py                            # Global paths + API keys
│
├── data/
│   ├── dicoms/
│   ├── lesions/
│   └── brca_tcga_pan_can_atlas_2018_clinical_data.tsv
│
└── outputs/                             # PNGs, masks, overlays, reports
```

---

## 6. Installation

### 6.1 Conda Environment

```bash
conda create -n dip_env python=3.10
conda activate dip_env
```

### 6.2 Install Dependencies

```bash
pip install pydicom opencv-python numpy matplotlib biopython langchain-core google-generativeai
```

---

## 7. How to Run

### **Run full workflow**
```bash
python agent/step7_run_agent.py
```

### **Run individual tools for debugging**
```bash
python tools/step2_medsam_segmentation.py
python tools/step3_evaluation.py
```

---

## 8. Output Files

Saved to `outputs/`:

- `{patient_id}.png` — preprocessed DICOM  
- `{patient_id}_mask.png` — MedSAM mask  
- `{patient_id}_overlay.png` — mask overlay  
- `{patient_id}_pred_vs_gt.png` — Dice/IoU comparison  
- `{patient_id}_summary.txt` — final radiology report  

---

## 9. Citations

Please cite the datasets and models used:

### **TCGA-BRCA Dataset**
> The Cancer Genome Atlas Program (TCGA), National Cancer Institute.

### **MedSAM**
> Ma, J., et al. "Segment Anything Model for Medical Images." (2023).

### **Gemini API**
> Google DeepMind, Gemini 2.5 Flash Vision.

---

## 10. Work-in-Progress Notice

This repository continues to evolve and may include experimental modules such as:

- multi-slice 3D inference  
- radiogenomics integration  
- AI agent–based error correction  
- improved PubMed ranking models  

---
