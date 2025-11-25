# **Radiology AI Agent for Breast Cancer MRI: MedSAM Segmentation + TCGA Clinical Integration + LLM Reporting**

This repository contains an end-to-end AI Agent workflow for **breast cancer radiology analysis**, integrating medical image preprocessing, MedSAM segmentation, Dice/IoU evaluation, TCGA clinical data extraction, PubMed literature retrieval, and Gemini Vision report generation.

- automatic DICOM preprocessing  
- enhanced MedSAM segmentation  
- Dice/IoU evaluation using TCGA Radiogenomics lesion masks  
- clinical metadata extraction from TCGA PanCancer Atlas (via cBioPortal)  
- PubMed top-20 literature retrieval  
- Gemini Vision–powered structured radiology report 

The design follows a modular multi-step architecture inspired by modern AI agent systems (LangChain RunnableSequence), ensuring reproducibility, interpretability, and component-level debug capability.

This project was developed for the **Digital Image Processing (DIP)** course and is extendable to future research in radiogenomics and clinical AI.

---

# **1. Introduction**

Accurate breast cancer imaging analysis requires consistent segmentation, structured reporting, and meaningful integration of clinical and literature knowledge.

This workflow proposes a **DICOM → PNG → Segmentation → Evaluation → Clinical Data → PubMed → LLM Report** pipeline designed for:

- Research  
- Radiogenomics exploratory analysis  
- AI-assisted structured radiology reporting  
- Medical imaging coursework projects  

The pipeline supports single-case or batch inference and includes a **test mode** to accelerate debugging.

---

# **2. Project Objectives**

###  2.1 Medical Image Processing & Segmentation
- Convert DICOM → normalized PNG  
- Preprocessing (denoise, contrast enhancement optional)  
- MedSAM segmentation with:
  - intensity-based foreground seed extraction  
  - adaptive bounding box  
  - largest connected component post-processing  
- Tumor shape prediction (Round/Oval vs Irregular)

###  2.2 Quantitative Evaluation
- Decode TCGA `.les` polygon annotations  
- Compute Dice and IoU  
- Save GT vs prediction visual overlays

###  2.3 Clinical Data Integration
Extract from TCGA TSV:
- Diagnosis Age  
- Subtype  
- TNM Stage Code  
- Validate patient–row matching  

Automatic matching via DICOM Patient ID.

###  2.4 PubMed Literature Retrieval
- Use: `"breast cancer" + predicted tumor shape`  
- Retrieve top-20 **most relevant** (not latest) papers  
- Extract: title, journal, year for LLM summarization

###  2.5 Radiology Report Generation (LLM)
Gemini Vision 2.5 Flash produces:
- tumor findings  
- segmentation interpretation  
- literature-integrated clinical context  
- suggested next steps  
- uncertainty & limitations  

---

# **3. Data Sources**

This project does **NOT** use the full TCGA-BRCA imaging dataset.  
Instead, **only the first 10–20 patients from the TCGA Breast Radiogenomics annotation file** are selected, and their DICOMs are downloaded using:

- **SERIES_UID**
- **IMAGE_UID**

to locate the corresponding MRI slice in the TCIA DICOM repository.

---

## **3.1 TCGA Imaging Data (DICOM)**  
Breast MRI data accessed using `SERIES_UID` + `IMAGE_UID`:

**The Cancer Imaging Archive (TCIA)**  
**TCGA-BRCA Collection**  
🔗 https://www.cancerimagingarchive.net/collection/tcga-brca/

---

## **3.2 Lesion Ground Truth (.les)**  
Polygon-based radiologist annotations used for Dice/IoU:

**TCGA Breast Radiogenomics – Analysis Results @ TCIA**  
🔗 https://www.cancerimagingarchive.net/analysis-result/tcga-breast-radiogenomics/

---

## **3.3 Clinical Data (TSV)**  
Clinical metadata loaded from:

**cBioPortal – Breast Invasive Carcinoma (TCGA, PanCancer Atlas)**  
🔗 https://www.cbioportal.org/study/summary?id=brca_tcga_pan_can_atlas_2018

Fields used:
- Diagnosis Age  
- Subtype  
- Tumor Stage Code  
- Lymph Node Stage Code  
- Metastasis Stage Code  

---

## **Required Citations**

If you use this repository, please cite:

### **TCGA Program**
The Cancer Genome Atlas (TCGA) Research Network  
🔗 https://www.cancer.gov/tcga  

### **TCIA**
Clark et al., *The Cancer Imaging Archive (TCIA): Maintaining and Operating a Public Information Repository*, J Digit Imaging (2013).  
🔗 https://doi.org/10.1007/s10278-013-9622-7  

### **cBioPortal**
Cerami et al., 2012; Gao et al., 2013  
🔗 https://www.cbioportal.org/cite  

---

# **4. Methods**

### **4.1 DICOM Preprocessing**
- DICOM → normalized PNG  
- Optional brightness / contrast adjustment  
- Grayscale conversion for intensity-based seed extraction  


### **4.2 Enhanced MedSAM Segmentation**
Segmentation uses:
- **Intensity-based seed points** (top 0.5% bright pixels)  
- **Auto bounding box**  
- **Largest connected component extraction**  
- **Tumor shape classification**  


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

### **Run the whole agent pipeline**
```bash
python agent/step7_run_agent.py
```
### **Run in test mode(only first case)**
in step7_run_agent_py:
```bash
TEST_MODE = TRUE
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

Please cite the models used:

### **MedSAM**
> Ma, J., et al. "Segment Anything Model for Medical Images." (2023).

### **Gemini API**
> Google DeepMind, Gemini 2.5 Flash Vision.

---

## 10. Work-in-Progress Notice

This repository continues to evolve and may include experimental modules such as:

- refined segmentation heuristics
- multi-slice 3D inference  
- radiogenomics integration  
- AI agent–based error correction  
- improved PubMed ranking  

---
