# Breast Cancer Radiology AI Agent Workflow (DIP Project)

This project implements an end-to-end AI-assisted workflow for breast MRI analysis using
Segment Anything (MedSAM), PubMed literature retrieval, and a Gemini-based LLM report generator.

The workflow is designed as a **modular, LangChain-powered pipeline** and was developed as part of a
Digital Image Processing (DIP) course project.

---

## 1. Overall Workflow

For each breast MRI DICOM file, the pipeline performs:

1. **Case initialization (Step 0)**
   - Infer `patient_id` from the DICOM filename.
   - Look up the corresponding ground-truth lesion file (`.les`) from the TCGA Breast Radiogenomics dataset.
   - Join with the clinical TSV (TCGA BRCA PanCancer Atlas) to extract:
     - Age (`Diagnosis Age`)
     - Molecular subtype (`Subtype`)
     - TNM-based stage code (combined from three columns):
       - `American Joint Committee on Cancer Tumor Stage Code`
       - `Neoplasm Disease Lymph Node Stage American Joint Committee on Cancer Code`
       - `American Joint Committee on Cancer Metastasis Stage Code`

2. **DICOM → PNG conversion (Step 1)**
   - Read the DICOM using `pydicom`.
   - Normalize the pixel intensities to `[0, 1]`.
   - Save as a 2D PNG slice for visualization and downstream processing.

3. **Preprocessing + MedSAM segmentation (Step 2)**
   - Apply basic preprocessing on the PNG (in BGR space):
     - Convert to grayscale.
     - Intensity normalization to `[0, 255]`.
     - CLAHE (local contrast enhancement).
     - Light Gaussian smoothing.
   - Convert the preprocessed image back to RGB and feed it into MedSAM.
   - Use **intensity-based seed points** (brightest pixels) and an **auto bounding box**
     as prompts for SAM.
   - Post-process the predicted mask by keeping only the **largest connected component**.
   - Perform a simple shape classification based on contour circularity:
     - `Round-Oval` vs `Irregular` vs `Unknown`.
   - Save:
     - Binary mask PNG (`*_mask.png`)
     - Overlay PNG (`*_overlay.png`) with the mask drawn on top of the original image.

4. **Dice / IoU evaluation (Step 3)**
   - If a matching `.les` file is available:
     - Decode the polygon coordinates into a ground-truth (GT) mask.
     - Compute **Dice** and **IoU** between predicted mask and GT.
     - Save a visualization that overlays:
       - Prediction (e.g., red)
       - Ground truth (e.g., green)
       - Overlap region
   - If no `.les` exists, the evaluation step is skipped for that case.

5. **PubMed literature retrieval (Step 4)**
   - Build a PubMed query using:
     - `"breast cancer"` + the predicted tumor shape (e.g., `"Irregular"`).
   - Use NCBI Entrez:
     - `esearch` with `retmax=20` and `sort="relevance"` to get up to 20 relevant PMIDs.
     - `efetch` in XML mode to extract:
       - Article title
       - Journal name
       - Publication year (or MedlineDate).
   - Store the resulting papers as a list of dictionaries:
     ```python
     {
       "title": "...",
       "journal": "...",
       "year": "..."
     }
     ```

6. **LLM prompt construction (Step 5)**
   - Build a structured prompt that includes:
     - Patient basic information:
       - Patient ID
       - Age
       - Subtype
       - Stage code
     - A short description of:
       - Original PNG path
       - Segmentation mask path
       - Predicted tumor shape
     - Summary of retrieved literature (titles / journals / years).
     - Clear instructions to the LLM to produce a radiology report including:
       - Findings (based on both the original image and the mask)
       - Quantified metrics (e.g., shape category, qualitative size/extent)
       - Suggested **next clinical steps** (treatment-related)
       - Literature context
       - Uncertainty / limitations

7. **Gemini-based radiology report (Step 6)**
   - Use `google-generativeai` and a Gemini model (e.g., `gemini-2.5-flash`) in **Vision mode**:
     - Provide:
       - The original PNG image
       - The segmentation overlay image
       - The structured textual prompt from Step 5
   - Generate a final, human-readable radiology-style report.
   - Save the report to a `.txt` file under `outputs/`.

8. **LangChain orchestration (Step 7)**
   - `step7_run_agent.py` uses `RunnableSequence` and `RunnableLambda` from `langchain-core`
     to connect all the above steps into a single **agent-like workflow**:
     ```text
     Step0 → Step1 → Step2 → Step3 → Step4 → Step5 → Step6
     ```
   - `TEST_MODE` can be used to:
     - Run only one DICOM case for debugging.
     - Later, switch to batch mode over a folder of DICOM files.

---

## 2. Project Structure

A simplified project layout:

```text
DIP Project/
├── config.py
├── step0_prepare_case.py
├── step1_dicom_to_png.py
├── step2_medsam_segmentation.py
├── step3_evaluation.py
├── step4_literature_search.py
├── step5_build_prompt.py
├── step6_gemini_summary.py
├── step7_run_agent.py
├── data/
│   ├── dicoms/             # input DICOMs (not included in repo)
│   ├── lesions/            # *.les ground-truth files (not included in repo)
│   └── brca_tcga_pan_can_atlas_2018_clinical_data.tsv
├── models/
│   └── medsam_vit_b.pth    # MedSAM checkpoint (not included in repo)
├── outputs/
│   ├── *.png               # generated PNG, masks, overlays
│   └── *_summary.txt       # LLM reports
└── README.md

Note: DICOMs, lesion files, and large model checkpoints are not committed to the repository
due to privacy and size constraints.

## 3. How to Run
	1.	Create and activate a Conda environment (example)
      conda create -n dip_env python=3.10
      conda activate dip_env

	2.	Install required packages

      You can install the core dependencies with:
      pip install pydicom opencv-python matplotlib biopython google-generativeai langchain-core
      # plus PyTorch and Segment Anything according to your OS / CUDA setup

  3.	Set the Gemini API key

      In your shell (e.g., ~/.zshrc):
      export GOOGLE_API_KEY="YOUR_API_KEY_HERE"
      Then restart the terminal or source ~/.zshrc.

  4.	Prepare data

	    •	Put your DICOM files under data/dicoms/.
	    •	Put your .les files under data/lesions/.
	    •	Put the clinical TSV file under data/ and make sure its name matches the one in config.py.

	5.	Run the full pipeline

     From the project root:
     python step7_run_agent.py

     •	In TEST_MODE, the script will only run on the first DICOM file (for debugging).
     •	In batch mode, it will iterate over all DICOMs in data/dicoms/.

     After completion, the generated reports will be saved in outputs/.



⸻

## 4. Evaluation and Limitations
	•	The current Dice/IoU scores can be quite low, indicating that:
	•	The automatic MedSAM prompts (intensity seeds + box) are still imperfect.
	•	Only a single 2D slice is used, not full 3D or multi-sequence MRI.
	•	The project is not training MedSAM; instead, it reuses a fixed checkpoint and focuses on:
	•	Prompt engineering
	•	Preprocessing
	•	Post-processing and evaluation
	•	Literature retrieval is based on:
	•	A simple PubMed query: "breast cancer" + predicted shape
	•	Top 20 relevant papers by PubMed’s ranking.
	•	Future improvements could include:
	•	More sophisticated preprocessing and multi-slice fusion.
	•	Better shape descriptors and radiomics features.
	•	More targeted literature filters (e.g., MRI-specific, subtype-specific).
	•	Comparing reports with vs. without the full pipeline to quantify the added value of
segmentation + clinical + literature context.



---

## 5. Acknowledgements

This project uses:
	•	TCGA BRCA datasets (radiogenomics and clinical).
	•	MedSAM / Segment Anything for lesion segmentation.
	•	LangChain for workflow orchestration.
	•	Google Gemini for LLM-based report generation.

It was developed as part of a graduate-level Digital Image Processing course.
