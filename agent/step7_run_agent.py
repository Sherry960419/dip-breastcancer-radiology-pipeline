"""
Main LangChain-based workflow runner for DIP project.

Pipeline:
0) step0_prepare_case      - infer patient_id & GT .les path
1) step1_dicom_to_png      - DICOM → PNG
2) step2_medsam_segmentation - MedSAM segmentation
3) step3_evaluation        - Dice / IoU with .les polygon
4) step4_literature_search - PubMed: 'breast cancer'
5) step5_build_prompt      - build LLM prompt
6) step6_llm_summary       - call Gemini Vision to generate report
"""

from pathlib import Path
import glob

from langchain_core.runnables import RunnableLambda, RunnableSequence

from config import DICOM_DIR
from step0_prepare_case import run_step as step0
from step1_dicom_to_png import run_step as step1
from step2_medsam_segmentation import run_step as step2
from step3_evaluation import run_step as step3
from step4_literature_search import run_step as step4
from step5_build_prompt import run_step as step5
from step6_llm_summary import run_step as step6


def _wrap_step(name, func):
    """Wrap step(context) into RunnableLambda."""
    def inner(context: dict):
        print(f"\n===== Running {name} =====")
        result = func(context)
        if result is None:
            return context
        context.update(result)
        return context
    return RunnableLambda(inner)


def build_chain() -> RunnableSequence:
    """Build the LangChain workflow (sequence of steps)."""
    chain = RunnableSequence(
        _wrap_step("Step0: prepare case (patient_id & GT)", step0),
        _wrap_step("Step1: DICOM → PNG", step1),
        _wrap_step("Step2: MedSAM segmentation", step2),
        _wrap_step("Step3: Dice/IoU evaluation", step3),
        _wrap_step("Step4: PubMed 'breast cancer' search", step4),
        _wrap_step("Step5: build LLM prompt", step5),
        _wrap_step("Step6: Gemini Vision summary", step6),
    )
    return chain


def run_for_one_dicom(dicom_path: Path):
    """Complete the entire workflow for a single DICOM file"""
    chain = build_chain()
    context = {"dicom_path": str(dicom_path)}
    final_context = chain.invoke(context)
    return final_context


def run_batch(test_mode: bool = True):
    """
    test_mode = True  → Only run the first DICOM (for debugging)
    test_mode = False → Run all DICOM files in the DICOM_DIR directory.
    """
    dicom_files = sorted(
        list(DICOM_DIR.glob("*.dcm")) + list(DICOM_DIR.glob("*.dicom"))
    )

    if not dicom_files:
        print(f"No DICOM files found in: {DICOM_DIR}")
        return

    if test_mode:
        dicom_files = dicom_files[:1]
        print("[Runner] TEST_MODE = True, will only run 1 case.")
    else:
        print(f"[Runner] TEST_MODE = False, will run {len(dicom_files)} cases.")

    for dcm in dicom_files:
        print("\n=======================================")
        print("Running pipeline for:", dcm.name)
        print("=======================================")
        ctx = run_for_one_dicom(dcm)
        print("\n>>> Summary saved at:", ctx.get("summary_path"))


if __name__ == "__main__":
    TEST_MODE = True
    run_batch(test_mode=TEST_MODE)
