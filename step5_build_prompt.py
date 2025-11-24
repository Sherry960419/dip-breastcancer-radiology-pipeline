from pathlib import Path
from datetime import datetime

def run_step(context: dict) -> dict:
    """
    Build the prompt for the LLM/Gemini.
    Includes:
      - Clinical info (patient_id, age, subtype, stage)
      - Image paths
      - Tumor shape
      - Literature (titles/journals/years)
    """

    patient_id = context.get("patient_id", "")
    age = context.get("age", "")
    subtype = context.get("subtype", "")
    stage = context.get("stage_code", "")

    png_path = context.get("png_path", "")
    mask_path = context.get("mask_path", "")
    shape = context.get("shape", "")
    papers = context.get("papers", [])

    # Format literature text
    lit_section = "\n".join(
        [f"- {p['title']} ({p['journal']}, {p['year']})" for p in papers]
    )

    date_str = datetime.now().strftime("%Y-%m-%d")
    
    prompt = f"""
You are a radiology AI assistant. Generate a structured, clinically useful report based on the provided information.

Basic Clinical Information:
- Patient ID: {patient_id}
- Age: {age}
- Subtype: {subtype}
- Stage Code: {stage}
- Date of Report: {date_str}

Image Information:
- Original PNG: {png_path}
- Segmentation Mask: {mask_path}
- Predicted Tumor Shape: {shape}

Instructions:
Using all the information above (image, mask, shape, clinical data, and literature titles), generate a structured radiology-style report including:

1. **Findings**
   - Detailed description of tumor appearance in the *original PNG image*
   - Clear explanation of what the *segmentation mask* highlights vs. misses
   - Use precise radiology language (location, margins, enhancement pattern, architectural distortion, etc.)

2. **Literature Context**
   - Synthesize insights from the 20 provided paper titles
   - Integrate findings with patient-specific factors (age, subtype, stage, tumor shape)
   - Avoid quoting PMIDs or listing papers individually—provide an integrated discussion.

3. **Suggested Next Clinical Steps**
   - Provide actionable, realistic clinical recommendations
   - Include biopsy, MDT consultation, receptor testing, genetic counseling, imaging follow-up, etc.
   - Tailor recommendations to subtype (Luminal A), patient age, and shape characteristics.

4. **Uncertainty / Limitations**
   - Discuss segmentation accuracy limitations
   - Single-slice imaging limitations
   - Potential diagnostic uncertainty or areas needing further evaluation

Write the report in paragraphs, concise and clinically oriented.
"""

    context["llm_prompt"] = prompt
    print("[Step5] LLM prompt constructed.")
    return {"llm_prompt": prompt}
