import google.generativeai as genai
from config import GENAI_API_KEY, OUTPUT_DIR
from pathlib import Path

def run_step(context: dict) -> dict:
    """
    Use Gemini-2.5-Flash to generate a final radiology report.
    """

    if GENAI_API_KEY is None:
        raise RuntimeError("GOOGLE_API_KEY not found")

    genai.configure(api_key=GENAI_API_KEY)
    model = genai.GenerativeModel("gemini-2.5-flash")

    prompt = context["llm_prompt"]

    print("[Step6] Calling Gemini-2.5-Flash Vision API...")

    response = model.generate_content(prompt)
    summary = response.text

    out_path = OUTPUT_DIR / f"{context['patient_id']}_summary.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(summary)

    print(f"[Step6] Summary saved to: {out_path}")

    return {"summary_path": str(out_path)}
