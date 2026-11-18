"""Time a single bare Gemini call, to isolate whether DeepEval judge
slowness (test_judge.py, judge_provider: gemini) is the Gemini
API itself or something in DeepEval's own timeout/retry path.

Run from the repo root:
    python3 helper_scripts/llm_diagnostics/check_gemini_latency.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from langchain_google_genai import ChatGoogleGenerativeAI

from rag._09_common.config import load_config

PROMPT = (
    "Score the faithfulness of this answer to the context on a scale of 0 to 1. "
    "Context: The sky appears blue due to Rayleigh scattering of sunlight. "
    "Answer: The sky is blue because of Rayleigh scattering. "
    "Respond with just a number."
)

config = load_config()
model = config.evaluation.judge_model

print(f"=== {model} ===")
try:
    chat = ChatGoogleGenerativeAI(model=model, temperature=0)
    start = time.monotonic()
    response = chat.invoke(PROMPT)
    elapsed = time.monotonic() - start
    print(f"elapsed: {elapsed:.2f}s")
    print(f"response: {response.content!r}")
except Exception as exc:
    elapsed = time.monotonic() - start if "start" in dir() else None
    message = str(exc)
    if "RESOURCE_EXHAUSTED" in message or "prepayment credits are depleted" in message:
        print(
            f"FAILED after {elapsed}: Gemini prepayment credits are depleted.\n"
            "  Top up at https://ai.studio/projects to restore judge access."
        )
    else:
        print(f"FAILED after {elapsed}: {type(exc).__name__}: {exc}")
