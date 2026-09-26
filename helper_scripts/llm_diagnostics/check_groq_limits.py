"""Cheap Groq call that prints the rate-limit headers Groq returns on
every response (x-ratelimit-remaining-requests/-tokens etc.), to check
current usage/limits without waiting to actually get 429'd. Uses the
groq SDK directly (not LangChain) since LangChain doesn't surface
these headers.

Run from the repo root:
    python3 helper_scripts/llm_diagnostics/check_groq_limits.py
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from groq import Groq

from rag._09_common.config import load_config

config = load_config()
api_key_env = config.llm.groq.api_key_env
api_key = os.environ.get(api_key_env)
if not api_key:
    raise SystemExit(f"{api_key_env} not set in environment")

client = Groq(api_key=api_key)

response = client.chat.completions.with_raw_response.create(
    model=config.llm.groq.model,
    messages=[{"role": "user", "content": "reply with just: ok"}],
    max_tokens=5,
)

headers = response.headers
print("HTTP status:", response.status_code)
for key in sorted(headers.keys()):
    if key.lower().startswith("x-ratelimit") or key.lower() == "retry-after":
        print(f"{key}: {headers[key]}")

parsed = response.parse()
print("\nreply:", parsed.choices[0].message.content)
