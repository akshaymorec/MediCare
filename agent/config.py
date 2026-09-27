from pathlib import Path
import os
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "patient_data.csv"

load_dotenv(ROOT_DIR / ".env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
GUARDRAIL_MODEL = os.getenv("GUARDRAIL_MODEL", OPENAI_MODEL)


def validate_config() -> None:
    if not OPENAI_API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY is missing. Add it to the project .env file."
        )
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Patient dataset not found: {DATA_PATH}")