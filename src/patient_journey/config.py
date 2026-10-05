"""
Configuration module for the GenAI Patient Journey Analytics pipeline.
Handles environment variables, file paths, model configurations, and concurrency parameters.
"""

from __future__ import annotations

import os
from pathlib import Path
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

# Base directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
TESTS_DIR = PROJECT_ROOT / "tests"

# Ensure runtime directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# File paths
TRANSCRIPTS_FILE = DATA_DIR / "transcripts.json"
GROUND_TRUTH_FILE = DATA_DIR / "ground_truth.json"
MANUAL_LABELS_FILE = DATA_DIR / "manual_labels.csv"

EXTRACTED_RECORDS_FILE = OUTPUTS_DIR / "extracted_records.json"
PATIENTS_FLAT_FILE = OUTPUTS_DIR / "patients_flat.csv"
VALIDATION_REPORT_FILE = OUTPUTS_DIR / "validation_report.md"
EXTRACTION_ERRORS_LOG = OUTPUTS_DIR / "extraction_errors.log"


@dataclass(frozen=True)
class PipelineSettings:
    """Settings for LLM calls, rate limiting, and reproducibility."""
    groq_api_key: str | None = os.getenv("GROQ_API_KEY")
    gemini_api_key: str | None = os.getenv("GEMINI_API_KEY")
    
    # Models: Groq LLaMA primary, Gemini free tier fallback
    primary_model: str = os.getenv("PRIMARY_MODEL", "groq/llama-3.3-70b-versatile")
    fallback_model: str = os.getenv("FALLBACK_MODEL", "gemini/gemini-1.5-flash")
    
    # Concurrency and rate limiting
    max_concurrent_requests: int = int(os.getenv("MAX_CONCURRENT_REQUESTS", "3"))
    max_retries: int = int(os.getenv("MAX_RETRIES", "4"))
    request_timeout: float = float(os.getenv("REQUEST_TIMEOUT", "60.0"))
    
    # Generation parameters
    random_seed: int = int(os.getenv("RANDOM_SEED", "42"))
    num_transcripts: int = 60
    temperature_generation: float = 0.75
    temperature_extraction: float = 0.0


settings = PipelineSettings()
