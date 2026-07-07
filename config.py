# -*- coding: utf-8 -*-
"""
SmartGrader Configuration

Loads API keys from environment variables and defines project-wide constants.
"""

import os
from dotenv import load_dotenv

# Load .env file if it exists
load_dotenv()

# ── API Keys ─────────────────────────────────────────────────────────────────
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# ── Roboflow Model Settings ─────────────────────────────────────────────────
ROBOFLOW_WORKSPACE = "ai-models-dexjl"
ROBOFLOW_PROJECT = "hand_print_frcnn-qeugu"
ROBOFLOW_MODEL_VERSION = 6

# ── Detection Defaults ───────────────────────────────────────────────────────
DEFAULT_CONFIDENCE = 47
DEFAULT_OVERLAP = 30

# ── OCR Settings ─────────────────────────────────────────────────────────────
EASYOCR_LANGUAGES = ["en"]
TROCR_MODEL_NAME = "microsoft/trocr-base-handwritten"

# ── LLM Settings ─────────────────────────────────────────────────────────────
LLM_MODEL_ID = "gpt-4-turbo"

# ── Grading Defaults ─────────────────────────────────────────────────────────
DEFAULT_MAX_SCORE = 10
LINE_THRESHOLD = 50  # px – max vertical gap to merge words into one line

# ── Ensure OpenAI key is available to the agno library ────────────────────────
if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
