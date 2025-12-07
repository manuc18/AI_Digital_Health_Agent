import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "MedDialog"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# File paths
TRAIN_DATA_PATH = RAW_DATA_DIR / "english-train.json"
DEV_DATA_PATH = RAW_DATA_DIR / "english-dev.json"
TEST_DATA_PATH = RAW_DATA_DIR / "english-test.json"
CLEANED_DATA_PATH = PROCESSED_DATA_DIR / "meddialog_cleaned.csv"

# Model configurations
CLINICAL_BERT_MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"
MAX_SEQ_LENGTH = 128
BATCH_SIZE = 16
