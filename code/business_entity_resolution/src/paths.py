"""Repository-relative paths shared by command-line pipeline scripts."""
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
DATASET_ROOT = REPO_ROOT / "student_resource" / "dataset"
MODEL_ROOT = REPO_ROOT / "models"
OUTPUT_ROOT = REPO_ROOT / "output"
