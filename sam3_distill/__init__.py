"""
SAM3 Detection Distillation Framework

A framework for creating object detection datasets using SAM3 auto-labeling,
then training YOLO and RT-DETR models on the generated labels.

Usage:
    python scripts/run.py --config configs/your_project.yaml
"""

__version__ = "0.1.0"
__author__ = "SAM3 Distillation Team"

from .config import Config, load_config
from .scraper import ImageScraper
from .labeler import SAM3Labeler, Detection
from .exporter import DatasetExporter
from .trainer import ModelTrainer
from .evaluator import ModelEvaluator
from .pipeline import Pipeline

__all__ = [
    "Config",
    "load_config",
    "ImageScraper",
    "SAM3Labeler",
    "Detection",
    "DatasetExporter",
    "ModelTrainer",
    "ModelEvaluator",
    "Pipeline",
]
