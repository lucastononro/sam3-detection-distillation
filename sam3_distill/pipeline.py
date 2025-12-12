"""
Pipeline orchestrator for the SAM3 Detection Distillation framework.
"""

import json
from datetime import datetime
from pathlib import Path

from .config import Config, load_config, validate_config
from .scraper import ImageScraper
from .labeler import SAM3Labeler, Detection
from .exporter import DatasetExporter
from .trainer import ModelTrainer
from .evaluator import ModelEvaluator


class Pipeline:
    """
    Orchestrates the full SAM3 Detection Distillation pipeline.

    Stages:
        1. Scrape - Download images from Bing
        2. Label - Auto-label with SAM3
        3. Export - Convert to YOLO/COCO format
        4. Train - Train YOLO and RT-DETR models
        5. Evaluate - Evaluate and compare models
    """

    def __init__(self, config_path: str | Path):
        """
        Initialize the pipeline.

        Args:
            config_path: Path to the YAML configuration file.
        """
        self.config_path = Path(config_path)
        self.config = load_config(config_path)

        # Validate config
        warnings = validate_config(self.config)
        if warnings:
            print("Configuration warnings:")
            for w in warnings:
                print(f"  - {w}")

        # Initialize components
        self.scraper = ImageScraper(self.config)
        self.labeler = SAM3Labeler(self.config)
        self.exporter = DatasetExporter(self.config)
        self.trainer = ModelTrainer(self.config)
        self.evaluator = ModelEvaluator(self.config)

        # State tracking
        self.detections: list[Detection] = []
        self.exported_paths: dict[str, Path] = {}
        self.trained_models: dict[str, Path] = {}
        self.eval_results: dict[str, dict] = {}

    def _print_banner(self, stage: str):
        """Print a stage banner."""
        print("\n" + "=" * 70)
        print(f"  SAM3 Detection Distillation Pipeline - {stage}")
        print(f"  Project: {self.config.name}")
        print("=" * 70)

    def scrape(self) -> dict[str, int]:
        """
        Stage 1: Scrape images from Bing.

        Returns:
            Dictionary mapping class names to image counts.
        """
        self._print_banner("SCRAPE")
        return self.scraper.scrape_all()

    def label(self) -> list[Detection]:
        """
        Stage 2: Auto-label images with SAM3.

        Returns:
            List of all detections.
        """
        self._print_banner("LABEL")
        self.detections = self.labeler.label_all()
        return self.detections

    def export(self, detections: list[Detection] = None) -> dict[str, Path]:
        """
        Stage 3: Export detections to YOLO/COCO format.

        Args:
            detections: Optional list of detections. Uses cached if not provided.

        Returns:
            Dictionary mapping format names to output paths.
        """
        self._print_banner("EXPORT")

        if detections is None:
            detections = self.detections

        if not detections:
            print("Error: No detections to export. Run label() first.")
            return {}

        self.exported_paths = self.exporter.export_all(detections)
        return self.exported_paths

    def train(self, data_paths: dict[str, Path] = None) -> dict[str, Path]:
        """
        Stage 4: Train YOLO and RT-DETR models.

        Args:
            data_paths: Optional dictionary of data.yaml paths. Uses cached if not provided.

        Returns:
            Dictionary mapping model names to weight paths.
        """
        self._print_banner("TRAIN")

        if data_paths is None:
            data_paths = self.exported_paths

        yolo_yaml = data_paths.get("yolo")
        coco_yaml = data_paths.get("coco")

        # For RT-DETR, we can use the YOLO data.yaml since Ultralytics handles both
        if coco_yaml and (coco_yaml / "data.yaml").exists():
            coco_yaml = coco_yaml / "data.yaml"
        elif coco_yaml:
            coco_yaml = None  # Invalid path

        self.trained_models = self.trainer.train_all(
            yolo_data_yaml=yolo_yaml,
            coco_data_yaml=coco_yaml
        )
        return self.trained_models

    def evaluate(
        self,
        trained_models: dict[str, Path] = None,
        data_yaml: Path = None
    ) -> dict[str, dict]:
        """
        Stage 5: Evaluate and compare trained models.

        Args:
            trained_models: Optional dictionary of model weights. Uses cached if not provided.
            data_yaml: Optional data.yaml path. Uses YOLO format if not provided.

        Returns:
            Dictionary mapping model names to evaluation metrics.
        """
        self._print_banner("EVALUATE")

        if trained_models is None:
            trained_models = self.trained_models

        if not trained_models:
            print("Error: No models to evaluate. Run train() first.")
            return {}

        if data_yaml is None:
            data_yaml = self.exported_paths.get("yolo")

        if not data_yaml:
            print("Error: No data.yaml found for evaluation.")
            return {}

        self.eval_results = self.evaluator.evaluate_all(trained_models, data_yaml)

        # Generate report
        report_path = self.config.logs_dir / "evaluation_report.md"
        self.evaluator.generate_report(self.eval_results, report_path)

        return self.eval_results

    def run(
        self,
        skip_scrape: bool = False,
        skip_label: bool = False,
        skip_export: bool = False,
        skip_train: bool = False,
        skip_eval: bool = False
    ) -> dict:
        """
        Run the full pipeline.

        Args:
            skip_scrape: Skip the scraping stage.
            skip_label: Skip the labeling stage.
            skip_export: Skip the export stage.
            skip_train: Skip the training stage.
            skip_eval: Skip the evaluation stage.

        Returns:
            Dictionary with all pipeline results.
        """
        start_time = datetime.now()

        print("\n" + "=" * 70)
        print("  SAM3 Detection Distillation Pipeline")
        print(f"  Project: {self.config.name}")
        print(f"  Config: {self.config_path}")
        print(f"  Output: {self.config.output_dir}")
        print("=" * 70)
        print(f"\nClasses ({len(self.config.classes)}):")
        for cls in self.config.classes:
            print(f"  - {cls.name}: prompt='{cls.prompt}', queries={len(cls.search_queries)}")
        print()

        results = {
            "config": str(self.config_path),
            "project": self.config.name,
            "stages": {}
        }

        # Stage 1: Scrape
        if not skip_scrape:
            scrape_results = self.scrape()
            results["stages"]["scrape"] = {
                "images_per_class": scrape_results,
                "total_images": sum(scrape_results.values())
            }
        else:
            print("\n[SKIP] Scraping stage skipped")

        # Stage 2: Label
        if not skip_label:
            detections = self.label()
            results["stages"]["label"] = {
                "total_detections": len(detections),
                "stats": self.exporter.get_dataset_stats(detections)
            }
        else:
            print("\n[SKIP] Labeling stage skipped")

        # Stage 3: Export
        if not skip_export:
            if self.detections:
                exported = self.export()
                results["stages"]["export"] = {
                    "formats": list(exported.keys()),
                    "paths": {k: str(v) for k, v in exported.items()}
                }
            else:
                print("\n[SKIP] Export skipped - no detections available")
        else:
            print("\n[SKIP] Export stage skipped")

        # Stage 4: Train
        if not skip_train:
            if self.exported_paths:
                trained = self.train()
                results["stages"]["train"] = {
                    "models": list(trained.keys()),
                    "weights": {k: str(v) for k, v in trained.items()}
                }
            else:
                print("\n[SKIP] Training skipped - no exported datasets")
        else:
            print("\n[SKIP] Training stage skipped")

        # Stage 5: Evaluate
        if not skip_eval:
            if self.trained_models:
                eval_results = self.evaluate()
                results["stages"]["evaluate"] = eval_results
            else:
                print("\n[SKIP] Evaluation skipped - no trained models")
        else:
            print("\n[SKIP] Evaluation stage skipped")

        # Summary
        end_time = datetime.now()
        duration = end_time - start_time

        print("\n" + "=" * 70)
        print("  Pipeline Complete!")
        print("=" * 70)
        print(f"  Duration: {duration}")
        print(f"  Output directory: {self.config.output_dir}")
        print()

        # Save results
        results["duration_seconds"] = duration.total_seconds()
        results["completed_at"] = end_time.isoformat()

        results_path = self.config.logs_dir / "pipeline_results.json"
        results_path.parent.mkdir(parents=True, exist_ok=True)
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Results saved to: {results_path}")

        return results

    def run_stage(self, stage: str) -> dict:
        """
        Run a single pipeline stage.

        Args:
            stage: Stage name (scrape, label, export, train, eval).

        Returns:
            Stage results.
        """
        stage_map = {
            "scrape": (self.scrape, {}),
            "label": (self.label, {}),
            "export": (self.export, {}),
            "train": (self.train, {}),
            "eval": (self.evaluate, {}),
            "evaluate": (self.evaluate, {}),
        }

        if stage not in stage_map:
            raise ValueError(f"Unknown stage: {stage}. Valid: {list(stage_map.keys())}")

        func, kwargs = stage_map[stage]
        return func(**kwargs)


def run_pipeline(config_path: str, **kwargs) -> dict:
    """
    Convenience function to run the full pipeline.

    Args:
        config_path: Path to the configuration file.
        **kwargs: Arguments passed to Pipeline.run().

    Returns:
        Pipeline results.
    """
    pipeline = Pipeline(config_path)
    return pipeline.run(**kwargs)
