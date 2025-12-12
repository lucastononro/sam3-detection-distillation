"""
Model evaluation module for computing metrics and comparing models.
"""

from pathlib import Path

from .config import Config


class ModelEvaluator:
    """
    Evaluates trained models and compares their performance.
    """

    def __init__(self, config: Config):
        """
        Initialize the evaluator.

        Args:
            config: Pipeline configuration.
        """
        self.config = config

    def evaluate_yolo(self, model_path: Path, data_yaml: Path) -> dict:
        """
        Evaluate a YOLO model on the test set.

        Args:
            model_path: Path to the trained model weights.
            data_yaml: Path to the data.yaml file.

        Returns:
            Dictionary with evaluation metrics.
        """
        print("\n" + "="*60)
        print(f"Evaluating YOLO Model")
        print("="*60)
        print(f"Model: {model_path}")
        print(f"Dataset: {data_yaml}")

        try:
            from ultralytics import YOLO

            model = YOLO(str(model_path))
            results = model.val(
                data=str(data_yaml),
                split="test",
                verbose=True,
                plots=True
            )

            metrics = {
                "model_type": "yolo",
                "model_path": str(model_path),
                "mAP50": float(results.box.map50),
                "mAP50-95": float(results.box.map),
                "precision": float(results.box.mp),
                "recall": float(results.box.mr),
                "per_class_ap50": {},
                "per_class_ap50-95": {}
            }

            # Per-class metrics
            for i, class_name in enumerate(self.config.class_names):
                if i < len(results.box.ap50):
                    metrics["per_class_ap50"][class_name] = float(results.box.ap50[i])
                if i < len(results.box.ap):
                    metrics["per_class_ap50-95"][class_name] = float(results.box.ap[i])

            return metrics

        except Exception as e:
            print(f"Error evaluating YOLO: {e}")
            return {"error": str(e)}

    def evaluate_rtdetr(self, model_path: Path, data_yaml: Path) -> dict:
        """
        Evaluate an RT-DETR model on the test set.

        Args:
            model_path: Path to the trained model weights.
            data_yaml: Path to the data.yaml file.

        Returns:
            Dictionary with evaluation metrics.
        """
        print("\n" + "="*60)
        print(f"Evaluating RT-DETR Model")
        print("="*60)
        print(f"Model: {model_path}")
        print(f"Dataset: {data_yaml}")

        try:
            from ultralytics import RTDETR

            model = RTDETR(str(model_path))
            results = model.val(
                data=str(data_yaml),
                split="test",
                verbose=True,
                plots=True
            )

            metrics = {
                "model_type": "rtdetr",
                "model_path": str(model_path),
                "mAP50": float(results.box.map50),
                "mAP50-95": float(results.box.map),
                "precision": float(results.box.mp),
                "recall": float(results.box.mr),
                "per_class_ap50": {},
                "per_class_ap50-95": {}
            }

            # Per-class metrics
            for i, class_name in enumerate(self.config.class_names):
                if i < len(results.box.ap50):
                    metrics["per_class_ap50"][class_name] = float(results.box.ap50[i])
                if i < len(results.box.ap):
                    metrics["per_class_ap50-95"][class_name] = float(results.box.ap[i])

            return metrics

        except Exception as e:
            print(f"Error evaluating RT-DETR: {e}")
            return {"error": str(e)}

    def evaluate(self, model_path: Path, data_yaml: Path, model_type: str = "yolo") -> dict:
        """
        Evaluate a model on the test set.

        Args:
            model_path: Path to the trained model weights.
            data_yaml: Path to the data.yaml file.
            model_type: Type of model (yolo, rtdetr).

        Returns:
            Dictionary with evaluation metrics.
        """
        if model_type == "yolo":
            return self.evaluate_yolo(model_path, data_yaml)
        elif model_type == "rtdetr":
            return self.evaluate_rtdetr(model_path, data_yaml)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def compare_models(
        self,
        yolo_metrics: dict = None,
        rtdetr_metrics: dict = None
    ) -> None:
        """
        Print a comparison table of model metrics.

        Args:
            yolo_metrics: YOLO evaluation metrics.
            rtdetr_metrics: RT-DETR evaluation metrics.
        """
        print("\n" + "="*60)
        print("Model Comparison")
        print("="*60)

        # Header
        print(f"\n{'Metric':<20} {'YOLO':<15} {'RT-DETR':<15} {'Winner':<10}")
        print("-" * 60)

        metrics_to_compare = [
            ("mAP50", "mAP50"),
            ("mAP50-95", "mAP50-95"),
            ("Precision", "precision"),
            ("Recall", "recall")
        ]

        for display_name, key in metrics_to_compare:
            yolo_val = yolo_metrics.get(key, "N/A") if yolo_metrics else "N/A"
            rtdetr_val = rtdetr_metrics.get(key, "N/A") if rtdetr_metrics else "N/A"

            # Determine winner
            winner = ""
            if isinstance(yolo_val, (int, float)) and isinstance(rtdetr_val, (int, float)):
                if yolo_val > rtdetr_val:
                    winner = "YOLO"
                elif rtdetr_val > yolo_val:
                    winner = "RT-DETR"
                else:
                    winner = "Tie"

            # Format values
            yolo_str = f"{yolo_val:.4f}" if isinstance(yolo_val, (int, float)) else str(yolo_val)
            rtdetr_str = f"{rtdetr_val:.4f}" if isinstance(rtdetr_val, (int, float)) else str(rtdetr_val)

            print(f"{display_name:<20} {yolo_str:<15} {rtdetr_str:<15} {winner:<10}")

        # Per-class comparison
        if yolo_metrics and rtdetr_metrics:
            print("\n" + "-" * 60)
            print("Per-Class AP50:")
            print("-" * 60)

            yolo_per_class = yolo_metrics.get("per_class_ap50", {})
            rtdetr_per_class = rtdetr_metrics.get("per_class_ap50", {})

            for class_name in self.config.class_names:
                yolo_val = yolo_per_class.get(class_name, "N/A")
                rtdetr_val = rtdetr_per_class.get(class_name, "N/A")

                winner = ""
                if isinstance(yolo_val, (int, float)) and isinstance(rtdetr_val, (int, float)):
                    if yolo_val > rtdetr_val:
                        winner = "YOLO"
                    elif rtdetr_val > yolo_val:
                        winner = "RT-DETR"

                yolo_str = f"{yolo_val:.4f}" if isinstance(yolo_val, (int, float)) else str(yolo_val)
                rtdetr_str = f"{rtdetr_val:.4f}" if isinstance(rtdetr_val, (int, float)) else str(rtdetr_val)

                print(f"  {class_name:<18} {yolo_str:<15} {rtdetr_str:<15} {winner:<10}")

        print()

    def evaluate_all(
        self,
        trained_models: dict[str, Path],
        data_yaml: Path
    ) -> dict[str, dict]:
        """
        Evaluate all trained models and compare them.

        Args:
            trained_models: Dictionary mapping model types to weight paths.
            data_yaml: Path to the data.yaml file.

        Returns:
            Dictionary mapping model types to their metrics.
        """
        print("\n" + "="*60)
        print("SAM3 Detection Distillation - Model Evaluation")
        print("="*60)

        results = {}

        # Evaluate each model
        for model_type, model_path in trained_models.items():
            if model_path and model_path.exists():
                metrics = self.evaluate(model_path, data_yaml, model_type)
                results[model_type] = metrics

        # Compare if we have both
        yolo_metrics = results.get("yolo")
        rtdetr_metrics = results.get("rtdetr")

        if yolo_metrics or rtdetr_metrics:
            self.compare_models(yolo_metrics, rtdetr_metrics)

        return results

    def generate_report(
        self,
        metrics: dict[str, dict],
        output_path: Path = None
    ) -> str:
        """
        Generate a markdown report of the evaluation results.

        Args:
            metrics: Dictionary of model metrics.
            output_path: Optional path to save the report.

        Returns:
            Markdown report string.
        """
        report = []
        report.append("# SAM3 Detection Distillation - Evaluation Report\n")
        report.append(f"**Project:** {self.config.name}\n")
        report.append(f"**Classes:** {', '.join(self.config.class_names)}\n")
        report.append("")

        # Summary table
        report.append("## Summary\n")
        report.append("| Metric | " + " | ".join(metrics.keys()) + " |")
        report.append("|--------|" + "|".join(["--------"] * len(metrics)) + "|")

        metric_keys = ["mAP50", "mAP50-95", "precision", "recall"]
        for key in metric_keys:
            row = f"| {key} |"
            for model_type, model_metrics in metrics.items():
                val = model_metrics.get(key, "N/A")
                val_str = f"{val:.4f}" if isinstance(val, (int, float)) else str(val)
                row += f" {val_str} |"
            report.append(row)

        report.append("")

        # Per-class metrics
        report.append("## Per-Class AP50\n")
        report.append("| Class | " + " | ".join(metrics.keys()) + " |")
        report.append("|-------|" + "|".join(["--------"] * len(metrics)) + "|")

        for class_name in self.config.class_names:
            row = f"| {class_name} |"
            for model_type, model_metrics in metrics.items():
                per_class = model_metrics.get("per_class_ap50", {})
                val = per_class.get(class_name, "N/A")
                val_str = f"{val:.4f}" if isinstance(val, (int, float)) else str(val)
                row += f" {val_str} |"
            report.append(row)

        report_str = "\n".join(report)

        # Save if path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                f.write(report_str)
            print(f"Report saved to: {output_path}")

        return report_str
