"""
Model training module for YOLO and RT-DETR.
"""

import os
from pathlib import Path

from .config import Config


class ModelTrainer:
    """
    Trains YOLO and RT-DETR models on the exported datasets.
    """

    def __init__(self, config: Config):
        """
        Initialize the trainer.

        Args:
            config: Pipeline configuration.
        """
        self.config = config
        self.models_dir = config.models_dir
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def _setup_wandb(self, model_type: str) -> None:
        """
        Setup Weights & Biases logging if enabled.

        Args:
            model_type: Type of model being trained (yolo, rtdetr).
        """
        if not self.config.logging.wandb.enabled:
            return

        try:
            import wandb

            wandb.init(
                project=self.config.logging.wandb.project,
                entity=self.config.logging.wandb.entity,
                name=f"{self.config.name}-{model_type}",
                config={
                    "project": self.config.name,
                    "model_type": model_type,
                    "classes": self.config.class_names,
                    "num_classes": len(self.config.class_names),
                }
            )
            print(f"W&B logging enabled: {wandb.run.url}")
        except Exception as e:
            print(f"Warning: Could not initialize W&B: {e}")

    def _finish_wandb(self) -> None:
        """Finish W&B run if active."""
        if not self.config.logging.wandb.enabled:
            return

        try:
            import wandb
            if wandb.run is not None:
                wandb.finish()
        except Exception:
            pass

    def train_yolo(self, data_yaml: Path) -> Path:
        """
        Train a YOLO model on the dataset.

        Args:
            data_yaml: Path to the YOLO data.yaml file.

        Returns:
            Path to the best model weights.
        """
        if not self.config.training.yolo.enabled:
            print("YOLO training is disabled in config")
            return None

        print("\n" + "="*60)
        print("Training YOLO Model")
        print("="*60)

        yolo_config = self.config.training.yolo
        print(f"Model: {yolo_config.model}")
        print(f"Epochs: {yolo_config.epochs}")
        print(f"Batch size: {yolo_config.batch_size}")
        print(f"Image size: {yolo_config.imgsz}")
        print(f"Dataset: {data_yaml}")

        try:
            from ultralytics import YOLO

            # Initialize model
            model = YOLO(f"{yolo_config.model}.pt")

            # Output directory
            output_dir = self.models_dir / "yolo"
            output_dir.mkdir(parents=True, exist_ok=True)

            # Setup W&B
            self._setup_wandb("yolo")

            # Train
            results = model.train(
                data=str(data_yaml),
                epochs=yolo_config.epochs,
                batch=yolo_config.batch_size,
                imgsz=yolo_config.imgsz,
                project=str(output_dir),
                name="train",
                exist_ok=True,
                verbose=True,
                plots=True,
                save=True,
            )

            # Finish W&B
            self._finish_wandb()

            # Find best weights
            best_weights = output_dir / "train" / "weights" / "best.pt"
            if best_weights.exists():
                print(f"\nYOLO training complete!")
                print(f"Best weights: {best_weights}")
                return best_weights
            else:
                # Try to find weights in results
                print(f"Warning: Could not find best.pt at expected location")
                return None

        except ImportError:
            print("Error: ultralytics package not installed")
            print("Install with: pip install ultralytics")
            return None

        except Exception as e:
            print(f"Error training YOLO: {e}")
            self._finish_wandb()
            return None

    def train_rtdetr(self, data_yaml: Path) -> Path:
        """
        Train an RT-DETR model on the dataset.

        Args:
            data_yaml: Path to the data.yaml file (can use YOLO format).

        Returns:
            Path to the best model weights.
        """
        if not self.config.training.rtdetr.enabled:
            print("RT-DETR training is disabled in config")
            return None

        print("\n" + "="*60)
        print("Training RT-DETR Model")
        print("="*60)

        rtdetr_config = self.config.training.rtdetr
        print(f"Model: {rtdetr_config.model}")
        print(f"Epochs: {rtdetr_config.epochs}")
        print(f"Batch size: {rtdetr_config.batch_size}")
        print(f"Image size: {rtdetr_config.imgsz}")
        print(f"Dataset: {data_yaml}")

        try:
            from ultralytics import RTDETR

            # Initialize model
            model = RTDETR(f"{rtdetr_config.model}.pt")

            # Output directory
            output_dir = self.models_dir / "rtdetr"
            output_dir.mkdir(parents=True, exist_ok=True)

            # Setup W&B
            self._setup_wandb("rtdetr")

            # Train
            results = model.train(
                data=str(data_yaml),
                epochs=rtdetr_config.epochs,
                batch=rtdetr_config.batch_size,
                imgsz=rtdetr_config.imgsz,
                project=str(output_dir),
                name="train",
                exist_ok=True,
                verbose=True,
                plots=True,
                save=True,
            )

            # Finish W&B
            self._finish_wandb()

            # Find best weights
            best_weights = output_dir / "train" / "weights" / "best.pt"
            if best_weights.exists():
                print(f"\nRT-DETR training complete!")
                print(f"Best weights: {best_weights}")
                return best_weights
            else:
                print(f"Warning: Could not find best.pt at expected location")
                return None

        except ImportError:
            print("Error: ultralytics package not installed")
            print("Install with: pip install ultralytics")
            return None

        except Exception as e:
            print(f"Error training RT-DETR: {e}")
            self._finish_wandb()
            return None

    def train_all(self, yolo_data_yaml: Path = None, coco_data_yaml: Path = None) -> dict[str, Path]:
        """
        Train all enabled models.

        Args:
            yolo_data_yaml: Path to YOLO format data.yaml.
            coco_data_yaml: Path to COCO format data.yaml (falls back to yolo_data_yaml).

        Returns:
            Dictionary mapping model names to best weight paths.
        """
        print("\n" + "="*60)
        print("SAM3 Detection Distillation - Model Training")
        print("="*60)

        results = {}

        # Train YOLO
        if self.config.training.yolo.enabled and yolo_data_yaml:
            weights = self.train_yolo(yolo_data_yaml)
            if weights:
                results["yolo"] = weights

        # Train RT-DETR (can use YOLO format data.yaml with Ultralytics)
        if self.config.training.rtdetr.enabled:
            # RT-DETR in Ultralytics uses same data.yaml format as YOLO
            rtdetr_data = coco_data_yaml or yolo_data_yaml
            if rtdetr_data:
                weights = self.train_rtdetr(rtdetr_data)
                if weights:
                    results["rtdetr"] = weights

        print("\n" + "="*60)
        print("Training Complete!")
        print("="*60)

        if results:
            for model_name, weights_path in results.items():
                print(f"  {model_name}: {weights_path}")
        else:
            print("  No models were trained successfully")

        return results


class InferenceRunner:
    """
    Run inference with trained models for testing/demo purposes.
    """

    def __init__(self, model_path: Path, model_type: str = "yolo"):
        """
        Initialize the inference runner.

        Args:
            model_path: Path to the trained model weights.
            model_type: Type of model (yolo, rtdetr).
        """
        self.model_path = model_path
        self.model_type = model_type
        self.model = None

    def load_model(self):
        """Load the model."""
        if self.model is not None:
            return

        from ultralytics import YOLO, RTDETR

        if self.model_type == "yolo":
            self.model = YOLO(str(self.model_path))
        elif self.model_type == "rtdetr":
            self.model = RTDETR(str(self.model_path))
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        print(f"Loaded {self.model_type} model from {self.model_path}")

    def predict(self, image_path: Path, conf: float = 0.25) -> dict:
        """
        Run inference on a single image.

        Args:
            image_path: Path to the image.
            conf: Confidence threshold.

        Returns:
            Dictionary with predictions.
        """
        self.load_model()

        results = self.model.predict(
            source=str(image_path),
            conf=conf,
            verbose=False
        )[0]

        detections = []
        for box in results.boxes:
            detections.append({
                "class_id": int(box.cls),
                "class_name": results.names[int(box.cls)],
                "confidence": float(box.conf),
                "bbox": box.xyxy[0].tolist()  # [x1, y1, x2, y2]
            })

        return {
            "image_path": str(image_path),
            "detections": detections
        }

    def predict_batch(self, image_dir: Path, conf: float = 0.25) -> list[dict]:
        """
        Run inference on all images in a directory.

        Args:
            image_dir: Directory containing images.
            conf: Confidence threshold.

        Returns:
            List of prediction dictionaries.
        """
        self.load_model()

        image_paths = []
        for ext in [".jpg", ".jpeg", ".png", ".webp"]:
            image_paths.extend(image_dir.glob(f"*{ext}"))

        results = []
        for img_path in image_paths:
            pred = self.predict(img_path, conf)
            results.append(pred)

        return results
