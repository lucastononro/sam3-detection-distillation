"""
Dataset export module for YOLO and COCO formats.
"""

import json
import random
import shutil
from datetime import datetime
from pathlib import Path

import yaml
from tqdm import tqdm

from .config import Config
from .labeler import Detection


class DatasetExporter:
    """
    Exports detections to YOLO and COCO dataset formats.
    """

    def __init__(self, config: Config, seed: int = 42):
        """
        Initialize the exporter.

        Args:
            config: Pipeline configuration.
            seed: Random seed for reproducible splits.
        """
        self.config = config
        self.seed = seed
        self.output_dir = config.labeled_dir

    def split_detections(
        self,
        detections: list[Detection]
    ) -> dict[str, list[Detection]]:
        """
        Split detections into train/val/test sets by image.

        Args:
            detections: List of all detections.

        Returns:
            Dictionary with 'train', 'valid', 'test' keys.
        """
        # Group detections by image
        images = {}
        for det in detections:
            img_path = str(det.image_path)
            if img_path not in images:
                images[img_path] = []
            images[img_path].append(det)

        # Get unique image paths and shuffle
        image_paths = list(images.keys())
        random.seed(self.seed)
        random.shuffle(image_paths)

        # Calculate split sizes
        n_images = len(image_paths)
        train_ratio, val_ratio, test_ratio = self.config.dataset.split

        n_train = int(n_images * train_ratio)
        n_val = int(n_images * val_ratio)

        # Split image paths
        train_paths = set(image_paths[:n_train])
        val_paths = set(image_paths[n_train:n_train + n_val])
        test_paths = set(image_paths[n_train + n_val:])

        # Split detections
        splits = {"train": [], "valid": [], "test": []}
        for det in detections:
            img_path = str(det.image_path)
            if img_path in train_paths:
                splits["train"].append(det)
            elif img_path in val_paths:
                splits["valid"].append(det)
            else:
                splits["test"].append(det)

        print(f"\nDataset split:")
        print(f"  Train: {len(train_paths)} images, {len(splits['train'])} detections")
        print(f"  Valid: {len(val_paths)} images, {len(splits['valid'])} detections")
        print(f"  Test:  {len(test_paths)} images, {len(splits['test'])} detections")

        return splits

    def export_yolo(self, detections: list[Detection]) -> Path:
        """
        Export detections to YOLO format.

        Directory structure:
            yolo/
                train/
                    images/
                    labels/
                valid/
                    images/
                    labels/
                test/
                    images/
                    labels/
                data.yaml

        Args:
            detections: List of all detections.

        Returns:
            Path to the data.yaml file.
        """
        print("\n" + "="*60)
        print("Exporting to YOLO format")
        print("="*60)

        yolo_dir = self.output_dir / "yolo"
        splits = self.split_detections(detections)

        for split_name, split_dets in splits.items():
            split_dir = yolo_dir / split_name
            images_dir = split_dir / "images"
            labels_dir = split_dir / "labels"

            images_dir.mkdir(parents=True, exist_ok=True)
            labels_dir.mkdir(parents=True, exist_ok=True)

            # Group by image
            images = {}
            for det in split_dets:
                img_path = det.image_path
                if img_path not in images:
                    images[img_path] = []
                images[img_path].append(det)

            # Export each image
            for img_path, dets in tqdm(images.items(), desc=f"Exporting {split_name}", unit="img"):
                # Copy image
                img_name = img_path.name
                dst_img = images_dir / img_name
                if not dst_img.exists():
                    shutil.copy2(img_path, dst_img)

                # Create label file
                label_name = img_path.stem + ".txt"
                label_path = labels_dir / label_name

                with open(label_path, "w") as f:
                    for det in dets:
                        cx, cy, w, h = det.bbox_yolo
                        # YOLO format: class_id cx cy w h
                        f.write(f"{det.class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

        # Create data.yaml
        data_yaml = {
            "path": str(yolo_dir.absolute()),
            "train": "train/images",
            "val": "valid/images",
            "test": "test/images",
            "nc": len(self.config.class_names),
            "names": self.config.class_names
        }

        yaml_path = yolo_dir / "data.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(data_yaml, f, default_flow_style=False, sort_keys=False)

        print(f"\nYOLO dataset exported to: {yolo_dir}")
        print(f"data.yaml: {yaml_path}")

        return yaml_path

    def export_coco(self, detections: list[Detection]) -> Path:
        """
        Export detections to COCO format.

        Directory structure:
            coco/
                train/
                    images/
                    _annotations.coco.json
                valid/
                    images/
                    _annotations.coco.json
                test/
                    images/
                    _annotations.coco.json

        Args:
            detections: List of all detections.

        Returns:
            Path to the coco directory.
        """
        print("\n" + "="*60)
        print("Exporting to COCO format")
        print("="*60)

        coco_dir = self.output_dir / "coco"
        splits = self.split_detections(detections)

        for split_name, split_dets in splits.items():
            split_dir = coco_dir / split_name
            images_dir = split_dir / "images"
            images_dir.mkdir(parents=True, exist_ok=True)

            # Group by image
            images = {}
            for det in split_dets:
                img_path = det.image_path
                if img_path not in images:
                    images[img_path] = {
                        "detections": [],
                        "width": det.image_width,
                        "height": det.image_height
                    }
                images[img_path]["detections"].append(det)

            # Build COCO structure
            coco_data = {
                "info": {
                    "description": f"SAM3 Auto-labeled Dataset - {self.config.name}",
                    "version": "1.0",
                    "year": datetime.now().year,
                    "date_created": datetime.now().isoformat()
                },
                "licenses": [],
                "images": [],
                "annotations": [],
                "categories": [
                    {"id": i, "name": name, "supercategory": "object"}
                    for i, name in enumerate(self.config.class_names)
                ]
            }

            annotation_id = 1
            for img_id, (img_path, img_data) in enumerate(
                tqdm(images.items(), desc=f"Exporting {split_name}", unit="img"),
                start=1
            ):
                # Copy image
                img_name = img_path.name
                dst_img = images_dir / img_name
                if not dst_img.exists():
                    shutil.copy2(img_path, dst_img)

                # Add image entry
                coco_data["images"].append({
                    "id": img_id,
                    "file_name": img_name,
                    "width": img_data["width"],
                    "height": img_data["height"]
                })

                # Add annotations
                for det in img_data["detections"]:
                    x, y, w, h = det.bbox_coco
                    area = w * h

                    coco_data["annotations"].append({
                        "id": annotation_id,
                        "image_id": img_id,
                        "category_id": det.class_id,
                        "bbox": [x, y, w, h],
                        "area": area,
                        "iscrowd": 0,
                        "score": det.score
                    })
                    annotation_id += 1

            # Save JSON
            json_path = split_dir / "_annotations.coco.json"
            with open(json_path, "w") as f:
                json.dump(coco_data, f, indent=2)

        # Create data.yaml for RT-DETR (uses same format as YOLO but points to COCO)
        data_yaml = {
            "path": str(coco_dir.absolute()),
            "train": "train/images",
            "val": "valid/images",
            "test": "test/images",
            "nc": len(self.config.class_names),
            "names": self.config.class_names
        }

        yaml_path = coco_dir / "data.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(data_yaml, f, default_flow_style=False, sort_keys=False)

        print(f"\nCOCO dataset exported to: {coco_dir}")

        return coco_dir

    def export_all(self, detections: list[Detection]) -> dict[str, Path]:
        """
        Export detections to all configured formats.

        Args:
            detections: List of all detections.

        Returns:
            Dictionary mapping format names to output paths.
        """
        print("\n" + "="*60)
        print("SAM3 Detection Distillation - Dataset Export")
        print("="*60)
        print(f"Output directory: {self.output_dir}")
        print(f"Formats: {self.config.dataset.formats}")
        print(f"Total detections: {len(detections)}")

        results = {}

        if "yolo" in self.config.dataset.formats:
            results["yolo"] = self.export_yolo(detections)

        if "coco" in self.config.dataset.formats:
            results["coco"] = self.export_coco(detections)

        print("\n" + "="*60)
        print("Export Complete!")
        print("="*60)
        for fmt, path in results.items():
            print(f"  {fmt}: {path}")

        return results

    def get_dataset_stats(self, detections: list[Detection]) -> dict:
        """
        Get statistics about the dataset.

        Args:
            detections: List of all detections.

        Returns:
            Dictionary with dataset statistics.
        """
        # Count detections per class
        class_counts = {}
        for det in detections:
            class_counts[det.class_name] = class_counts.get(det.class_name, 0) + 1

        # Count unique images
        unique_images = len(set(det.image_path for det in detections))

        # Average detections per image
        avg_per_image = len(detections) / unique_images if unique_images > 0 else 0

        return {
            "total_detections": len(detections),
            "unique_images": unique_images,
            "avg_detections_per_image": avg_per_image,
            "class_counts": class_counts
        }
