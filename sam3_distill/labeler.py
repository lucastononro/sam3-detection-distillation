"""
SAM3 auto-labeling module for generating bounding box annotations.
"""

import os
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

from .config import Config


@dataclass
class Detection:
    """A single detection from SAM3."""
    image_path: Path
    class_name: str
    class_id: int
    bbox: tuple[float, float, float, float]  # (x1, y1, x2, y2) in pixels
    score: float
    image_width: int
    image_height: int

    @property
    def bbox_normalized(self) -> tuple[float, float, float, float]:
        """Get bounding box normalized to [0, 1]."""
        x1, y1, x2, y2 = self.bbox
        return (
            x1 / self.image_width,
            y1 / self.image_height,
            x2 / self.image_width,
            y2 / self.image_height
        )

    @property
    def bbox_yolo(self) -> tuple[float, float, float, float]:
        """Get bounding box in YOLO format (cx, cy, w, h) normalized."""
        x1, y1, x2, y2 = self.bbox_normalized
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        return (cx, cy, w, h)

    @property
    def bbox_coco(self) -> tuple[float, float, float, float]:
        """Get bounding box in COCO format (x, y, w, h) in pixels."""
        x1, y1, x2, y2 = self.bbox
        return (x1, y1, x2 - x1, y2 - y1)


class SAM3Labeler:
    """
    Auto-labeling using SAM3 (Segment Anything Model 3).

    Uses text prompts to detect objects and generate bounding boxes.
    """

    # Supported image extensions
    VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

    def __init__(self, config: Config, device: str = "auto"):
        """
        Initialize the labeler.

        Args:
            config: Pipeline configuration.
            device: Device to run inference on (auto, cuda, mps, cpu).
        """
        self.config = config
        self.device = self._get_device(device if device != "auto" else config.labeling.device)
        self.threshold = config.labeling.threshold
        self.max_detections = config.labeling.max_detections_per_image

        self.model = None
        self.processor = None

    def _get_device(self, device: str) -> str:
        """Determine the best device to use."""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return device

    def load_model(self):
        """Load the SAM3 model and processor."""
        if self.model is not None:
            return

        print(f"Loading SAM3 model on {self.device}...")

        # Login to HuggingFace if token is available
        token = os.environ.get("HF_TOKEN")
        if token:
            from huggingface_hub import login
            login(token=token, add_to_git_credential=False)

        from transformers import Sam3Model, Sam3Processor

        self.processor = Sam3Processor.from_pretrained("facebook/sam3")
        self.model = Sam3Model.from_pretrained("facebook/sam3").to(self.device)
        self.model.eval()

        print("SAM3 model loaded successfully!")

    def label_image(
        self,
        image_path: Path,
        prompts: dict[str, str]
    ) -> list[Detection]:
        """
        Label a single image with all class prompts.

        Args:
            image_path: Path to the image file.
            prompts: Mapping of class name to SAM3 text prompt.

        Returns:
            List of detections found in the image.
        """
        self.load_model()

        # Load image
        try:
            image = Image.open(image_path).convert("RGB")
            width, height = image.size
        except Exception as e:
            print(f"Warning: Could not load image {image_path}: {e}")
            return []

        detections = []

        # Run inference for each class prompt
        for class_name, prompt in prompts.items():
            class_id = self.config.get_class_id(class_name)

            try:
                # Process inputs
                inputs = self.processor(
                    images=image,
                    text=prompt,
                    return_tensors="pt"
                ).to(self.device)

                # Run inference
                with torch.no_grad():
                    outputs = self.model(**inputs)

                # Post-process results
                results = self.processor.post_process_instance_segmentation(
                    outputs,
                    threshold=self.threshold,
                    mask_threshold=0.5,
                    target_sizes=[(height, width)]
                )[0]

                # Extract detections
                boxes = results.get("boxes", [])
                scores = results.get("scores", [])

                if len(boxes) == 0:
                    continue

                # Convert to list and sort by score
                box_score_pairs = list(zip(boxes, scores))
                box_score_pairs.sort(key=lambda x: x[1], reverse=True)

                # Limit detections per class
                for box, score in box_score_pairs[:self.max_detections]:
                    # Convert tensor to float
                    if hasattr(box, 'cpu'):
                        box = box.cpu().numpy()
                    if hasattr(score, 'item'):
                        score = score.item()

                    x1, y1, x2, y2 = box
                    detections.append(Detection(
                        image_path=image_path,
                        class_name=class_name,
                        class_id=class_id,
                        bbox=(float(x1), float(y1), float(x2), float(y2)),
                        score=float(score),
                        image_width=width,
                        image_height=height
                    ))

            except Exception as e:
                print(f"Warning: Error processing '{prompt}' on {image_path}: {e}")
                continue

        return detections

    def label_directory(
        self,
        image_dir: Path,
        prompts: dict[str, str]
    ) -> list[Detection]:
        """
        Label all images in a directory.

        Args:
            image_dir: Directory containing images.
            prompts: Mapping of class name to SAM3 text prompt.

        Returns:
            List of all detections found.
        """
        # Find all images
        image_paths = []
        for ext in self.VALID_EXTENSIONS:
            image_paths.extend(image_dir.glob(f"*{ext}"))
            image_paths.extend(image_dir.glob(f"*{ext.upper()}"))

        if not image_paths:
            print(f"Warning: No images found in {image_dir}")
            return []

        print(f"Found {len(image_paths)} images in {image_dir}")

        detections = []
        for img_path in tqdm(image_paths, desc=f"Labeling {image_dir.name}", unit="img"):
            dets = self.label_image(img_path, prompts)
            detections.extend(dets)

        return detections

    def label_all(self) -> list[Detection]:
        """
        Label all images in the raw directory for all classes.

        Returns:
            List of all detections found across all images.
        """
        self.load_model()

        print("\n" + "="*60)
        print("SAM3 Detection Distillation - Auto-Labeling")
        print("="*60)
        print(f"Input directory: {self.config.raw_dir}")
        print(f"Classes: {self.config.class_names}")
        print(f"Prompts: {self.config.prompts}")
        print(f"Threshold: {self.threshold}")
        print(f"Device: {self.device}")

        all_detections = []
        images_processed = set()

        # Process each class directory
        for cls in self.config.classes:
            class_dir = self.config.raw_dir / cls.name

            if not class_dir.exists():
                print(f"Warning: Directory not found for class '{cls.name}': {class_dir}")
                continue

            # For each image, run all prompts (not just the class prompt)
            # This allows detecting multiple classes in scraped images
            image_paths = []
            for ext in self.VALID_EXTENSIONS:
                image_paths.extend(class_dir.glob(f"*{ext}"))
                image_paths.extend(class_dir.glob(f"*{ext.upper()}"))

            print(f"\nProcessing class '{cls.name}': {len(image_paths)} images")

            for img_path in tqdm(image_paths, desc=f"Labeling {cls.name}", unit="img"):
                if img_path in images_processed:
                    continue
                images_processed.add(img_path)

                # Run inference with all prompts
                dets = self.label_image(img_path, self.config.prompts)
                all_detections.extend(dets)

        # Print summary
        print("\n" + "="*60)
        print("Labeling Complete!")
        print("="*60)
        print(f"Total images processed: {len(images_processed)}")
        print(f"Total detections: {len(all_detections)}")

        # Per-class statistics
        class_counts = {}
        for det in all_detections:
            class_counts[det.class_name] = class_counts.get(det.class_name, 0) + 1

        for class_name in self.config.class_names:
            count = class_counts.get(class_name, 0)
            print(f"  {class_name}: {count} detections")

        return all_detections

    def get_images_with_detections(
        self,
        detections: list[Detection]
    ) -> dict[Path, list[Detection]]:
        """
        Group detections by image path.

        Args:
            detections: List of all detections.

        Returns:
            Dictionary mapping image paths to their detections.
        """
        grouped = {}
        for det in detections:
            if det.image_path not in grouped:
                grouped[det.image_path] = []
            grouped[det.image_path].append(det)
        return grouped
