"""
Configuration loading and validation for SAM3 Detection Distillation.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import yaml


@dataclass
class ClassConfig:
    """Configuration for a single class."""
    name: str
    search_queries: list[str]
    prompt: str  # Simple 1-2 word prompt for SAM3


@dataclass
class ScrapingConfig:
    """Configuration for image scraping."""
    images_per_class: int = 1000
    timeout: int = 60


@dataclass
class LabelingConfig:
    """Configuration for SAM3 labeling."""
    threshold: float = 0.5
    max_detections_per_image: int = 10
    device: str = "auto"  # auto, cuda, mps, cpu


@dataclass
class DatasetConfig:
    """Configuration for dataset creation."""
    split: tuple[float, float, float] = (0.7, 0.2, 0.1)  # train, val, test
    formats: list[str] = field(default_factory=lambda: ["yolo", "coco"])


@dataclass
class YOLOTrainingConfig:
    """Configuration for YOLO training."""
    enabled: bool = True
    model: str = "yolov8m"
    epochs: int = 100
    batch_size: int = 16
    imgsz: int = 640


@dataclass
class RTDETRTrainingConfig:
    """Configuration for RT-DETR training."""
    enabled: bool = True
    model: str = "rtdetr-l"
    epochs: int = 100
    batch_size: int = 8
    imgsz: int = 640


@dataclass
class TrainingConfig:
    """Configuration for model training."""
    yolo: YOLOTrainingConfig = field(default_factory=YOLOTrainingConfig)
    rtdetr: RTDETRTrainingConfig = field(default_factory=RTDETRTrainingConfig)


@dataclass
class WandbConfig:
    """Configuration for Weights & Biases logging."""
    enabled: bool = False
    project: str = "sam3-distillation"
    entity: Optional[str] = None


@dataclass
class LoggingConfig:
    """Configuration for logging."""
    wandb: WandbConfig = field(default_factory=WandbConfig)


@dataclass
class Config:
    """Main configuration for the SAM3 Detection Distillation pipeline."""
    # Project settings
    name: str
    output_dir: Path

    # Classes to detect
    classes: list[ClassConfig]

    # Pipeline stages
    scraping: ScrapingConfig = field(default_factory=ScrapingConfig)
    labeling: LabelingConfig = field(default_factory=LabelingConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    @property
    def raw_dir(self) -> Path:
        """Directory for raw scraped images."""
        return self.output_dir / "raw"

    @property
    def labeled_dir(self) -> Path:
        """Directory for labeled datasets."""
        return self.output_dir / "labeled"

    @property
    def models_dir(self) -> Path:
        """Directory for trained models."""
        return self.output_dir / "models"

    @property
    def logs_dir(self) -> Path:
        """Directory for logs."""
        return self.output_dir / "logs"

    @property
    def class_names(self) -> list[str]:
        """List of class names in order."""
        return [c.name for c in self.classes]

    @property
    def prompts(self) -> dict[str, str]:
        """Mapping of class name to SAM3 prompt."""
        return {c.name: c.prompt for c in self.classes}

    def get_class_id(self, class_name: str) -> int:
        """Get the class ID for a class name."""
        return self.class_names.index(class_name)


def load_config(config_path: str | Path) -> Config:
    """
    Load configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Config object with all settings.

    Raises:
        FileNotFoundError: If the config file doesn't exist.
        ValueError: If required fields are missing or invalid.
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        data = yaml.safe_load(f)

    # Validate required fields
    if "project" not in data:
        raise ValueError("Config must have 'project' section with 'name' and 'output_dir'")
    if "classes" not in data:
        raise ValueError("Config must have 'classes' section")

    # Parse project settings
    project = data["project"]
    name = project.get("name")
    output_dir = Path(project.get("output_dir", f"./projects/{name}"))

    if not name:
        raise ValueError("Project 'name' is required")

    # Parse classes
    classes = []
    for class_name, class_data in data["classes"].items():
        if not isinstance(class_data, dict):
            raise ValueError(f"Class '{class_name}' must be a dictionary")

        search_queries = class_data.get("search_queries", [])
        prompt = class_data.get("prompt", class_name)  # Default to class name

        if not search_queries:
            raise ValueError(f"Class '{class_name}' must have 'search_queries'")

        classes.append(ClassConfig(
            name=class_name,
            search_queries=search_queries,
            prompt=prompt
        ))

    # Parse scraping config
    scraping_data = data.get("scraping", {})
    scraping = ScrapingConfig(
        images_per_class=scraping_data.get("images_per_class", 1000),
        timeout=scraping_data.get("timeout", 60)
    )

    # Parse labeling config
    labeling_data = data.get("labeling", {})
    labeling = LabelingConfig(
        threshold=labeling_data.get("threshold", 0.5),
        max_detections_per_image=labeling_data.get("max_detections_per_image", 10),
        device=labeling_data.get("device", "auto")
    )

    # Parse dataset config
    dataset_data = data.get("dataset", {})
    split = dataset_data.get("split", [0.7, 0.2, 0.1])
    if isinstance(split, list):
        split = tuple(split)
    dataset = DatasetConfig(
        split=split,
        formats=dataset_data.get("formats", ["yolo", "coco"])
    )

    # Parse training config
    training_data = data.get("training", {})

    yolo_data = training_data.get("yolo", {})
    yolo = YOLOTrainingConfig(
        enabled=yolo_data.get("enabled", True),
        model=yolo_data.get("model", "yolov8m"),
        epochs=yolo_data.get("epochs", 100),
        batch_size=yolo_data.get("batch_size", 16),
        imgsz=yolo_data.get("imgsz", 640)
    )

    rtdetr_data = training_data.get("rtdetr", {})
    rtdetr = RTDETRTrainingConfig(
        enabled=rtdetr_data.get("enabled", True),
        model=rtdetr_data.get("model", "rtdetr-l"),
        epochs=rtdetr_data.get("epochs", 100),
        batch_size=rtdetr_data.get("batch_size", 8),
        imgsz=rtdetr_data.get("imgsz", 640)
    )

    training = TrainingConfig(yolo=yolo, rtdetr=rtdetr)

    # Parse logging config
    logging_data = data.get("logging", {})
    wandb_data = logging_data.get("wandb", {})
    wandb = WandbConfig(
        enabled=wandb_data.get("enabled", False),
        project=wandb_data.get("project", "sam3-distillation"),
        entity=wandb_data.get("entity")
    )
    logging_config = LoggingConfig(wandb=wandb)

    return Config(
        name=name,
        output_dir=output_dir,
        classes=classes,
        scraping=scraping,
        labeling=labeling,
        dataset=dataset,
        training=training,
        logging=logging_config
    )


def validate_config(config: Config) -> list[str]:
    """
    Validate a configuration and return a list of warnings/errors.

    Args:
        config: The configuration to validate.

    Returns:
        List of warning/error messages (empty if valid).
    """
    warnings = []

    # Check split sums to 1
    split_sum = sum(config.dataset.split)
    if abs(split_sum - 1.0) > 0.01:
        warnings.append(f"Dataset split should sum to 1.0, got {split_sum}")

    # Check valid formats
    valid_formats = {"yolo", "coco"}
    for fmt in config.dataset.formats:
        if fmt not in valid_formats:
            warnings.append(f"Unknown format '{fmt}', valid formats: {valid_formats}")

    # Check device
    valid_devices = {"auto", "cuda", "mps", "cpu"}
    if config.labeling.device not in valid_devices:
        warnings.append(f"Unknown device '{config.labeling.device}', valid: {valid_devices}")

    # Check threshold
    if not 0.0 <= config.labeling.threshold <= 1.0:
        warnings.append(f"Threshold should be between 0 and 1, got {config.labeling.threshold}")

    # Check classes have prompts
    for cls in config.classes:
        if not cls.prompt:
            warnings.append(f"Class '{cls.name}' has no prompt, using class name")

    return warnings
