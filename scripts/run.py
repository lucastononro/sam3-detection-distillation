#!/usr/bin/env python3
"""
CLI entry point for the SAM3 Detection Distillation pipeline.

Usage:
    # Run full pipeline
    python scripts/run.py --config configs/my_project.yaml

    # Skip specific stages
    python scripts/run.py --config configs/my_project.yaml --skip-scrape
    python scripts/run.py --config configs/my_project.yaml --skip-label
    python scripts/run.py --config configs/my_project.yaml --skip-train

    # Run only specific stages
    python scripts/run.py --config configs/my_project.yaml --only-scrape
    python scripts/run.py --config configs/my_project.yaml --only-label
    python scripts/run.py --config configs/my_project.yaml --only-train
    python scripts/run.py --config configs/my_project.yaml --only-eval
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from sam3_distill.pipeline import Pipeline, run_pipeline


def main():
    parser = argparse.ArgumentParser(
        description="SAM3 Detection Distillation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline
  python scripts/run.py --config configs/ppe_detection.yaml

  # Skip scraping (use existing images)
  python scripts/run.py --config configs/ppe_detection.yaml --skip-scrape

  # Only run labeling
  python scripts/run.py --config configs/ppe_detection.yaml --only-label

  # Skip training and evaluation (just prepare dataset)
  python scripts/run.py --config configs/ppe_detection.yaml --skip-train --skip-eval
        """
    )

    # Required arguments
    parser.add_argument(
        "--config", "-c",
        type=str,
        required=True,
        help="Path to the YAML configuration file"
    )

    # Skip flags
    parser.add_argument(
        "--skip-scrape",
        action="store_true",
        help="Skip the image scraping stage"
    )
    parser.add_argument(
        "--skip-label",
        action="store_true",
        help="Skip the SAM3 labeling stage"
    )
    parser.add_argument(
        "--skip-export",
        action="store_true",
        help="Skip the dataset export stage"
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip the model training stage"
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip the model evaluation stage"
    )

    # Only flags (run single stage)
    parser.add_argument(
        "--only-scrape",
        action="store_true",
        help="Only run the image scraping stage"
    )
    parser.add_argument(
        "--only-label",
        action="store_true",
        help="Only run the SAM3 labeling stage"
    )
    parser.add_argument(
        "--only-export",
        action="store_true",
        help="Only run the dataset export stage"
    )
    parser.add_argument(
        "--only-train",
        action="store_true",
        help="Only run the model training stage"
    )
    parser.add_argument(
        "--only-eval",
        action="store_true",
        help="Only run the model evaluation stage"
    )

    args = parser.parse_args()

    # Validate config path
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)

    # Handle --only flags
    only_flags = {
        "only_scrape": args.only_scrape,
        "only_label": args.only_label,
        "only_export": args.only_export,
        "only_train": args.only_train,
        "only_eval": args.only_eval,
    }

    only_count = sum(only_flags.values())
    if only_count > 1:
        print("Error: Cannot specify multiple --only flags")
        sys.exit(1)

    try:
        pipeline = Pipeline(config_path)

        if only_count == 1:
            # Run single stage
            if args.only_scrape:
                pipeline.scrape()
            elif args.only_label:
                pipeline.label()
            elif args.only_export:
                # Need to load detections first
                detections = pipeline.labeler.label_all()
                pipeline.export(detections)
            elif args.only_train:
                # Need to find existing data.yaml
                yolo_yaml = pipeline.config.labeled_dir / "yolo" / "data.yaml"
                if not yolo_yaml.exists():
                    print(f"Error: No dataset found at {yolo_yaml}")
                    print("Run export stage first.")
                    sys.exit(1)
                pipeline.exported_paths = {"yolo": yolo_yaml}
                pipeline.train()
            elif args.only_eval:
                # Need to find existing models and data
                yolo_yaml = pipeline.config.labeled_dir / "yolo" / "data.yaml"
                yolo_weights = pipeline.config.models_dir / "yolo" / "train" / "weights" / "best.pt"
                rtdetr_weights = pipeline.config.models_dir / "rtdetr" / "train" / "weights" / "best.pt"

                trained = {}
                if yolo_weights.exists():
                    trained["yolo"] = yolo_weights
                if rtdetr_weights.exists():
                    trained["rtdetr"] = rtdetr_weights

                if not trained:
                    print("Error: No trained models found")
                    print("Run training stage first.")
                    sys.exit(1)

                pipeline.trained_models = trained
                pipeline.exported_paths = {"yolo": yolo_yaml}
                pipeline.evaluate()
        else:
            # Run full pipeline with skip flags
            results = pipeline.run(
                skip_scrape=args.skip_scrape,
                skip_label=args.skip_label,
                skip_export=args.skip_export,
                skip_train=args.skip_train,
                skip_eval=args.skip_eval
            )

        print("\nPipeline completed successfully!")

    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
