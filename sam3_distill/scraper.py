"""
Image scraping module using bing-image-downloader.
"""

import hashlib
import shutil
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from .config import Config


class ImageScraper:
    """
    Scrapes images from Bing for each class defined in the config.

    Uses multiple search queries per class to get diverse images.
    """

    # Supported image extensions
    VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}

    # Minimum image dimensions (filter out tiny images)
    MIN_WIDTH = 100
    MIN_HEIGHT = 100

    def __init__(self, config: Config):
        """
        Initialize the scraper.

        Args:
            config: Pipeline configuration.
        """
        self.config = config
        self.output_dir = config.raw_dir
        self.images_per_class = config.scraping.images_per_class
        self.timeout = config.scraping.timeout

    def scrape(self, class_name: str, queries: list[str], num_images: int) -> int:
        """
        Scrape images for a single class using multiple queries.

        Args:
            class_name: Name of the class (used for output directory).
            queries: List of search queries to use.
            num_images: Target number of images to download.

        Returns:
            Number of valid images downloaded.
        """
        from bing_image_downloader import downloader

        class_dir = self.output_dir / class_name
        class_dir.mkdir(parents=True, exist_ok=True)

        # Calculate images per query
        images_per_query = max(1, num_images // len(queries))
        extra_images = num_images % len(queries)

        print(f"\n{'='*60}")
        print(f"Scraping images for class: {class_name}")
        print(f"Target: {num_images} images using {len(queries)} queries")
        print(f"{'='*60}")

        temp_dir = self.output_dir / "_temp"
        all_downloaded = []

        for i, query in enumerate(queries):
            # Add extra images to first queries
            target = images_per_query + (1 if i < extra_images else 0)

            print(f"\n[{i+1}/{len(queries)}] Query: '{query}' (target: {target} images)")

            try:
                # Download to temp directory
                query_dir = temp_dir / query.replace(" ", "_")
                query_dir.mkdir(parents=True, exist_ok=True)

                downloader.download(
                    query,
                    limit=target + 20,  # Download extra to account for filtering
                    output_dir=str(temp_dir),
                    adult_filter_off=False,
                    force_replace=False,
                    timeout=self.timeout,
                    verbose=False
                )

                # Find downloaded images (bing-image-downloader creates a subfolder)
                downloaded_dir = temp_dir / query
                if downloaded_dir.exists():
                    for img_path in downloaded_dir.iterdir():
                        if img_path.suffix.lower() in self.VALID_EXTENSIONS:
                            all_downloaded.append(img_path)

            except Exception as e:
                print(f"  Warning: Error downloading '{query}': {e}")
                continue

        # Deduplicate and validate images
        print(f"\nProcessing {len(all_downloaded)} downloaded images...")
        valid_count = self._process_and_deduplicate(all_downloaded, class_dir, num_images)

        # Cleanup temp directory
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

        print(f"\nClass '{class_name}': {valid_count} valid images saved")
        return valid_count

    def _process_and_deduplicate(
        self,
        image_paths: list[Path],
        output_dir: Path,
        max_images: int
    ) -> int:
        """
        Process downloaded images: validate, deduplicate, and rename.

        Args:
            image_paths: List of downloaded image paths.
            output_dir: Directory to save processed images.
            max_images: Maximum number of images to keep.

        Returns:
            Number of valid images saved.
        """
        seen_hashes = set()

        # Load existing image hashes to avoid duplicates across runs
        for existing in output_dir.iterdir():
            if existing.suffix.lower() in self.VALID_EXTENSIONS:
                try:
                    h = self._image_hash(existing)
                    seen_hashes.add(h)
                except Exception:
                    pass

        valid_count = len(seen_hashes)
        next_idx = valid_count

        for img_path in tqdm(image_paths, desc="Validating images", unit="img"):
            if valid_count >= max_images:
                break

            try:
                # Validate image
                if not self._is_valid_image(img_path):
                    continue

                # Check for duplicates
                img_hash = self._image_hash(img_path)
                if img_hash in seen_hashes:
                    continue

                seen_hashes.add(img_hash)

                # Copy to output with standardized name
                ext = img_path.suffix.lower()
                if ext == ".jpeg":
                    ext = ".jpg"
                new_name = f"{next_idx:06d}{ext}"
                new_path = output_dir / new_name

                shutil.copy2(img_path, new_path)
                valid_count += 1
                next_idx += 1

            except Exception as e:
                # Skip problematic images
                continue

        return valid_count

    def _is_valid_image(self, path: Path) -> bool:
        """
        Check if an image file is valid and meets minimum requirements.

        Args:
            path: Path to the image file.

        Returns:
            True if the image is valid, False otherwise.
        """
        try:
            with Image.open(path) as img:
                # Check dimensions
                width, height = img.size
                if width < self.MIN_WIDTH or height < self.MIN_HEIGHT:
                    return False

                # Check if image is readable
                img.verify()

            return True

        except Exception:
            return False

    def _image_hash(self, path: Path) -> str:
        """
        Compute a hash of an image for deduplication.

        Uses a perceptual hash based on resized image content.

        Args:
            path: Path to the image file.

        Returns:
            Hash string for the image.
        """
        try:
            with Image.open(path) as img:
                # Resize to small thumbnail for consistent hashing
                img = img.convert("RGB").resize((8, 8), Image.Resampling.LANCZOS)
                pixels = list(img.getdata())
                return hashlib.md5(str(pixels).encode()).hexdigest()
        except Exception:
            # Fallback to file hash
            return hashlib.md5(path.read_bytes()).hexdigest()

    def scrape_all(self) -> dict[str, int]:
        """
        Scrape images for all classes defined in the config.

        Returns:
            Dictionary mapping class names to number of images downloaded.
        """
        print("\n" + "="*60)
        print("SAM3 Detection Distillation - Image Scraper")
        print("="*60)
        print(f"Output directory: {self.output_dir}")
        print(f"Classes: {len(self.config.classes)}")
        print(f"Target images per class: {self.images_per_class}")

        results = {}
        total_downloaded = 0

        for cls in self.config.classes:
            count = self.scrape(
                class_name=cls.name,
                queries=cls.search_queries,
                num_images=self.images_per_class
            )
            results[cls.name] = count
            total_downloaded += count

        # Print summary
        print("\n" + "="*60)
        print("Scraping Complete!")
        print("="*60)
        for class_name, count in results.items():
            status = "OK" if count >= self.images_per_class * 0.8 else "LOW"
            print(f"  {class_name}: {count} images [{status}]")
        print(f"\nTotal: {total_downloaded} images")

        return results

    def get_stats(self) -> dict[str, int]:
        """
        Get statistics about currently downloaded images.

        Returns:
            Dictionary mapping class names to image counts.
        """
        stats = {}
        for cls in self.config.classes:
            class_dir = self.output_dir / cls.name
            if class_dir.exists():
                count = sum(
                    1 for f in class_dir.iterdir()
                    if f.suffix.lower() in self.VALID_EXTENSIONS
                )
                stats[cls.name] = count
            else:
                stats[cls.name] = 0
        return stats
