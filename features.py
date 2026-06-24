from pathlib import Path
import argparse
import os


DEFAULT_IMAGE = Path("processed_dataset") / "circle" / "circle_6.png"


def get_test_image_path(image_path=DEFAULT_IMAGE):
    """Return a valid image path for quick testing."""
    path = Path(image_path)

    if not path.is_absolute():
        path = Path(__file__).resolve().parent / path

    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    return path


def open_test_image(image_path=DEFAULT_IMAGE):
    """Open an image with the system image viewer."""
    path = get_test_image_path(image_path)
    os.startfile(path)
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Open a shape image for testing.")
    parser.add_argument(
        "image",
        nargs="?",
        default=DEFAULT_IMAGE,
        help=f"Image path to open. Default: {DEFAULT_IMAGE}",
    )
    args = parser.parse_args()

    opened_path = open_test_image(args.image)
    print(f"Opened image: {opened_path}")

