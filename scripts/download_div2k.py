"""Download the DIV2K validation set for 4x super-resolution.

The 100 validation images of DIV2K (Agustsson and Timofte, CVPRW 2017), 2040
pixels on the long side, and their 4x bicubic downsamplings (MATLAB's
imresize), from the official site:

    https://data.vision.ee.ethz.ch/cvl/DIV2K/

The archives unpack in the release's layout, as DIV2K_valid_HR/0801.png and
DIV2K_valid_LR_bicubic/X4/0801x4.png; notebooks/image_super_resolution.ipynb
reads image 0882 from there. An archive whose images all exist is skipped.

Usage:
    python scripts/download_div2k.py
"""

import argparse
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

URL = "https://data.vision.ee.ethz.ch/cvl/DIV2K/{}.zip"
ARCHIVES = {  # archive: where each of its images unpacks
    "DIV2K_valid_HR": "DIV2K_valid_HR/{:04d}.png",
    "DIV2K_valid_LR_bicubic_X4": "DIV2K_valid_LR_bicubic/X4/{:04d}x4.png",
}
IMAGES = range(801, 901)
ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "DIV2K")
    parser.add_argument("--force", action="store_true", help="Re-download files")
    args = parser.parse_args()

    for name, image in ARCHIVES.items():
        paths = [args.output_dir / image.format(i) for i in IMAGES]
        if all(path.exists() for path in paths) and not args.force:
            print(f"  {name} exists, skipping")
            continue
        print(f"  {name}", flush=True)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        # Beside the images rather than in /tmp, which is small on some clusters.
        with tempfile.TemporaryFile(dir=args.output_dir) as archive:
            with urllib.request.urlopen(URL.format(name)) as response:
                shutil.copyfileobj(response, archive)
            with zipfile.ZipFile(archive) as files:
                files.extractall(args.output_dir)

    print(f"Done. Images in {args.output_dir}")


if __name__ == "__main__":
    main()
