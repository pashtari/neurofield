#!/usr/bin/env python
"""Download the Kodak Lossless True Color Image Suite.

The 24 images (768x512, kodim01.png ... kodim24.png) used by SIREN, FINER,
and most image-representation benchmarks, fetched from Rich Franzen's mirror:

    https://r0k.us/graphics/kodak/

Usage:
    python scripts/download_kodak.py
    python scripts/download_kodak.py --images 1 17 23
"""

import argparse
import urllib.request
from pathlib import Path

URL = "https://r0k.us/graphics/kodak/kodak/kodim{:02d}.png"
ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "Kodak")
    parser.add_argument(
        "--images",
        type=int,
        nargs="+",
        default=range(1, 25),
        help="Image numbers in 1-24 (default: all)",
    )
    parser.add_argument("--force", action="store_true", help="Re-download files")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for i in args.images:
        dest = args.output_dir / f"kodim{i:02d}.png"
        if dest.exists() and not args.force:
            print(f"  {dest.name} exists, skipping")
            continue
        urllib.request.urlretrieve(URL.format(i), dest)
        print(f"  {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")

    print(f"Done. Images in {args.output_dir}")


if __name__ == "__main__":
    main()
