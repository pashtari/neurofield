"""Download the NeRF synthetic (Blender) scenes.

The 8 scenes from NeRF (ECCV 2020), each with 100 train / 100 val / 200 test
800x800 RGBA views and transforms_{split}.json camera poses. The official
release lives on Google Drive, so files are fetched from a pinned Hugging Face
mirror instead:

    https://huggingface.co/datasets/pablovela5620/nerf-synthetic-mirror

Test-view depth maps are skipped (BlenderDataset does not use them) unless
--depth is given. Existing files are kept, so an interrupted download resumes.

Available scenes: chair, drums, ficus, hotdog, lego, materials, mic, ship.

Usage:
    python scripts/download_blender.py
    python scripts/download_blender.py --scenes lego ship
"""

import argparse
import json
import re
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = "pablovela5620/nerf-synthetic-mirror"
REVISION = "23e7eaa65ff836405329c7b6b4a34afaeb3a70e2"
SCENES = ["chair", "drums", "ficus", "hotdog", "lego", "materials", "mic", "ship"]
API = f"https://huggingface.co/api/datasets/{REPO}"
FILES = f"https://huggingface.co/datasets/{REPO}"
ROOT = Path(__file__).resolve().parents[1]


def list_files(scene: str) -> list[str]:
    """Paths of all files under ``scene`` in the mirror, following pagination."""
    url = f"{API}/tree/{REVISION}/{scene}?recursive=true"
    paths = []
    while url:
        with urllib.request.urlopen(url) as response:
            items = json.load(response)
            link = response.headers.get("Link", "")
        paths += [item["path"] for item in items if item["type"] == "file"]
        next_page = re.search(r'<([^>]+)>;\s*rel="next"', link)
        url = next_page and next_page.group(1)
    return paths


def download_scene(
    scene: str, output_dir: Path, depth: bool, force: bool, workers: int
) -> None:
    paths = [p for p in list_files(scene) if depth or "_depth_" not in p]
    todo = [p for p in paths if force or not (output_dir / p).exists()]
    print(f"  [{scene}] {len(todo)} / {len(paths)} files to download")

    def fetch(path: str) -> None:
        dest = output_dir / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        urllib.request.urlretrieve(f"{FILES}/resolve/{REVISION}/{path}", tmp)
        tmp.replace(dest)

    with ThreadPoolExecutor(workers) as pool:
        for done, _ in enumerate(pool.map(fetch, todo), 1):
            print(f"\r  {done} / {len(todo)}", end="", flush=True)
    if todo:
        print()


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/nerf/blender")
    parser.add_argument(
        "--scenes",
        nargs="+",
        default=SCENES,
        choices=SCENES,
        metavar="SCENE",
        help="Subset to download (default: all)",
    )
    parser.add_argument("--depth", action="store_true", help="Also get depth maps")
    parser.add_argument("--force", action="store_true", help="Re-download files")
    parser.add_argument("--workers", type=int, default=16, help="Parallel downloads")
    args = parser.parse_args()

    for scene in args.scenes:
        download_scene(scene, args.output_dir, args.depth, args.force, args.workers)

    print(f"Done. Scenes in {args.output_dir}")


if __name__ == "__main__":
    main()
