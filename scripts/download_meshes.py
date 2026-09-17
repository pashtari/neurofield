#!/usr/bin/env python
"""Download meshes from the Stanford 3D Scanning Repository.

The 5 meshes used by FINER (CVPR 2024) and BACON (CVPR 2022), fetched from the
official source:

    http://graphics.stanford.edu/data/3Dscanrep/

Some releases are not in canonical pose, so their vertices are rotated to
Y-up and front-facing. ``OccupancyCoordinateDataset`` then centers and scales
each mesh (FINER's convention) and voxelizes it with trimesh.

Available meshes: armadillo, dragon, happy_buddha, lucy, thai_statue.

Usage:
    python scripts/download_meshes.py
    python scripts/download_meshes.py --meshes armadillo thai_statue
"""

import argparse
import gzip
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STANFORD = "http://graphics.stanford.edu"

# ``member`` selects the mesh inside a tar archive; ``rotation`` is applied to
# the vertices, and is omitted for meshes that are already in canonical pose.
MESHES = {
    "armadillo": {
        "url": f"{STANFORD}/pub/3Dscanrep/armadillo/Armadillo.ply.gz",
        "rotation": [[1, 0, 0], [0, 1, 0], [0, 0, -1]],
    },
    "dragon": {
        "url": f"{STANFORD}/pub/3Dscanrep/dragon/dragon_recon.tar.gz",
        "member": "dragon_vrip.ply",
    },
    "happy_buddha": {
        "url": f"{STANFORD}/pub/3Dscanrep/happy/happy_recon.tar.gz",
        "member": "happy_vrip.ply",
    },
    "lucy": {
        "url": f"{STANFORD}/data/3Dscanrep/lucy.tar.gz",
        "member": "lucy.ply",
        "rotation": [[1, 0, 0], [0, 0, 1], [0, 1, 0]],
    },
    "thai_statue": {
        "url": f"{STANFORD}/data/3Dscanrep/xyzrgb/xyzrgb_statuette.ply.gz",
    },
}


def report(count: int, block_size: int, total: int) -> None:
    """Print progress every 10 MB (:func:`urllib.request.urlretrieve` callback)."""
    downloaded = count * block_size
    if downloaded % (10 * 2**20) < block_size:
        total_mb = f"{total / 1e6:.0f}" if total > 0 else "?"
        print(f"\r  {downloaded / 1e6:.0f} / {total_mb} MB", end="", flush=True)


def extract(archive: Path, member: str | None, dest: Path) -> None:
    """Write the mesh in ``archive`` to ``dest``, unwrapping gzip or tar."""
    if member is None:
        with gzip.open(archive, "rb") as source, open(dest, "wb") as target:
            shutil.copyfileobj(source, target)
        return

    with tarfile.open(archive, "r:gz") as tar:
        names = [name for name in tar.getnames() if name.endswith(member)]
        if not names:
            raise FileNotFoundError(f"no {member} in {archive.name}")
        with tar.extractfile(names[0]) as source, open(dest, "wb") as target:
            shutil.copyfileobj(source, target)


def rotate(path: Path, rotation: list[list[int]]) -> None:
    """Rotate a mesh in place into canonical pose."""
    import numpy as np
    import trimesh

    transform = np.eye(4)
    transform[:3, :3] = np.array(rotation, dtype=np.float64)

    mesh = trimesh.load(path, force="mesh")
    mesh.apply_transform(transform)
    if np.linalg.det(transform[:3, :3]) < 0:  # a reflection flips face winding
        mesh.invert()
    mesh.export(path)


def download_mesh(name: str, info: dict, output_dir: Path, force: bool) -> None:
    dest = output_dir / f"{name}.ply"
    if dest.exists() and not force:
        print(f"  {dest.name} exists, skipping")
        return

    print(f"  [{name}] {info['url']}")
    suffix = ".tar.gz" if "member" in info else ".ply.gz"
    with tempfile.NamedTemporaryFile(suffix=suffix) as archive:
        urllib.request.urlretrieve(info["url"], archive.name, reporthook=report)
        print()
        extract(Path(archive.name), info.get("member"), dest)

    if "rotation" in info:
        rotate(dest, info["rotation"])
    print(f"  {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "occupancy")
    parser.add_argument(
        "--meshes",
        nargs="+",
        default=list(MESHES),
        choices=list(MESHES),
        metavar="MESH",
        help="Subset to download (default: all)",
    )
    parser.add_argument("--force", action="store_true", help="Re-download files")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name in args.meshes:
        download_mesh(name, MESHES[name], args.output_dir, args.force)

    print(f"Done. Meshes in {args.output_dir}")


if __name__ == "__main__":
    main()
