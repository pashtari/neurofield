#!/usr/bin/env python
"""Download meshes from the Stanford 3D Scanning Repository.

Downloads the same meshes used in FINER (CVPR 2024) and BACON (CVPR 2022)
directly from the official Stanford source:

    http://graphics.stanford.edu/data/3Dscanrep/

After downloading, neurofield's PointCloudDataset applies the same
normalization as FINER/BACON (center on vertex mean, scale to 90% of
[-0.5, 0.5]) and voxelizes with trimesh. Results are cached as .pt files.

Available meshes: armadillo, dragon, happy_buddha, lucy, thai_statue.

Usage:
    python scripts/download_meshes.py
    python scripts/download_meshes.py --meshes armadillo thai_statue
"""

import argparse
import gzip
import os
import shutil
import tarfile
import tempfile
import urllib.request

# Official Stanford 3D Scanning Repository URLs
# "orientation" is a 3x3 matrix applied to vertices so that all meshes end up
# in canonical pose: Y-up, front facing the camera (+Z toward viewer).
MESHES = {
    "armadillo": {
        "url": "http://graphics.stanford.edu/pub/3Dscanrep/armadillo/Armadillo.ply.gz",
        "format": "ply.gz",
        "orientation": [[1, 0, 0], [0, 1, 0], [0, 0, -1]],
    },
    "dragon": {
        "url": "http://graphics.stanford.edu/pub/3Dscanrep/dragon/dragon_recon.tar.gz",
        "format": "tar.gz",
        "ply_glob": "dragon_vrip.ply",
    },
    "happy_buddha": {
        "url": "http://graphics.stanford.edu/pub/3Dscanrep/happy/happy_recon.tar.gz",
        "format": "tar.gz",
        "ply_glob": "happy_vrip.ply",
        # "orientation": [[0, 0, 1], [0, 1, 0], [-1, 0, 0]],
    },
    "lucy": {
        "url": "http://graphics.stanford.edu/data/3Dscanrep/lucy.tar.gz",
        "format": "tar.gz",
        "ply_glob": "lucy.ply",
        "orientation": [[1, 0, 0], [0, 0, 1], [0, 1, 0]],
    },
    "thai_statue": {
        "url": "http://graphics.stanford.edu/data/3Dscanrep/xyzrgb/xyzrgb_statuette.ply.gz",
        "format": "ply.gz",
    },
}


def _reporthook(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100, downloaded * 100 // total_size)
        print(
            f"\r  {downloaded / 1e6:.1f} / {total_size / 1e6:.1f} MB ({pct}%)",
            end="",
            flush=True,
        )
    else:
        print(f"\r  {downloaded / 1e6:.1f} MB", end="", flush=True)


def download_ply_gz(url: str, dest: str) -> None:
    """Download a gzipped PLY file and decompress it."""
    with tempfile.NamedTemporaryFile(suffix=".ply.gz", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        urllib.request.urlretrieve(url, tmp_path, reporthook=_reporthook)
        print()
        with gzip.open(tmp_path, "rb") as f_in, open(dest, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
    finally:
        os.unlink(tmp_path)


def download_tar_gz(url: str, ply_glob: str, dest: str) -> None:
    """Download a tar.gz archive and extract the target PLY file."""
    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        urllib.request.urlretrieve(url, tmp_path, reporthook=_reporthook)
        print()

        with tarfile.open(tmp_path, "r:gz") as tar:
            # Find the target PLY member
            ply_member = None
            for member in tar.getmembers():
                if member.name.endswith(ply_glob):
                    ply_member = member
                    break

            if ply_member is None:
                # Fallback: find any .ply file
                for member in tar.getmembers():
                    if member.name.endswith(".ply"):
                        ply_member = member
                        break

            if ply_member is None:
                raise FileNotFoundError(
                    f"No PLY file matching '{ply_glob}' found in archive. " f"Contents: {[m.name for m in tar.getmembers()[:20]]}"
                )

            print(f"  Extracting {ply_member.name} ...")
            with tar.extractfile(ply_member) as f_in, open(dest, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
    finally:
        os.unlink(tmp_path)


def orient_mesh(dest: str, orientation: list) -> None:
    """Apply orientation transform to a PLY mesh for canonical rendering."""
    import numpy as np
    import trimesh

    mesh = trimesh.load(dest, force="mesh")

    R = np.array(orientation, dtype=np.float64)
    transform = np.eye(4)
    transform[:3, :3] = R
    mesh.apply_transform(transform)

    # Fix face winding if transform includes a reflection
    if np.linalg.det(R) < 0:
        mesh.invert()

    mesh.export(dest)
    print(f"  Applied orientation transform to {dest}")


def download_mesh(name: str, info: dict, output_dir: str, force: bool = False) -> None:
    dest = os.path.join(output_dir, f"{name}.ply")
    if os.path.exists(dest) and not force:
        print(f"  {dest} already exists — skipping.")
        return

    url = info["url"]
    fmt = info["format"]
    print(f"  [{name}] {url}")

    if fmt == "ply.gz":
        download_ply_gz(url, dest)
    elif fmt == "tar.gz":
        download_tar_gz(url, info.get("ply_glob", ".ply"), dest)
    else:
        raise ValueError(f"Unknown format: {fmt}")

    size_mb = os.path.getsize(dest) / 1e6
    print(f"  Saved {dest} ({size_mb:.1f} MB)")

    # Apply canonical orientation so PointCloudDataset renders correctly
    orientation = info.get("orientation")
    if orientation:
        orient_mesh(dest, orientation)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save .ply files (default: data/occupancy)",
    )
    parser.add_argument(
        "--meshes",
        nargs="*",
        default=None,
        help=f"Subset to download (default: all). Choices: {', '.join(MESHES)}",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download and reorient even if files already exist",
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)

    output_dir = args.output_dir or os.path.join(project_dir, "data", "occupancy")
    os.makedirs(output_dir, exist_ok=True)

    meshes = args.meshes or list(MESHES.keys())
    for name in meshes:
        if name not in MESHES:
            print(f"  Unknown mesh '{name}', skipping. Available: {list(MESHES)}")
            continue
        download_mesh(name, MESHES[name], output_dir, force=args.force)

    print("\nDone! Files saved to:", output_dir)
    print("PointCloudDataset will normalize using FINER/BACON convention and cache")
    print("voxelized grids on first load.")


if __name__ == "__main__":
    main()
