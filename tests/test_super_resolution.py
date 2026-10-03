"""The super-resolution benchmark's forward model, minibatches and FUTON's rank."""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import to_pil_image

import neurofield as nf

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


@pytest.fixture(scope="module")
def task():
    """The benchmark script as a module; it imports train_common from its folder."""
    sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(
        "train_super_resolution", SCRIPTS / "train_super_resolution.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def image_pair(task, tmp_path):
    """A random 32 x 48 image and its 4x downsampling, laid out as DIV2K is."""
    torch.manual_seed(0)
    image = torch.randint(256, (3, 32, 48), dtype=torch.uint8)
    device = torch.device("cpu")
    kernels = [task.resize_matrix(32, 8, device), task.resize_matrix(48, 12, device)]
    low = task.downsample(image.movedim(0, -1).float(), kernels)
    low = low.round().clamp(0, 255).to(torch.uint8).movedim(-1, 0)
    (tmp_path / "DIV2K_valid_HR").mkdir()
    (tmp_path / "DIV2K_valid_LR_bicubic" / "X4").mkdir(parents=True)
    path = tmp_path / "DIV2K_valid_HR" / "0801.png"
    to_pil_image(image).save(path)
    to_pil_image(low).save(task.low_resolution(path, 4))
    return path, image, kernels


def test_resize_matrix_is_the_antialiased_bicubic_operator(task):
    torch.manual_seed(0)
    image = torch.rand(1, 3, 32, 48)
    expected = F.interpolate(image, size=(8, 12), mode="bicubic", antialias=True)
    device = torch.device("cpu")
    kernels = [task.resize_matrix(32, 8, device), task.resize_matrix(48, 12, device)]
    downsampled = task.downsample(image[0].movedim(0, -1), kernels).movedim(-1, 0)
    torch.testing.assert_close(downsampled, expected[0], atol=1e-5, rtol=1e-5)
    assert int((kernels[0] != 0).sum(1).max()) == 16  # taps per LR pixel at 4x


def test_blocks_downsample_to_the_low_resolution_pixels(task, image_pair):
    path, image, kernels = image_pair
    dataset = task.SuperResolutionDataset(task.low_resolution(path, 4), kernels, 0.5, 2)
    torch.manual_seed(0)
    batch = dataset[0]
    # A field that returns the HR pixels under the coordinates it is given.
    rows = ((batch["input"][..., 0] + 1) / 2 * 31).round().long()
    columns = ((batch["input"][..., 1] + 1) / 2 * 47).round().long()
    rendered = dataset.preprocess(image)[rows, columns]  # (B, L, L, C) in [-1, 1]
    fitted = task.downsample(rendered, (batch["weight_y"], batch["weight_x"]))
    assert batch["input"].shape[1:] == (dataset.length, dataset.length, 2)
    assert batch["target"].shape[1:] == (2, 2, 3)
    torch.testing.assert_close(fitted, batch["target"], atol=2 / 255, rtol=0)


def test_futon_rank_is_the_largest_within_the_budget(task):
    for h, w, budget in ((339, 510, 500_000), (510, 510, 500_000), (64, 96, 20_000)):
        rank = task.futon_rank(h, w, budget)

        def count(rank: int) -> int:
            model = nf.FUTON(
                in_features=2,
                out_features=3,
                basis=("lanczos", {"num_components": (h, w), "radius": 3}),
                combiner=("cp", {"rank": rank}),
                decoder=("mlp", {"hidden_layers": 1}),
            )
            return nf.count_parameters(model)

        assert count(rank) <= budget < count(rank + 1)
