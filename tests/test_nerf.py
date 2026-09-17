"""Construction/protocol smoke tests for :mod:`neurofield.nerf`.

Every row the NeRF notebook trains must at least *build* and run a forward
pass, and the marching/dataset constants the protocol is pinned to must not
drift. These are the checks that turn a construction-time ``TypeError`` into a
red test instead of a notebook that dies mid-run.
"""

import math
from pathlib import Path

import pytest
import torch

import neurofield as nf

DATA_ROOT = Path(__file__).resolve().parents[1] / "data/nerf/blender/lego"

GEOMETRY_FEATURES = 15
COLOR_NET = ("mlp", {"hidden_features": 64, "hidden_layers": 2})
MLP_CONFIG = {"hidden_features": 182, "hidden_layers": 3}

# Mirrors CONFIGS in notebooks/nerf.ipynb.
DENSITY_NETS = {
    # narrower: the 63-dim positional encoding widens layer 1
    "PE-MLP": (nf.PEMLP, {**MLP_CONFIG, "hidden_features": 168, "num_frequencies": 10}),
    "SIREN": (nf.SIREN, {**MLP_CONFIG, "omega": 30.0}),
    "WIRE": (nf.WIRE, {**MLP_CONFIG, "omega": 20.0, "scale": 10.0}),
    "Gauss": (nf.Gauss, {**MLP_CONFIG, "scale": 30.0}),
    "FINER": (nf.FINER, {**MLP_CONFIG, "omega": 30.0}),
    "Instant-NGP": (
        nf.InstantNGP,
        {
            "num_levels": 16,
            "log2_hashmap_size": 11,
            "max_resolution": 2048,
            "hidden_features": 64,
            "hidden_layers": 1,
        },
    ),
    "TensoRF-CP": (
        nf.TensoRF,
        {
            "resolution": 400,
            "rank": 52,
            "mode": "cp",
            "hidden_features": 64,
            "hidden_layers": 2,
        },
    ),
    "TensoRF-VM": (
        nf.TensoRF,
        {
            "resolution": 38,
            "rank": 14,
            "mode": "vm",
            "hidden_features": 64,
            "hidden_layers": 2,
        },
    ),
    "GA-Planes": (
        nf.GAPlanes,
        {
            "features": 12,
            "resolution": (64, 16),
            "hidden_features": 64,
            "hidden_layers": 2,
        },
    ),
    "FUTON-cosine": (
        nf.FUTON,
        {
            "basis": ("cosine", {"num_components": 130}),
            "combiner": ("cp", {"rank": 130}),
            "decoder": ("mlp", {"hidden_layers": 1}),
        },
    ),
    "FUTON-sinc": (
        nf.FUTON,
        {
            "basis": ("sinc", {"num_components": 130}),
            "combiner": ("cp", {"rank": 130}),
            "decoder": ("mlp", {"hidden_layers": 1}),
        },
    ),
    "FUTON-lanczos": (
        nf.FUTON,
        {
            "basis": ("lanczos", {"num_components": 130, "radius": 3}),
            "combiner": ("cp", {"rank": 130}),
            "decoder": ("mlp", {"hidden_layers": 1}),
        },
    ),
    "CP-tent": (  # the basis ablation: same budget, tent pulses
        nf.FUTON,
        {
            "basis": ("triangle", {"num_components": 130, "normalize": False}),
            "combiner": ("cp", {"rank": 130}),
            "decoder": ("mlp", {"hidden_layers": 1}),
        },
    ),
}


def _field(density_net, **kwargs):
    if isinstance(density_net, tuple):  # the notebook supplies this per row
        cls, kw = density_net
        density_net = (cls, {"output_activation": torch.nn.Identity(), **kw})
    kwargs.setdefault("color_net", COLOR_NET)
    kwargs.setdefault("geometry_features", GEOMETRY_FEATURES)
    return nf.nerf.RadianceField(density_net=density_net, **kwargs)


@pytest.mark.parametrize("name", sorted(DENSITY_NETS))
def test_backbone_builds_and_runs(name):
    """Every notebook backbone constructs and returns finite (rgb, sigma)."""
    torch.manual_seed(0)
    field = _field(DENSITY_NETS[name])

    x = torch.rand(8, 3) * 2.0 - 1.0
    d = torch.nn.functional.normalize(torch.randn(8, 3), dim=-1)
    rgb, sigma = field(x, d)

    assert rgb.shape == (8, 3) and sigma.shape == (8,)
    assert torch.isfinite(rgb).all() and torch.isfinite(sigma).all()
    assert (rgb >= 0).all() and (rgb <= 1).all() and (sigma >= 0).all()


@pytest.mark.parametrize("name", sorted(DENSITY_NETS))
def test_backbone_parameter_matched(name):
    """The backbone comparison is only fair if the rows stay parameter-matched."""
    torch.manual_seed(0)
    num_params = nf.count_parameters(_field(DENSITY_NETS[name]))
    assert 74_000 <= num_params <= 79_000, f"{name}: {num_params:,} params"


def test_finer_reproduction_parameter_count():
    """FINER end to end is exactly network_finer.py's 201,842 parameters."""
    torch.manual_seed(0)
    field = _field(
        (nf.FINER, {**MLP_CONFIG, "omega": 30.0}),
        color_net=(
            nf.FINER,
            {**MLP_CONFIG, "omega": 30.0, "output_activation": torch.nn.Identity()},
        ),
        geometry_features=182,
        direction_encoding="identity",
    )
    assert nf.count_parameters(field) == 201_842


def test_density_net_logits_are_unbounded():
    """The caller supplies Identity; a tanh default would cap density at e."""
    field = _field((nf.SIREN, {**MLP_CONFIG, "output_activation": torch.nn.Identity()}))
    with torch.no_grad():
        for p in field.density_net.parameters():
            p.mul_(50)
        logits = field.density_net(torch.rand(256, 3) * 2 - 1)
    assert logits.abs().max() > 1.0, "density logits must not be tanh-bounded"


def test_renderer_marching_constants():
    """Ray marching is pinned to torch-ngp's --max_steps 1024 over the unit box."""
    renderer = nf.nerf.create_renderer(
        backend="pytorch", aabb=(-1.0, -1.0, -1.0, 1.0, 1.0, 1.0), near=0.2, far=10.0
    )
    assert renderer.num_marching_steps == 1024
    assert renderer.render_step_size == 2.0 * math.sqrt(3.0) / 1024


@pytest.mark.skipif(not DATA_ROOT.exists(), reason="Blender lego scene not present")
def test_blender_train_split_and_radius():
    """FINER's protocol: --trainskip 4 --downscale 4 --scale 0.8 -> 25 views."""
    dataset = nf.nerf.BlenderDataset(DATA_ROOT, split="train", downsample=4, skip=4)
    assert len(dataset) == 25
    assert dataset.images.shape[1:3] == (200, 200)
    assert dataset.radius == pytest.approx(3.224903, abs=1e-6)


@pytest.mark.skipif(not DATA_ROOT.exists(), reason="Blender lego scene not present")
def test_train_loop_runs_and_loss_is_finite():
    """A few optimizer steps must run end to end and produce a finite loss."""
    torch.manual_seed(0)
    dataset = nf.nerf.BlenderDataset(DATA_ROOT, split="train", downsample=4, skip=4)
    field = _field(DENSITY_NETS["FUTON-lanczos"], aabb=dataset.aabb)
    renderer = nf.nerf.create_renderer(
        backend="pytorch", aabb=dataset.aabb, near=dataset.near, far=dataset.far
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"

    result = nf.nerf.train(
        field,
        renderer,
        dataset,
        device=device,
        num_steps=5,
        num_rays=256,
        log_interval=5,
        eval_interval=0,
    )

    assert result["config"]["skipped_steps"] == 0
    assert math.isfinite(result["history"][-1]["loss"])
