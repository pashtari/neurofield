"""Tests for the block coordinate descent solvers of FUTON."""

import copy

import numpy as np
import pytest
import torch
import torch.nn.functional as F

import neurofield as nf
from neurofield.models.futon_solvers import LeastSquares, MultiplicativeUpdate

DOUBLE = torch.float64


def signed(z):
    """From [0, 1] to [-1, 1]: the affine output activation of a nonnegative model."""
    return 2 * z - 1


def grid(sizes, out_features, seed=0):
    """A whole grid: random coordinates per axis and a smooth nonnegative signal.

    The signal is a sum of a few random Gaussian bumps scaled into [0, 1] and
    then into [-1, 1], with a quarter of its values at the bottom, since
    zeros in the target are what drives multiplicative updates to the
    boundary.
    """
    generator = torch.Generator().manual_seed(seed)
    axes = [torch.rand(n, dtype=DOUBLE, generator=generator) * 2 - 1 for n in sizes]
    x = torch.stack(torch.meshgrid(*axes, indexing="ij"), -1)
    points = x.reshape(-1, len(sizes))
    centers = torch.rand(5, len(sizes), dtype=DOUBLE, generator=generator) * 2 - 1
    weights = torch.rand(5, out_features, dtype=DOUBLE, generator=generator)
    values = torch.exp(-torch.cdist(points, centers).square() / 0.1) @ weights
    values = values / values.max()
    values[torch.rand(values.shape, generator=generator) < 0.25] = 0
    return {"input": x, "target": signed(values).reshape(*sizes, out_features)}


def sample(batch, fraction, generator):
    """A random share of a grid's points, flattened, as a subsampling dataset draws it."""
    x, y = (value.reshape(-1, value.shape[-1]) for value in batch.values())
    index = torch.randperm(len(x), generator=generator)[: int(fraction * len(x))]
    return {"input": x[index], "target": y[index]}


def futon(
    in_features,
    out_features,
    num_components,
    rank,
    basis="sinc",
    activation=None,
    **options,
):
    """The paper's model, a CP combiner and a linear decoder without bias, in double precision."""
    return nf.FUTON(
        in_features,
        out_features,
        basis=(basis, {"num_components": num_components, **options}),
        combiner=("cp", {"rank": rank}),
        decoder=("linear", {"bias": False}),
        output_activation=activation,
    ).to(DOUBLE)


def mse(model, batch):
    with torch.no_grad():
        return F.mse_loss(model(batch["input"]), batch["target"])


def recorded(solver, batch, sweeps):
    """The batch's error before every block's update and after the last sweep."""
    errors, solve = [], solver.solve

    def recording(*args, **kwargs):
        errors.append(mse(solver.model, batch))
        return solve(*args, **kwargs)

    solver.solve = recording
    for _ in range(sweeps):
        solver.step(batch)
    errors.append(mse(solver.model, batch))
    return torch.stack(errors)


def separable(basis):
    """A model of rank one with chosen factors, and a grid it fits exactly."""
    torch.manual_seed(0)
    truth = futon(
        2, 1, [8, 6], 1, basis, signed if basis == "triangle" else None, normalize=False
    )
    with torch.no_grad():
        for linear, k in zip(truth.combiner.linears, (3, 4)):
            linear.weight.zero_()
            linear.weight[0, k] = 1
        truth.decoder.weight.fill_(1)
    x = grid((40, 50), 1)["input"]
    return truth, {"input": x, "target": truth(x).detach()}


class TestSolver:
    def test_fits_the_papers_model_only(self):
        basis = ("sinc", {"num_components": 8})
        linear = ("linear", {"bias": False})
        for combiner, decoder in [
            (("cp", {"rank": 2}), "linear"),  # a decoder with bias
            (("cp", {"rank": 2}), ("mlp", {"hidden_layers": 1})),
            (("cp", {"rank": 2, "bias": True}), linear),
            (("tr", {"rank": 2}), linear),
        ]:
            with pytest.raises(ValueError):
                LeastSquares(nf.FUTON(2, 1, basis, combiner, decoder))
        with pytest.raises(ValueError):
            MultiplicativeUpdate(futon(2, 1, 8, 2, "triangle", signed), momentum=1.0)

    def test_is_an_optimizer_over_the_blocks(self):
        model = futon(3, 2, [5, 6, 7], 4)
        solver = LeastSquares(model, ridge=1e-3, momentum=0.5)
        assert isinstance(solver, torch.optim.Optimizer)
        params = solver.param_groups[0]["params"]
        assert [tuple(p.shape) for p in params] == [(4, 5), (4, 6), (4, 7), (2, 4)]
        assert solver.defaults == {"lr": 0.5, "initialize": True, "ridge": 1e-3}
        solver.step(grid((6, 5, 4), 2))
        state = solver.state_dict()
        assert state["param_groups"][0]["ridge"] == 1e-3
        assert all(entry["total"] == 0.5 for entry in state["state"].values())

    def test_takes_any_batch_of_points(self):
        """A grid, its points in any order or a loader's batch of one: one sweep."""
        batch = grid((7, 9), 2)
        torch.manual_seed(0)
        model = futon(2, 2, 6, 3)
        flat = {key: value.reshape(-1, value.shape[-1]) for key, value in batch.items()}
        order = torch.randperm(63)
        variants = [
            flat,
            {key: value[order] for key, value in flat.items()},
            {key: value.unsqueeze(0) for key, value in batch.items()},
        ]
        reference = copy.deepcopy(model)
        LeastSquares(reference, initialize=False).step(batch)
        for variant in variants:
            other = copy.deepcopy(model)
            LeastSquares(other, initialize=False).step(variant)
            for p, q in zip(other.parameters(), reference.parameters()):
                assert torch.allclose(p, q, atol=1e-10)

    def test_draws_the_parameters_its_solver_wants(self):
        torch.manual_seed(0)
        model = futon(2, 2, 8, 3, "triangle", signed)
        own = copy.deepcopy(model)
        LeastSquares(copy.deepcopy(model), initialize=False)
        assert any(bool((p < 0).any()) for p in model.parameters())
        MultiplicativeUpdate(model)
        assert all(bool((p > 0).all()) for p in model.parameters())
        kept = copy.deepcopy(own)
        LeastSquares(kept, initialize=False)
        for p, q in zip(kept.parameters(), own.parameters()):
            assert torch.equal(p, q)

    def test_the_first_batch_counts_whole(self):
        """The average is bias-corrected: its first batch has all the weight."""
        batch = grid((10, 12), 2)
        torch.manual_seed(0)
        model = futon(2, 2, 6, 3)
        alone, averaged = copy.deepcopy(model), copy.deepcopy(model)
        LeastSquares(alone, momentum=0.0, initialize=False).step(batch)
        LeastSquares(averaged, momentum=0.9, initialize=False).step(batch)
        for p, q in zip(alone.parameters(), averaged.parameters()):
            assert torch.allclose(p, q)

    def test_factors_are_held_at_unit_scale(self):
        """A sweep leaves every factor's components at unit mean square."""
        batch = grid((10, 12), 2)
        torch.manual_seed(0)
        model = futon(2, 2, 6, 3)
        before = model(batch["input"]).detach()
        solver = LeastSquares(model)
        solver.step(batch)
        x = batch["input"]
        for c, (u, linear) in enumerate(
            zip((x[:, 0, 0], x[0, :, 1]), model.combiner.linears)
        ):
            response = model.basis.axis_features(u, c) @ linear.weight.T
            assert torch.allclose(
                response.square().mean(0), torch.ones(3, dtype=DOUBLE)
            )
        assert not torch.allclose(model(x), before)


class TestLeastSquares:
    def test_block_updates_are_least_squares_solutions(self):
        """On a whole grid, each block takes the solution of its own problem."""
        batch = grid((12, 10), 2)
        torch.manual_seed(1)
        model = futon(2, 2, [6, 5], 3)
        solver = LeastSquares(model, ridge=0.0, momentum=0.0, initialize=False)
        x, y = batch["input"], batch["target"].reshape(-1, 2)
        fibers = (x[:, 0, 0], x[0, :, 1])
        solutions, solve = [], solver.solve

        def recording(value, right, gram, left=None):
            with torch.no_grad():
                features = [
                    (
                        model.basis.axis_features(u, c) @ linear.weight.T
                        if c != len(solutions)
                        else model.basis.axis_features(u, c)
                    )
                    for c, (u, linear) in enumerate(zip(fibers, model.combiner.linears))
                ]
                weight = model.decoder.weight
                if (
                    len(solutions) < 2
                ):  # a factor: the design is its features times the rest
                    other = features[1 - len(solutions)]
                    phi = features[len(solutions)]
                    if len(solutions) == 0:
                        design = torch.einsum("ik,jr,dr->ijdkr", phi, other, weight)
                    else:
                        design = torch.einsum("jk,ir,dr->ijdkr", phi, other, weight)
                    design = design.reshape(-1, phi.shape[1] * 3)
                    target = y.reshape(-1)
                else:  # the decoder: the design is the responses' product
                    design = torch.einsum("ir,jr->ijr", *features).reshape(-1, 3)
                    target = y
                expected = torch.linalg.lstsq(design, target).solution
            result = solve(value, right, gram, left)
            solutions.append(
                (
                    result,
                    (
                        expected.reshape(result.shape)
                        if len(solutions) < 2
                        else expected.T
                    ),
                )
            )
            return result

        solver.solve = recording
        solver.step(batch)
        for result, expected in solutions:
            assert torch.allclose(result, expected, atol=1e-8)

    @pytest.mark.parametrize("basis", ["sinc", "lanczos", "triangle", "cosine"])
    def test_every_update_descends(self, basis):
        """On a whole grid a block's exact minimizer never raises the error."""
        batch = grid((30, 20), 3)
        torch.manual_seed(2)
        model = futon(2, 3, 7, 5, basis)
        errors = recorded(LeastSquares(model, ridge=0.0, momentum=0.0), batch, 30)
        assert bool((errors[1:] <= errors[:-1] + 1e-10).all())
        assert errors[-1] < 0.5 * errors[0]

    def test_three_axes(self):
        batch = grid((12, 10, 8), 2)
        torch.manual_seed(3)
        solver = LeastSquares(futon(3, 2, 6, 4), ridge=0.0, momentum=0.0)
        errors = recorded(solver, batch, 20)
        assert bool((errors[1:] <= errors[:-1] + 1e-10).all())
        assert errors[-1] < 0.5 * errors[0]

    def test_recovers_a_separable_signal(self):
        """From the model's own draw, a product of two sinc bumps is fitted exactly."""
        truth, batch = separable("sinc")
        model = futon(2, 1, [8, 6], 1, "sinc", normalize=False)
        solver = LeastSquares(model, ridge=0.0, momentum=0.0, initialize=False)
        start = mse(model, batch)
        for _ in range(50):
            solver.step(batch)
        assert mse(model, batch) < 1e-10 * start

    def test_activation_lowers_the_error(self):
        """With the tanh, the Gauss-Newton sweeps keep lowering the error."""
        batch = grid((30, 20), 3)
        torch.manual_seed(4)
        model = futon(2, 3, 7, 5, activation=torch.tanh)
        solver = LeastSquares(model, momentum=0.0)
        losses = torch.stack([solver.step(batch) for _ in range(30)])
        assert losses[-1] < 0.5 * losses[0]
        assert losses[-1] <= losses[:10].min()

    def test_minibatches_with_momentum_approach_the_whole_grid(self):
        batch = grid((40, 50), 2)
        generator = torch.Generator().manual_seed(0)
        torch.manual_seed(5)
        model = futon(2, 2, [10, 12], 4)
        whole = copy.deepcopy(model)
        solver = LeastSquares(whole, momentum=0.0)
        for _ in range(30):
            solver.step(batch)
        solver = LeastSquares(model, momentum=0.9)
        for _ in range(300):
            solver.step(sample(batch, 0.2, generator))
        assert mse(model, batch) < 1.5 * mse(whole, batch)


class TestMultiplicativeUpdate:
    def test_needs_a_nonnegative_model(self):
        batch = grid((20, 30), 2)
        for model in [
            futon(2, 2, 8, 3, "sinc", signed),  # a signed basis
            futon(2, 2, 8, 3, "triangle"),  # targets in [-1, 1]
            futon(2, 2, 8, 3, "triangle", torch.tanh),  # Gauss-Newton targets of a tanh
        ]:
            with pytest.raises(ValueError):
                MultiplicativeUpdate(model).step(batch)
        kept = MultiplicativeUpdate(
            futon(2, 2, 8, 3, "triangle", signed), initialize=False
        )
        with pytest.raises(ValueError):  # the model's own draw has negative entries
            kept.step(batch)

    @pytest.mark.parametrize("sizes", [(40,), (30, 20), (12, 10, 8)])
    def test_every_update_descends(self, sizes):
        """On a whole grid the error never increases, after any block's update."""
        batch = grid(sizes, 3)
        torch.manual_seed(1)
        model = futon(len(sizes), 3, 7, 5, "triangle", signed, normalize=False)
        errors = recorded(MultiplicativeUpdate(model, momentum=0.0), batch, 100)
        assert errors.shape == (100 * (len(sizes) + 1) + 1,)
        assert bool((errors[1:] <= errors[:-1] + 1e-12).all())
        assert errors[-1] < 0.6 * errors[0]

    def test_every_update_is_the_gradient_ratio(self):
        """Each block moves by the ratio of its gradient's two parts.

        The negative part of the squared error's gradient is the gradient of
        the inner product with the target, the positive part that of half the
        squared norm of the output; autograd supplies both, at the state each
        block is updated from.
        """
        batch = grid((20, 15), 3)
        torch.manual_seed(3)
        model = futon(2, 3, 6, 4, "triangle", signed, normalize=False)
        solver = MultiplicativeUpdate(model, momentum=0.0)
        blocks = [
            *(linear.weight for linear in model.combiner.linears),
            model.decoder.weight,
        ]
        updates, solve = [], solver.solve

        def recording(value, right, gram, left=None):
            block = blocks[len(updates)]
            with torch.enable_grad():
                output = (model(batch["input"]) + 1) / 2
                target = (batch["target"] + 1) / 2
                negative = torch.autograd.grad(
                    (output * target).sum(), block, retain_graph=True
                )[0]
                positive = torch.autograd.grad(0.5 * output.square().sum(), block)[0]
            expected = block.detach() * negative / positive
            result = solve(value, right, gram, left)
            updates.append(
                (result, expected.T if block is not blocks[-1] else expected)
            )
            return result

        solver.solve = recording
        solver.step(batch)
        assert len(updates) == 3
        for result, expected in updates:
            assert torch.allclose(result, expected)

    def test_minibatches_keep_it_nonnegative(self):
        batch = grid((40, 50), 2)
        generator = torch.Generator().manual_seed(0)
        torch.manual_seed(2)
        model = futon(2, 2, 6, 4, "triangle", signed, normalize=False)
        solver = MultiplicativeUpdate(model, momentum=0.5)
        start = mse(model, batch)
        for _ in range(100):
            solver.step(sample(batch, 0.2, generator))
        assert mse(model, batch) < 0.5 * start
        assert all(
            bool((p >= 0).all() and p.isfinite().all()) for p in model.parameters()
        )

    def test_exact_solution_is_a_fixed_point(self):
        """A separable target the model holds exactly is left as it is."""
        truth, batch = separable("triangle")
        before = truth(batch["input"]).detach()
        MultiplicativeUpdate(truth, initialize=False).step(batch)
        assert mse(truth, batch) < 1e-24
        assert torch.allclose(truth(batch["input"]), before, atol=1e-12)

    def test_recovers_a_separable_signal(self):
        """From a positive draw, a product of two hat functions is fitted to 40 dB."""
        truth, batch = separable("triangle")
        model = futon(2, 1, [8, 6], 1, "triangle", signed, normalize=False)
        solver = MultiplicativeUpdate(model, momentum=0.0)
        start = mse(model, batch)
        for _ in range(500):
            solver.step(batch)
        assert mse(model, batch) < 1e-4 * start


class TestTraining:
    @pytest.mark.parametrize("solver", [LeastSquares, MultiplicativeUpdate])
    def test_train_takes_a_solver_in_place_of_adam(self, tmp_path, solver):
        torch.manual_seed(0)
        image = torch.arange(3 * 16 * 16, dtype=torch.uint8).reshape(3, 16, 16)
        dataset = nf.ImageCoordinateDataset(image, item_id="image", subsample=0.5)
        nonnegative = solver is MultiplicativeUpdate
        model = nf.FUTON(
            2,
            3,
            basis=(
                "triangle" if nonnegative else "sinc",
                {"num_components": 6, "normalize": not nonnegative},
            ),
            combiner=("cp", {"rank": 3}),
            decoder=("linear", {"bias": False}),
            output_activation=signed if nonnegative else torch.tanh,
        )
        results = nf.train(
            model,
            dataset,
            nf.ImageCoordinateDataset(image, item_id="image"),
            optimizer=solver(model, momentum=0.5),
            num_epochs=20,
            eval_interval=20,
            metrics={"psnr": nf.psnr},
            log_dir=tmp_path,
            device="cpu",
        )
        config, history = results["config"], results["history"]
        assert config["optimizer"] == solver.__name__
        assert config["lr"] == 0.5 and config["scheduler"] == "CosineAnnealingLR"
        assert len(history) == 20 and history[-1]["mse"] < history[0]["mse"]
        assert np.isfinite(history[-1]["eval"]["psnr"])
        assert (tmp_path / "checkpoint.pt").exists()
