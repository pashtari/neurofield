"""Block coordinate descent for FUTON: solvers in place of a gradient optimizer.

A FUTON with a CP combiner and a linear decoder, both without bias, is the
paper's model, and it is multilinear: with the other blocks held, its output
is linear in each factor matrix and in the decoder, so fitting one block to a
batch is a least-squares problem. :class:`BCD` sweeps the blocks once per
batch and leaves each block's update to a subclass. :class:`LeastSquares`
solves the block's problem, for any basis, and handles an output activation
by refitting Gauss-Newton targets every sweep; :class:`MultiplicativeUpdate`
takes Lee and Seung's step instead, which keeps a nonnegative model
nonnegative. Both are optimizers that :func:`neurofield.train` takes in place
of Adam, on the same datasets: a batch is any set of points, a whole grid or
a random tenth of one, and ``momentum`` averages the blocks' problems over
the batches, as online matrix factorization does, so that a minibatch still
sees the whole signal over a few steps.

References:
    Lee and Seung, "Algorithms for non-negative matrix factorization",
    NeurIPS 2001.
    Mairal, Bach, Ponce and Sapiro, "Online learning for matrix factorization
    and sparse coding", JMLR 2010.
"""

from collections.abc import Mapping, Sequence
from functools import reduce
from typing import Any

import torch
from torch import Tensor, nn
from torch.optim import Optimizer

from .futon import FUTON, CPCombiner

__all__ = ["BCD", "LeastSquares", "MultiplicativeUpdate"]

AXES = "ijklmn"  # einsum letters of the grid's axes; r is the rank and d a channel


def ridged(gram: Tensor, ridge: float) -> Tensor:
    """A Gram matrix with ``ridge`` times its mean diagonal added to the diagonal."""
    eye = torch.eye(len(gram), dtype=gram.dtype, device=gram.device)
    return gram + ridge * gram.diagonal().mean() * eye


class BCD(Optimizer):
    """Block coordinate descent over a FUTON's factor matrices and decoder.

    For a batch of ``N`` points, let ``Phi`` be axis ``c``'s features
    ``(N, K_c)``, ``A`` its factor matrix ``(K_c, R)``, ``H`` the product of
    the other axes' responses ``Phi_j A_j``, ``W`` the decoder ``(D, R)`` and
    ``T`` the targets ``(N, D)``. The output ``((Phi A) * H) W^T`` is linear in
    ``A``, and on a whole grid the block's normal equations are ``P A Q = B``,
    small and of the size of ``A``: ``P = Phi^T Phi / N`` is the basis's Gram
    matrix, ``Q`` the elementwise product of ``W^T W`` and of the other axes'
    Gram matrices ``(Phi_j A_j)^T (Phi_j A_j) / N``, and
    ``B = Phi^T (H * T W) / N``. The decoder's are ``W G = C``, with ``G`` the
    product of every axis's Gram matrix and ``C = T^T Z / N`` for the
    responses' product ``Z``.

    Any other batch, such as a random tenth of the points, is taken as the
    grid of its distinct coordinates, each weighted by its share of the
    points: the Gram matrices are that grid's, as they are in expectation for
    points drawn uniformly, and each right-hand side is what the model
    explains on that grid plus what it misses on the batch,
    ``B = P A Q + Phi^T (H * E W) / N`` for the residual ``E``. That is the
    batch's right-hand side in expectation, with the noise of the residual
    rather than of the signal, so a minibatch's problem approaches the whole
    signal's as the model fits it.

    :meth:`step` sweeps the factors in order and then the decoder, each from
    the others' latest values, and hands every problem to a subclass's
    :meth:`solve`. A factor's components are then rescaled to unit mean square
    and the decoder's columns take the scales, which leaves the model as it
    was and fixes the freedom of a CP model to trade scale between its blocks.
    On a whole grid without momentum the batch's error never increases.

    ``momentum`` averages each block's statistics over the batches, as online
    matrix factorization does: each block then solves the problem of every
    batch seen, the latest weighted the most, with weights that sum to one
    from the first batch on. The average lags the other blocks, which have
    moved since its earlier batches, so the best weight trades the noise of a
    few batches against the lag of many. The newest batch's weight,
    ``1 - momentum``, is the step size of the average and the solver's
    learning rate ``lr``, so a scheduler anneals it as it would a gradient
    step's: :func:`neurofield.train` takes it down to a hundredth, and the
    average reaches further back as the model settles.

    Args:
        model: A FUTON with a CP combiner and a linear decoder, both without
            bias, whose basis evaluates one axis at a time through
            ``axis_features``, as the built-in ones do. Its parameters are
            updated in place.
        momentum: Weight of the past batches in a block's statistics; ``0``
            takes each batch on its own, which suits whole grids.
        initialize: Draw the parameters anew with :meth:`reset_parameters`;
            ``False`` keeps the model's.
        **defaults: A subclass's settings, kept in ``self.defaults`` with the
            others.
    """

    def __init__(
        self,
        model: FUTON,
        momentum: float = 0.9,
        initialize: bool = True,
        **defaults: Any,
    ) -> None:
        if not (
            isinstance(model, FUTON)
            and isinstance(model.combiner, CPCombiner)
            and isinstance(model.decoder, nn.Linear)
            and model.combiner.linears[0].bias is None
            and model.decoder.bias is None
        ):
            raise ValueError(
                "block coordinate descent fits a FUTON with a CP combiner and a"
                " linear decoder, both without bias"
            )
        if not 0 <= momentum < 1:
            raise ValueError("momentum must be in [0, 1)")
        factors = [linear.weight for linear in model.combiner.linears]
        settings = {"lr": 1 - momentum, "initialize": initialize}
        super().__init__([*factors, model.decoder.weight], settings | defaults)
        self.model = model
        if initialize:
            self.reset_parameters()

    def _draw(self, param: Tensor) -> Tensor:
        """A parameter's entries drawn anew, before scaling."""
        raise NotImplementedError

    @torch.no_grad()
    def reset_parameters(self) -> None:
        """Draw the parameters as the solver wants them.

        A subclass draws the entries; each axis's response is then scaled to
        about one in the middle of the domain and the decoder sums the ``R``
        components to about one half, so that the first sweep starts from an
        output of the signal's order.
        """
        model = self.model
        for c, linear in enumerate(model.combiner.linears):
            middle = model.basis.axis_features(linear.weight.new_zeros(1), c)
            linear.weight.copy_(self._draw(linear.weight) / middle.sum())
        weight = model.decoder.weight
        weight.copy_(self._draw(weight) / (2 * weight.shape[1]))

    def _validate(self, features: Sequence[Tensor], targets: Tensor) -> None:
        """Check a batch before its sweep; the base class has no checks.

        ``features`` holds each axis's features at its distinct coordinates.
        """

    @staticmethod
    def _output(responses: Sequence[Tensor], weight: Tensor) -> Tensor:
        """The decoder's output on the grid, ``(M_1, ..., M_C, D)``."""
        letters = AXES[: len(responses)]
        operands = ",".join(f"{a}r" for a in letters)
        return torch.einsum(f"{operands},dr->{letters}d", *responses, weight)

    def _targets(self, y: Tensor, output: Tensor) -> Tensor:
        """What the decoder's output should fit for the model's to approach ``y``.

        Without an output activation the values themselves. With one, the
        Gauss-Newton target from the current ``output``: the output plus the
        residual over the activation's slope, with the squared slopes bounded
        by their largest so that every block stays an unweighted
        least-squares problem. For an affine activation it is exact.
        """
        activation = self.model.output_activation
        if isinstance(activation, nn.Identity):
            return y
        with torch.enable_grad():
            z = output.detach().requires_grad_()
            value = activation(z)
            slope = torch.autograd.grad(value.sum(), z)[0]
        return output + slope * (y - value.detach()) / slope.square().max()

    def _average(self, param: Tensor, *statistics: Tensor) -> tuple[Tensor, ...]:
        """A block's statistics, averaged over the batches seen."""
        lr = self.param_groups[0]["lr"]
        state = self.state[param]
        # The batches' weights sum to one, whatever the step size was at each.
        state["total"] = (1 - lr) * state.get("total", 0.0) + lr
        weight = lr / state["total"]
        averages = state.get("statistics", statistics)
        state["statistics"] = tuple(
            torch.lerp(average, new, weight)
            for average, new in zip(averages, statistics)
        )
        return state["statistics"]

    def solve(
        self, value: Tensor, right: Tensor, gram: Tensor, left: Tensor | None = None
    ) -> Tensor:
        """A block's new value from its problem ``left @ value @ gram = right``.

        Args:
            value: The block, ``(K_c, R)`` for a factor and ``(D, R)`` for the
                decoder.
            right: The right-hand side, of the block's shape.
            gram: The Gram matrix of the components, ``(R, R)``.
            left: The basis's Gram matrix ``(K_c, K_c)``; ``None``, the
                identity, for the decoder.
        """
        raise NotImplementedError

    @torch.no_grad()
    def step(self, batch: Mapping[str, Tensor]) -> Tensor:  # type: ignore[override]
        """One sweep over the factor matrices and the decoder on a batch.

        Args:
            batch: ``input``, coordinates ``(*, C)`` in [-1, 1], and
                ``target``, their values ``(*, D)``, as a dataset yields them.
                The points must lie on a grid, as a coordinate dataset's do,
                since the sweep evaluates the model on the grid of their
                distinct coordinates.

        Returns:
            The batch's mean squared error before the sweep.
        """
        model = self.model
        x = batch["input"].reshape(-1, model.in_features)
        y = batch["target"].reshape(-1, model.out_features)
        count = len(x)
        # The points lie on the grid of their distinct coordinates, where the
        # output is a CP tensor: every axis is evaluated once per coordinate,
        # and the points pick their entries of the tensor.
        axes = [torch.unique(u, return_inverse=True) for u in x.T]
        index = tuple(inverse for _, inverse in axes)
        features = [model.basis.axis_features(u, c) for c, (u, _) in enumerate(axes)]
        shares = [
            torch.bincount(inverse, minlength=len(u)).to(x.dtype)[:, None] / count
            for u, inverse in axes
        ]
        factors = [linear.weight for linear in model.combiner.linears]
        weight = model.decoder.weight
        responses = [phi @ factor.T for phi, factor in zip(features, factors)]
        grams = [r.T @ (share * r) for r, share in zip(responses, shares)]
        output = self._output(responses, weight)[index]
        loss = (model.output_activation(output) - y).square().mean()
        targets = self._targets(y, output)
        self._validate(features, targets)

        def residual(predicted: Tensor) -> Tensor:
            """The points' residuals, summed into the grid."""
            grid = y.new_zeros(*(len(r) for r in responses), y.shape[1])
            return grid.index_put_(index, targets - predicted, accumulate=True)

        letters = AXES[: len(responses)]
        missed = residual(output)
        for c, (phi, share, factor) in enumerate(zip(features, shares, factors)):
            rest = [j for j in range(len(responses)) if j != c]
            left = phi.T @ (share * phi)
            gram = reduce(torch.mul, [grams[j] for j in rest], weight.T @ weight)
            operands = ",".join(
                [f"{letters}d", *(f"{letters[j]}r" for j in rest), "dr"]
            )
            others = [responses[j] for j in rest]
            correlation = torch.einsum(
                f"{operands}->{letters[c]}r", missed, *others, weight
            )
            # What the model explains, from the Gram matrices, and what it
            # misses, from the batch; see the class's notes.
            right = left @ factor.T @ gram + phi.T @ correlation / count
            left, right, gram = self._average(factor, left, right, gram)
            value = self.solve(factor.T, right, gram, left)
            # A component scaled in one factor and back in the decoder leaves
            # the model as it was; holding every factor's components at unit
            # mean square keeps the averaged statistics in one scale.
            scale = (value * (left @ value)).sum(0).sqrt()
            scale = torch.where(scale > 0, scale, 1.0)
            factor.copy_((value / scale).T)
            weight.mul_(scale)
            responses[c] = phi @ factor.T
            grams[c] = responses[c].T @ (share * responses[c])
            missed = residual(self._output(responses, weight)[index])
        operands = ",".join([f"{letters}d", *(f"{a}r" for a in letters)])
        correlation = torch.einsum(f"{operands}->dr", missed, *responses)
        gram = reduce(torch.mul, grams)
        right, gram = self._average(weight, weight @ gram + correlation / count, gram)
        weight.copy_(self.solve(weight, right, gram))
        return loss


class LeastSquares(BCD):
    """Alternating least squares: every block takes its problem's solution.

    The problem ``P A Q = B`` is solved one side at a time, ``P`` from the
    left and then ``Q`` from the right, each by its Cholesky factorization
    with a ridge against collinear functions or components. Any basis will
    do, signed or not. With an activation at the output the targets are
    Gauss-Newton ones, refitted every sweep, so the sweeps descend the
    activated model's error rather than minimize a fixed one. The parameters
    start from normal draws.

    Args:
        model: See :class:`BCD`.
        ridge: Added to the diagonal of each Gram matrix, relative to its
            mean diagonal.
        momentum: See :class:`BCD`.
        initialize: See :class:`BCD`.
    """

    def __init__(
        self,
        model: FUTON,
        ridge: float = 1e-3,
        momentum: float = 0.9,
        initialize: bool = True,
    ) -> None:
        super().__init__(model, momentum, initialize, ridge=ridge)

    def _draw(self, param: Tensor) -> Tensor:
        return torch.randn_like(param)

    def solve(
        self, value: Tensor, right: Tensor, gram: Tensor, left: Tensor | None = None
    ) -> Tensor:
        ridge = self.defaults["ridge"]
        if left is not None:
            right = torch.cholesky_solve(
                right, torch.linalg.cholesky(ridged(left, ridge))
            )
        # The Gram matrix is symmetric, so the right solve is a left one transposed.
        factor = torch.linalg.cholesky(ridged(gram, ridge))
        return torch.cholesky_solve(right.T, factor).T


class MultiplicativeUpdate(BCD):
    """Lee and Seung's multiplicative updates, for a nonnegative model.

    With everything nonnegative, the two parts of a block's gradient are too,
    and the update multiplies the block elementwise by their ratio,
    ``A <- A * B / (P A Q)``, or sets an entry to zero where a minibatch's
    estimate of ``B`` is negative. That is the minimizer of the auxiliary
    function Lee and Seung build for a quadratic with a nonnegative Hessian:
    the block's problem is never worse after the update, the block stays
    nonnegative, and one at a fixed point stays there.

    The model needs a nonnegative basis, the triangle with
    ``normalize=False`` so that its functions sum to one, nonnegative targets
    and nonnegative parameters, which are drawn uniformly in [0.5, 1.5]
    before scaling. A signal in [-1, 1] is fitted through the affine output
    activation ``lambda z: 2 * z - 1``, under which the targets are its
    values rescaled to [0, 1].

    Args:
        model: See :class:`BCD`.
        floor: Lower bound on a denominator, against division by zero where a
            basis function sees no sample.
        momentum: See :class:`BCD`.
        initialize: See :class:`BCD`.
    """

    def __init__(
        self,
        model: FUTON,
        floor: float = 1e-12,
        momentum: float = 0.9,
        initialize: bool = True,
    ) -> None:
        super().__init__(model, momentum, initialize, floor=floor)

    def _draw(self, param: Tensor) -> Tensor:
        return torch.rand_like(param) + 0.5

    def _validate(self, features: Sequence[Tensor], targets: Tensor) -> None:
        # The targets of an affine activation are exact up to a few roundings,
        # which may leave a zero slightly negative.
        lowest = [*features, *self.param_groups[0]["params"], targets + 1e-5]
        if torch.stack([t.amin() for t in lowest]).amin() < 0:
            raise ValueError(
                "multiplicative updates need a nonnegative basis, nonnegative"
                " targets (an affine output activation, 2 z - 1, for a signal in"
                " [-1, 1]) and nonnegative parameters"
            )

    def solve(
        self, value: Tensor, right: Tensor, gram: Tensor, left: Tensor | None = None
    ) -> Tensor:
        denominator = value @ gram if left is None else left @ value @ gram
        floor = self.defaults["floor"]
        # A coefficient no sample of the batch reaches has both parts at zero
        # and keeps its value; one whose right-hand side is negative, which a
        # minibatch's estimate can be, goes to zero, the constrained minimizer
        # of its auxiliary function.
        ratio = (right / denominator.clamp_min(floor)).clamp_min(0)
        return value * torch.where(denominator > floor, ratio, 1.0)
