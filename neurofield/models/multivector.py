"""Learnable feature grids and expression-based combinations of lines, planes, and volumes."""

import operator
import re
from collections.abc import Sequence
from functools import reduce
from typing import Literal, TypeAlias

import torch
import torch.nn.functional as F
from torch import Tensor, nn

__all__ = ["FeatureGrid", "MultiVector"]


class FeatureGrid(nn.Module):
    """Learnable feature grid with linear, bilinear, or trilinear interpolation.

    Coordinates span ``[-1, 1]`` with ``align_corners=True``; points outside
    this interval sample the border. Input is ``(N, in_features)`` (also
    ``(N,)`` for 1D), and output is ``(N, features)``.

    Args:
        features: Feature channels at each grid vertex.
        resolution: Resolution per axis; an int creates a 1D grid, and a sequence
            of length two or three creates a 2D or 3D grid. Tensor order
            ``(H, W)`` or ``(D, H, W)`` is reversed relative to coordinate
            columns: column zero indexes ``W``, column one ``H``, and column
            two ``D``.

    The learnable ``grid`` has shape ``(1, features, *resolution)``.
    A 1D grid includes a dummy height axis for ``grid_sample``.

    Example::

        grid = FeatureGrid(features=8, resolution=(32, 64))
        features = grid(torch.rand(100, 2) * 2 - 1)  # (100, 8)
    """

    def __init__(self, features: int, resolution: int | Sequence[int]) -> None:
        super().__init__()
        self.features = features

        if isinstance(resolution, int):
            resolution = (resolution,)
        resolution = tuple(resolution)
        self.in_features = len(resolution)

        if self.in_features == 1:
            # grid_sample requires at least two spatial dimensions.
            resolution = (1,) + resolution
        elif self.in_features not in (2, 3):
            raise ValueError(f"FeatureGrid supports 1D–3D, got {self.in_features}D")

        self.grid = nn.Parameter(torch.empty(1, features, *resolution))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        r"""Re-initialize the grid features from :math:`\mathcal{N}(0, 0.1^2)`."""
        nn.init.normal_(self.grid, std=0.1)

    def forward(self, x: Tensor) -> Tensor:
        num_points = x.shape[0]

        if self.in_features == 1:
            sample_grid = F.pad(x.reshape(1, 1, num_points, 1), (0, 1))
        elif self.in_features == 2:
            sample_grid = x.reshape(1, 1, num_points, 2)
        else:
            sample_grid = x.reshape(1, 1, 1, num_points, 3)

        features = F.grid_sample(
            self.grid,
            sample_grid,
            align_corners=True,
            mode="bilinear",
            padding_mode="border",
        )
        return features.reshape(self.features, num_points).T


_ExpressionNode: TypeAlias = (
    tuple[Literal["atom"], str]
    | tuple[Literal["prod", "add", "concat"], list["_ExpressionNode"]]
)

_ATOM_RE = re.compile(r"^e(\d+)$")


def _parse_atom(name: str) -> tuple[str, tuple[int, ...]]:
    """Map an atom to its grid kind and zero-based coordinate axes."""
    match = _ATOM_RE.match(name)
    if not match:
        raise ValueError(f"Invalid atom: {name!r}")
    digits = match.group(1)
    if len(digits) == 1:
        return ("line", (int(digits) - 1,))
    if len(digits) == 2:
        return ("plane", (int(digits[0]) - 1, int(digits[1]) - 1))
    if len(digits) == 3:
        return ("volume", ())
    raise ValueError(f"Unsupported grid: {name!r}")


def _parse_expression(expression: str) -> _ExpressionNode:
    """Parse a multivector expression into a syntax tree.

    Grammar (precedence: o > + > ,):
        expr   := '[' term (',' term)* ']' | term
        term   := factor ('+' factor)*
        factor := atom ('o' atom)*
        atom   := e1 | e2 | e3 | e12 | e13 | e23 | e123
    """
    expression = expression.strip()
    if expression.startswith("[") and expression.endswith("]"):
        parts = _split_top_level(expression[1:-1], ",")
        return ("concat", [_parse_term(part) for part in parts])
    return _parse_term(expression)


def _parse_term(expression: str) -> _ExpressionNode:
    """Parse a sum of factors (``factor + factor + ...``)."""
    parts = _split_top_level(expression, "+")
    if len(parts) == 1:
        return _parse_factor(parts[0])
    return ("add", [_parse_factor(part) for part in parts])


def _parse_factor(expression: str) -> _ExpressionNode:
    """Parse a product of atoms (``atom o atom o ...``)."""
    parts = _split_top_level(expression, "o")
    if len(parts) == 1:
        return ("atom", parts[0].strip())
    return ("prod", [("atom", part.strip()) for part in parts])


def _split_top_level(expression: str, separator: str) -> list[str]:
    """Split a string by a single-character separator, respecting bracket nesting."""
    parts: list[str] = []
    current: list[str] = []
    depth = 0
    for char in expression:
        if char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
        if char == separator and depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(char)
    parts.append("".join(current))
    return parts


def _collect_atoms(node: _ExpressionNode) -> set[str]:
    """Collect the atom names used in an expression."""
    kind = node[0]
    if kind == "atom":
        return {node[1]}
    return set().union(*(_collect_atoms(child) for child in node[1]))


def _count_features(node: _ExpressionNode, atom_features: dict[str, int]) -> int:
    """Count output channels; products and sums use their first operand's width."""
    kind = node[0]
    if kind == "atom":
        return atom_features[node[1]]
    if kind in ("prod", "add"):
        return _count_features(node[1][0], atom_features)
    if kind == "concat":
        return sum(_count_features(term, atom_features) for term in node[1])
    raise ValueError(f"Unknown AST node: {kind}")


def _evaluate_node(node: _ExpressionNode, features: dict[str, Tensor]) -> Tensor:
    """Evaluate an expression using the sampled grid features."""
    kind = node[0]
    if kind == "atom":
        return features[node[1]]
    children = [_evaluate_node(child, features) for child in node[1]]
    if kind == "prod":
        return reduce(operator.mul, children)
    if kind == "add":
        return reduce(operator.add, children)
    if kind == "concat":
        return torch.cat(children, dim=-1)
    raise ValueError(f"Unknown AST node: {kind}")


def _per_grid(value: int | Sequence[int]) -> list[int]:
    """Expand an int or short sequence to one value per grid (line, plane, volume)."""
    values = [value] if isinstance(value, int) else list(value)
    return (values + values[-1:] * 3)[:3]


class MultiVector(nn.Module):
    """Combine learnable line, plane, and volume grids with an expression.

    Each atom samples a :class:`FeatureGrid` at the corresponding coordinate
    columns, following the grid decomposition of GA-Planes:

    - ``e1``, ``e2``, ``e3``: lines along coordinate columns zero, one, two.
    - ``e12``, ``e13``, ``e23``: planes over the named pair of columns.
    - ``e123``: a volume over all three columns.
    - ``o``: elementwise product; ``+``: elementwise sum.
    - ``[term, term, ...]``: channel concatenation.

    Precedence is ``o`` > ``+`` > ``,``. For example, ``"[e1 o e2, e12]"``
    concatenates a product of line features with plane features. Atoms using
    axis three require ``in_features=3``. Coordinate axes and operand widths
    are not checked for compatibility at construction.

    Args:
        in_features: Number of coordinate axes, two or three.
        expr: Expression selecting and combining grids.
        features: Feature channels per grid: an int for all grids, or
            ``(line, plane, volume)`` values; shorter sequences repeat their
            last entry. Operands of ``o`` and ``+`` must have equal widths.
        resolution: Resolution per axis per grid, with the same int/sequence
            convention. Planes are square and volumes are cubic.

    Input ``(N, in_features)`` in ``[-1, 1]`` produces ``(N, out_features)``.
    ``lines`` and ``planes`` are module dictionaries keyed by atom name, or
    ``None`` when unused; ``volume`` is the volume grid or ``None``.
    ``features_per_term`` lists the width of each top-level concatenated term
    (or one width when the expression is not bracketed).

    Example::

        model = MultiVector(2, "[e1 o e2, e12]", 8, (64, 16))
        features = model(torch.rand(100, 2) * 2 - 1)  # (100, 16)

    References:
        Sivgin et al., "Geometric Algebra Planes: Convex Implicit Neural
        Volumes", arXiv 2024.
    """

    def __init__(
        self,
        in_features: int,
        expr: str,
        features: int | Sequence[int],
        resolution: int | Sequence[int],
    ) -> None:
        super().__init__()

        if in_features not in (2, 3):
            raise ValueError(f"in_features must be 2 or 3, got {in_features}")
        self.in_features = in_features
        features = _per_grid(features)
        resolution = _per_grid(resolution)

        self._ast = _parse_expression(expr)

        atoms = _collect_atoms(self._ast)
        parsed_atoms = {name: _parse_atom(name) for name in atoms}
        line_axes = sorted(
            {axes[0] for kind, axes in parsed_atoms.values() if kind == "line"}
        )
        plane_pairs = sorted(
            {axes for kind, axes in parsed_atoms.values() if kind == "plane"}
        )
        needs_volume = any(kind == "volume" for kind, _ in parsed_atoms.values())

        self.lines = (
            nn.ModuleDict(
                {
                    f"e{axis + 1}": FeatureGrid(features[0], resolution[0])
                    for axis in line_axes
                }
            )
            if line_axes
            else None
        )
        self.planes = (
            nn.ModuleDict(
                {
                    f"e{first + 1}{second + 1}": FeatureGrid(
                        features[1], (resolution[1],) * 2
                    )
                    for first, second in plane_pairs
                }
            )
            if plane_pairs
            else None
        )
        self.volume = (
            FeatureGrid(features[2], (resolution[2],) * 3) if needs_volume else None
        )

        self._line_axes = {f"e{axis + 1}": axis for axis in line_axes}
        self._plane_pairs = {
            f"e{first + 1}{second + 1}": [first, second]
            for first, second in plane_pairs
        }

        grid = {"line": 0, "plane": 1, "volume": 2}
        self._atom_features = {
            name: features[grid[kind]] for name, (kind, _) in parsed_atoms.items()
        }

        # FeatureGrid already initializes itself; re-drawing here is redundant
        # but kept so seeded models reproduce earlier checkpoints exactly.
        self.reset_parameters()

        self._terms = self._ast[1] if self._ast[0] == "concat" else [self._ast]
        self.features_per_term = [
            _count_features(term, self._atom_features) for term in self._terms
        ]

    def reset_parameters(self) -> None:
        """Initialize every feature grid from a zero-mean normal distribution."""
        for grid in self.modules():
            if isinstance(grid, FeatureGrid):
                grid.reset_parameters()

    @property
    def out_features(self) -> int:
        """Total number of output features, ``sum(features_per_term)``."""
        return sum(self.features_per_term)

    def forward(self, x: Tensor) -> Tensor:
        features: dict[str, Tensor] = {}
        if self.lines is not None:
            for name, grid in self.lines.items():
                axis = self._line_axes[name]
                features[name] = grid(x[:, axis : axis + 1])
        if self.planes is not None:
            for name, grid in self.planes.items():
                features[name] = grid(x[:, self._plane_pairs[name]])
        if self.volume is not None:
            features["e123"] = self.volume(x)
        return _evaluate_node(self._ast, features)
