"""Render the package's Sphinx-style docstring roles in mkdocstrings.

The docstrings are written in Google style but use a few reStructuredText
roles: ``:math:`` for LaTeX, and ``:class:``, ``:func:``, ``:meth:``,
``:attr:``, ``:data:`` and ``:mod:`` for cross-references. This griffe
extension rewrites them into the Markdown and MathJax syntax the site renders.
A cross-reference becomes a link when its target is a documented object of the
package (looked up from the docstring's own scope outwards, as Sphinx does),
and plain code otherwise, so that external targets such as ``torch.save``
never break a strict build.
"""

import re
from collections.abc import Iterator

from griffe import AliasResolutionError, CyclicAliasError, Extension, Module, Object

_MATH = re.compile(r":math:`([^`]+)`")
# :role:`~pkg.mod.Name` shows "Name"; :role:`pkg.mod.Name` shows the full path.
_XREF = re.compile(
    r":(?:class|func|meth|attr|data|mod|obj):`(~?)([^`<]+?)(?:\s*<[^>]*>)?`"
)


def _walk(obj: Object) -> Iterator[Object]:
    yield obj
    for member in obj.members.values():
        if not member.is_alias:
            yield from _walk(member)


def _scopes(obj: Object) -> Iterator[Object]:
    scope: Object | None = obj
    while scope is not None:
        yield scope
        scope = scope.parent


def _lookup(pkg: Module, path: str) -> Object | None:
    """The documented object at ``path``, with aliases followed, or ``None``."""
    prefix = pkg.name + "."
    if path == pkg.name:
        return pkg
    relative = path[len(prefix) :] if path.startswith(prefix) else path
    try:
        target = pkg[relative]
        target = target.final_target if target.is_alias else target
    except (KeyError, AliasResolutionError, CyclicAliasError, ValueError):
        return None
    if target.docstring is None or target.name.startswith("_"):
        return None
    return target


def _resolve(pkg: Module, obj: Object, name: str) -> Object | None:
    candidates = [name] if name.startswith(pkg.name + ".") else []
    candidates += [f"{scope.path}.{name}" for scope in _scopes(obj)]
    candidates += [f"{pkg.name}.{name}"]
    candidates += [
        f"{member.path}.{name}"
        for member in pkg.members.values()
        if member.is_module and not member.is_alias
    ]
    for candidate in candidates:
        target = _lookup(pkg, candidate)
        if target is not None:
            return target
    return None


def rewrite(text: str, pkg: Module, obj: Object) -> str:
    def xref(match: re.Match) -> str:
        tilde, path = match.groups()
        path = path.strip()
        label = path.rsplit(".", 1)[-1] if tilde else path
        target = _resolve(pkg, obj, path)
        if target is None:
            return f"`{label}`"
        return f"[`{label}`][{target.path}]"

    text = _MATH.sub(lambda m: f"${m.group(1)}$", text)
    return _XREF.sub(xref, text)


class RSTRoles(Extension):
    """Rewrite ``:math:`` and cross-reference roles in every docstring."""

    def on_package(self, *, pkg: Module, **kwargs) -> None:
        """Rewrite every docstring once the whole package is loaded."""
        for obj in _walk(pkg):
            if obj.docstring is not None:
                obj.docstring.value = rewrite(obj.docstring.value, pkg, obj)

    # griffe 1.x named this hook differently.
    on_package_loaded = on_package
