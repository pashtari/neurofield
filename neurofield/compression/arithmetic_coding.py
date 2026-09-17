"""Integer arithmetic coding with per-position probability mass functions.

Every position of a symbol sequence is coded under its own probability mass
function (PMF), and alphabet sizes may differ between positions. PMFs are scaled
to integer frequency tables and coded with a 32-bit integer arithmetic coder
using bit-plus-follow (underflow) renormalization.

References:
    Witten et al., "Arithmetic Coding for Data Compression", Communications of
    the ACM 1987.
"""

import struct
from bisect import bisect_right
from collections.abc import Sequence
from typing import TypeAlias

import numpy as np
from torch import Tensor

__all__ = ["encode_arithmetic", "decode_arithmetic"]

# A probability vector; normalization is handled by the coder.
_PMF: TypeAlias = Tensor | np.ndarray | Sequence[float]

_STATE_BITS = 32
_MAX_STATE = (1 << _STATE_BITS) - 1
_HALF = 1 << (_STATE_BITS - 1)
_QUARTER = _HALF >> 1
_THREE_QUARTERS = _HALF + _QUARTER


class _BitWriter:
    """MSB-first bit accumulator that packs bits into bytes."""

    def __init__(self) -> None:
        self.buffer = bytearray()
        self.current_byte = 0
        self.bits_filled = 0

    def write_bit(self, bit: int) -> None:
        """Append the least significant bit of ``bit``."""
        self.current_byte = (self.current_byte << 1) | (bit & 1)
        self.bits_filled += 1
        if self.bits_filled == 8:
            self.buffer.append(self.current_byte)
            self.current_byte = 0
            self.bits_filled = 0

    def finish(self) -> bytes:
        """Flush the last partial byte (zero-padded on the right); return all bytes."""
        if self.bits_filled:
            self.current_byte <<= 8 - self.bits_filled
            self.buffer.append(self.current_byte)
            self.current_byte = 0
            self.bits_filled = 0
        return bytes(self.buffer)


class _BitReader:
    """MSB-first bit reader that yields zeros past the end of the data."""

    def __init__(self, data: bytes) -> None:
        self.data = data
        self.position = 0
        self.current_byte = 0
        self.bits_left = 0

    def read_bit(self) -> int:
        """Return the next bit, or ``0`` once the data is exhausted."""
        if self.bits_left == 0:
            if self.position < len(self.data):
                self.current_byte = self.data[self.position]
                self.position += 1
            else:
                self.current_byte = 0
            self.bits_left = 8
        bit = (self.current_byte >> 7) & 1
        self.current_byte = (self.current_byte << 1) & 0xFF
        self.bits_left -= 1
        return bit


def _to_numpy(x: _PMF) -> np.ndarray:
    """Convert a Tensor (any device) or array-like to a NumPy array."""
    if isinstance(x, Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x, dtype=float)


def _pmf_to_frequencies(pmf: _PMF, precision: int = 1 << 16) -> tuple[np.ndarray, int]:
    """Scale a PMF to integer symbol frequencies.

    Returns ``(frequencies, total)`` after normalizing, scaling by ``precision``
    and flooring. Positive probabilities keep at least one count, so the total
    may exceed ``precision``. Empty, negative or all-zero PMFs are rejected.
    """
    probabilities = _to_numpy(pmf).astype(float).ravel()
    if probabilities.size == 0:
        raise ValueError("Empty PMF")
    if np.any(probabilities < 0):
        raise ValueError("Negative probabilities in PMF")
    probability_sum = probabilities.sum()
    if probability_sum <= 0:
        raise ValueError("PMF sums to zero")
    frequencies = np.floor(probabilities / probability_sum * precision + 1e-12).astype(
        int
    )
    # Keep rare, positive-probability symbols representable after rounding.
    frequencies[(probabilities > 0) & (frequencies == 0)] = 1
    total = int(frequencies.sum())
    if total == 0:
        frequencies[:] = 1
        total = int(frequencies.sum())
    return frequencies, total


def _frequency_tables(
    pmfs: Sequence[_PMF], precision: int
) -> list[tuple[np.ndarray, int]]:
    """Return ``(cumulative counts with a leading zero, total)`` for each PMF."""
    tables = []
    for pmf in pmfs:
        frequencies, total = _pmf_to_frequencies(pmf, precision=precision)
        cumulative = np.zeros(len(frequencies) + 1, dtype=np.int64)
        cumulative[1:] = np.cumsum(frequencies, dtype=np.int64)
        tables.append((cumulative, total))
    return tables


def encode_arithmetic(
    symbols: Tensor | np.ndarray | Sequence[int],
    pmfs: list[_PMF] | tuple[_PMF, ...],
    precision: int = 1 << 16,
) -> bytes:
    """Encode a symbol sequence with arithmetic coding.

    Position ``i`` is coded under its own PMF ``pmfs[i]``, and alphabet sizes may
    differ between positions. Every symbol must have nonzero probability under
    its PMF: a zero-probability symbol is not rejected and silently corrupts the
    bitstream.

    Args:
        symbols: Integer symbols, flattened to shape :math:`(N,)`. A Tensor must
            be on the CPU.
        pmfs: List or tuple of ``N`` PMFs; ``pmfs[i][k]`` is the (not necessarily
            normalized) probability of symbol ``k`` at position ``i``. Each PMF is
            a Tensor (any device), a NumPy array or a sequence of floats.
        precision: Total integer frequency each PMF is scaled to (a count, not a
            number of bits). Nonzero probabilities that would round to zero get
            frequency ``1``, so a table's total can slightly exceed it. Must match
            the value passed to :func:`decode_arithmetic`. Default: ``1 << 16``.

    Returns:
        A 4-byte big-endian ``uint32`` symbol count ``N`` followed by the
        compressed bitstream.

    Raises:
        ValueError: If ``pmfs`` is not a list or tuple of length ``N``, a PMF is
            empty, has a negative entry or sums to zero, or a symbol is outside
            the alphabet of its PMF.

    Examples::

        >>> pmfs = [[0.7, 0.3], [0.1, 0.2, 0.7], [0.4, 0.6]]
        >>> encoded = encode_arithmetic([0, 2, 1], pmfs)
        >>> decode_arithmetic(encoded, pmfs)
        array([0, 2, 1])
    """
    symbols = np.asarray(symbols, dtype=np.int64).ravel()
    num_symbols = symbols.size
    if not isinstance(pmfs, (list, tuple)) or len(pmfs) != num_symbols:
        raise ValueError("pmfs must be a list/tuple with same length as symbols")
    tables = _frequency_tables(pmfs, precision)

    writer = _BitWriter()
    low = 0
    high = _MAX_STATE
    pending_bits = 0

    def write_bit_with_pending(bit: int) -> None:
        nonlocal pending_bits
        writer.write_bit(bit)
        while pending_bits > 0:
            writer.write_bit(1 - bit)
            pending_bits -= 1

    for index, symbol in enumerate(symbols):
        cumulative, total = tables[index]
        alphabet_size = len(cumulative) - 1
        if symbol < 0 or symbol >= alphabet_size:
            raise ValueError(
                f"symbol {symbol} out of range for PMF at position {index}"
            )
        interval_size = high - low + 1
        high = low + (interval_size * int(cumulative[symbol + 1]) // total) - 1
        low = low + (interval_size * int(cumulative[symbol]) // total)

        # Emit a shared leading bit; defer bits while the interval straddles 1/2.
        while True:
            if high < _HALF:
                write_bit_with_pending(0)
            elif low >= _HALF:
                write_bit_with_pending(1)
                low -= _HALF
                high -= _HALF
            elif low >= _QUARTER and high < _THREE_QUARTERS:
                pending_bits += 1
                low -= _QUARTER
                high -= _QUARTER
            else:
                break
            low = (low << 1) & _MAX_STATE
            high = ((high << 1) & _MAX_STATE) | 1

    # One final bit and the deferred bits identify the remaining interval.
    pending_bits += 1
    write_bit_with_pending(0 if low < _QUARTER else 1)

    return struct.pack(">I", num_symbols) + writer.finish()


def decode_arithmetic(
    encoded: bytes,
    pmfs: list[_PMF] | tuple[_PMF, ...],
    precision: int = 1 << 16,
) -> np.ndarray:
    """Decode a bitstream produced by :func:`encode_arithmetic`.

    Args:
        encoded: Bytes returned by :func:`encode_arithmetic`.
        pmfs: List or tuple of the PMFs used for encoding, one per symbol.
        precision: The ``precision`` used for encoding. Default: ``1 << 16``.

    Returns:
        The decoded symbols as an ``int64`` array of shape :math:`(N,)`.

    Raises:
        ValueError: If ``encoded`` is shorter than the 4-byte header, ``pmfs`` is
            not a list or tuple whose length equals the encoded symbol count, or a
            PMF is empty, has a negative entry or sums to zero.
    """
    if len(encoded) < 4:
        raise ValueError("Input too short")
    num_symbols = struct.unpack(">I", encoded[:4])[0]
    if not isinstance(pmfs, (list, tuple)) or len(pmfs) != num_symbols:
        raise ValueError(
            "pmfs must be a list/tuple with same length as encoded symbols"
        )
    tables = _frequency_tables(pmfs, precision)

    reader = _BitReader(encoded[4:])
    value = 0
    for _ in range(_STATE_BITS):
        value = (value << 1) | reader.read_bit()

    low = 0
    high = _MAX_STATE
    symbols = np.zeros(num_symbols, dtype=np.int64)

    for index, (cumulative, total) in enumerate(tables):
        interval_size = high - low + 1
        scaled = ((value - low + 1) * total - 1) // interval_size
        # Find the symbol whose cumulative-frequency interval contains scaled.
        symbol = max(bisect_right(cumulative, int(scaled)) - 1, 0)
        symbols[index] = symbol

        high = low + (interval_size * int(cumulative[symbol + 1]) // total) - 1
        low = low + (interval_size * int(cumulative[symbol]) // total)

        # Mirror the encoder's renormalization while consuming new bits.
        while True:
            if high < _HALF:
                pass
            elif low >= _HALF:
                value -= _HALF
                low -= _HALF
                high -= _HALF
            elif low >= _QUARTER and high < _THREE_QUARTERS:
                value -= _QUARTER
                low -= _QUARTER
                high -= _QUARTER
            else:
                break
            low = (low << 1) & _MAX_STATE
            high = ((high << 1) & _MAX_STATE) | 1
            value = ((value << 1) | reader.read_bit()) & _MAX_STATE

    return symbols
