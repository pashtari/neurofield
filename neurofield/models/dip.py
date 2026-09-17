"""Deep Image Prior networks (Ulyanov et al., CVPR 2018).

Adapted from https://github.com/DmitryUlyanov/deep-image-prior/tree/master/models.
DIPSkip is the skip-connected encoder-decoder baseline; DIPUNet is a U-Net.
"""

from collections.abc import Callable, Sequence
from typing import Literal

import torch
from torch import Tensor, nn

__all__ = ["DIPUNet", "DIPSkip"]


def _conv2d(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int = 1,
    bias: bool = True,
    padding_mode: Literal["zeros", "reflect"] = "reflect",
) -> nn.Sequential:
    """Convolution with centered zero or reflection padding."""
    padding = (kernel_size - 1) // 2
    if padding_mode == "reflect":
        return nn.Sequential(
            nn.ReflectionPad2d(padding),
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, bias=bias),
        )
    if padding_mode == "zeros":
        return nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size, stride, padding, bias=bias
            )
        )
    raise ValueError(f"padding_mode must be 'zeros' or 'reflect', got {padding_mode!r}")


def _conv2d_bn_lrelu(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int = 1,
    bias: bool = True,
    padding_mode: Literal["zeros", "reflect"] = "reflect",
) -> list[nn.Module]:
    """Convolution, BatchNorm and LeakyReLU(0.2) layers of a DIPSkip block."""
    return [
        _conv2d(in_channels, out_channels, kernel_size, stride, bias, padding_mode),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(0.2, inplace=True),
    ]


def _center_crop(x: Tensor, height: int, width: int) -> Tensor:
    """Crop the spatial center of an ``(B, C, H, W)`` tensor."""
    top = (x.size(2) - height) // 2
    left = (x.size(3) - width) // 2
    return x[:, :, top : top + height, left : left + width]


class _UNetConv2(nn.Module):
    """Two convolution/ReLU layers with optional normalization."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm_layer: Callable[[int], nn.Module] | None,
        bias: bool,
        padding_mode: Literal["zeros", "reflect"],
    ) -> None:
        super().__init__()

        def block(channels: int) -> nn.Sequential:
            conv = _conv2d(
                channels, out_channels, 3, bias=bias, padding_mode=padding_mode
            )
            norm = [] if norm_layer is None else [norm_layer(out_channels)]
            return nn.Sequential(conv, *norm, nn.ReLU())

        self.conv1 = block(in_channels)
        self.conv2 = block(out_channels)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv2(self.conv1(x))


class _UNetDown(nn.Module):
    """Max pooling followed by a double convolution."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm_layer: Callable[[int], nn.Module] | None,
        bias: bool,
        padding_mode: Literal["zeros", "reflect"],
    ) -> None:
        super().__init__()
        self.conv = _UNetConv2(
            in_channels, out_channels, norm_layer, bias, padding_mode
        )
        self.down = nn.MaxPool2d(2, 2)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(self.down(x))


class _UNetUp(nn.Module):
    """Upsample, concatenate a skip feature, and apply a double convolution.

    Decoder convolutions are unnormalized. Transposed convolutions always
    include a bias, matching the reference implementation.
    """

    def __init__(
        self,
        out_channels: int,
        upsample_mode: Literal["deconv", "nearest", "bilinear"],
        bias: bool,
        padding_mode: Literal["zeros", "reflect"],
        same_num_filters: bool = False,
    ) -> None:
        super().__init__()
        in_channels = out_channels if same_num_filters else out_channels * 2
        if upsample_mode == "deconv":
            self.up = nn.ConvTranspose2d(
                in_channels, out_channels, 4, stride=2, padding=1
            )
        elif upsample_mode in ("bilinear", "nearest"):
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode=upsample_mode),
                _conv2d(
                    in_channels, out_channels, 3, bias=bias, padding_mode=padding_mode
                ),
            )
        else:
            raise ValueError(f"Unknown upsample_mode: {upsample_mode}")
        self.conv = _UNetConv2(out_channels * 2, out_channels, None, bias, padding_mode)

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        """Upsample and crop the skip feature to match odd spatial dimensions."""
        x_up = self.up(x)
        if skip.shape[2:] != x_up.shape[2:]:
            skip = _center_crop(skip, x_up.size(2), x_up.size(3))
        return self.conv(torch.cat([x_up, skip], dim=1))


class DIPUNet(nn.Module):
    """U-Net from Deep Image Prior, with four downsampling/upsampling stages.

    Maps ``(B, in_channels, H, W)`` to ``(B, out_channels, H', W')``, where
    output dimensions are cropped to multiples of 16.

    Args:
        in_channels: Input noise channels; match the dataset's ``noise_channels``.
        out_channels: Output channels.
        feature_scale: Divisor of base widths ``[64, 128, 256, 512, 1024]``.
        norm_layer: Normalization constructor for input and encoder blocks,
            or ``None`` to disable it. Decoder blocks are unnormalized.
        upsample_mode: Transposed convolution (``"deconv"``), or interpolation
            (``"nearest"`` / ``"bilinear"``) followed by a convolution.
        padding_mode: ``"zeros"`` or ``"reflect"``.
        bias: Include convolution biases; transposed convolutions always do.
        sigmoid_output: Apply a sigmoid after the final convolution.
    """

    def __init__(
        self,
        in_channels: int = 16,
        out_channels: int = 3,
        feature_scale: int = 4,
        norm_layer: Callable[[int], nn.Module] | None = nn.BatchNorm2d,
        upsample_mode: Literal["deconv", "nearest", "bilinear"] = "bilinear",
        padding_mode: Literal["zeros", "reflect"] = "reflect",
        bias: bool = True,
        sigmoid_output: bool = True,
    ) -> None:
        super().__init__()
        widths = [width // feature_scale for width in (64, 128, 256, 512, 1024)]

        self.start = _UNetConv2(in_channels, widths[0], norm_layer, bias, padding_mode)
        self.down1 = _UNetDown(widths[0], widths[1], norm_layer, bias, padding_mode)
        self.down2 = _UNetDown(widths[1], widths[2], norm_layer, bias, padding_mode)
        self.down3 = _UNetDown(widths[2], widths[3], norm_layer, bias, padding_mode)
        self.down4 = _UNetDown(widths[3], widths[4], norm_layer, bias, padding_mode)

        self.up4 = _UNetUp(widths[3], upsample_mode, bias, padding_mode)
        self.up3 = _UNetUp(widths[2], upsample_mode, bias, padding_mode)
        self.up2 = _UNetUp(widths[1], upsample_mode, bias, padding_mode)
        self.up1 = _UNetUp(widths[0], upsample_mode, bias, padding_mode)

        self.final = _conv2d(
            widths[0], out_channels, 1, bias=bias, padding_mode=padding_mode
        )
        if sigmoid_output:
            self.final = nn.Sequential(self.final, nn.Sigmoid())

    def forward(self, x: Tensor) -> Tensor:
        start = self.start(x)
        down1 = self.down1(start)
        down2 = self.down2(down1)
        down3 = self.down3(down2)
        down4 = self.down4(down3)

        up4 = self.up4(down4, down3)
        up3 = self.up3(up4, down2)
        up2 = self.up2(up3, down1)
        up1 = self.up1(up2, start)

        return self.final(up1)


class DIPSkip(nn.Module):
    """Deep Image Prior encoder-decoder with a skip branch at each scale.

    Strided convolutions downsample the input; the decoder upsamples and
    concatenates skip features. Hidden convolutions use BatchNorm and
    LeakyReLU(0.2). Maps ``(B, in_channels, H, W)`` to
    ``(B, out_channels, H, W)`` for odd kernel sizes; even kernels shrink the
    output. Center crops align branches for odd spatial dimensions.

    Args:
        in_channels: Input noise channels; match the dataset's ``noise_channels``.
        out_channels: Output channels.
        down_channels: Encoder widths, one per scale.
        up_channels: Decoder widths, one per scale.
        skip_channels: Skip widths, one per scale; channel sequences must have
            equal length.
        down_kernel_size: Encoder kernel size, or a list/tuple per scale.
        up_kernel_size: Decoder kernel size, or a list/tuple per scale.
        skip_kernel_size: Skip convolution kernel size.
        up_conv1x1: Add a 1×1 convolution/BatchNorm/LeakyReLU decoder block.
        upsample_mode: Interpolation mode, or a list/tuple per scale.
        padding_mode: ``"zeros"`` or ``"reflect"``.
        bias: Include convolution biases.
        sigmoid_output: Apply a sigmoid after the final convolution.
    """

    def __init__(
        self,
        in_channels: int = 32,
        out_channels: int = 3,
        down_channels: Sequence[int] = (128, 128, 128, 128, 128),
        up_channels: Sequence[int] = (128, 128, 128, 128, 128),
        skip_channels: Sequence[int] = (4, 4, 4, 4, 4),
        down_kernel_size: int | Sequence[int] = 3,
        up_kernel_size: int | Sequence[int] = 3,
        skip_kernel_size: int = 1,
        up_conv1x1: bool = True,
        upsample_mode: str | Sequence[str] = "bilinear",
        padding_mode: Literal["zeros", "reflect"] = "reflect",
        bias: bool = True,
        sigmoid_output: bool = True,
    ) -> None:
        super().__init__()
        num_scales = len(down_channels)
        if not num_scales == len(up_channels) == len(skip_channels):
            raise ValueError(
                "down_channels, up_channels and skip_channels must have equal length"
            )
        self.num_scales = num_scales

        if not isinstance(down_kernel_size, (list, tuple)):
            down_kernel_size = [down_kernel_size] * num_scales
        if not isinstance(up_kernel_size, (list, tuple)):
            up_kernel_size = [up_kernel_size] * num_scales
        if not isinstance(upsample_mode, (list, tuple)):
            upsample_mode = [upsample_mode] * num_scales

        # Channels entering each scale from the input side and the deeper side.
        encoder_in = [in_channels, *down_channels[:-1]]
        concat_channels = [
            skip + deeper
            for skip, deeper in zip(
                skip_channels, [*up_channels[1:], down_channels[-1]]
            )
        ]
        # Modules are built group by group (all encoders, then all skips, ...)
        # so seeded initializations match the reference implementation.
        self.encoders = nn.ModuleList(
            nn.Sequential(
                *_conv2d_bn_lrelu(c_in, c_out, k, 2, bias, padding_mode),
                *_conv2d_bn_lrelu(c_out, c_out, k, 1, bias, padding_mode),
            )
            for c_in, c_out, k in zip(encoder_in, down_channels, down_kernel_size)
        )
        self.skip_convs = nn.ModuleList(
            nn.Sequential(
                *_conv2d_bn_lrelu(c_in, c_out, skip_kernel_size, 1, bias, padding_mode)
            )
            for c_in, c_out in zip(encoder_in, skip_channels)
        )
        self.upsamplers = nn.ModuleList(
            nn.Upsample(scale_factor=2, mode=mode) for mode in upsample_mode
        )
        self.concat_bns = nn.ModuleList(nn.BatchNorm2d(c) for c in concat_channels)
        self.decoders = nn.ModuleList(
            self._decoder(c_in, c_out, k, up_conv1x1, bias, padding_mode)
            for c_in, c_out, k in zip(concat_channels, up_channels, up_kernel_size)
        )

        self.final = _conv2d(
            up_channels[0], out_channels, 1, bias=bias, padding_mode=padding_mode
        )
        if sigmoid_output:
            self.final = nn.Sequential(self.final, nn.Sigmoid())

    @staticmethod
    def _decoder(
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        conv1x1: bool,
        bias: bool,
        padding_mode: Literal["zeros", "reflect"],
    ) -> nn.Sequential:
        layers = _conv2d_bn_lrelu(
            in_channels, out_channels, kernel_size, 1, bias, padding_mode
        )
        if conv1x1:
            layers += _conv2d_bn_lrelu(
                out_channels, out_channels, 1, 1, bias, padding_mode
            )
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        # Skip features branch off before each downsampling step.
        skips: list[Tensor] = []
        for skip_conv, encoder in zip(self.skip_convs, self.encoders):
            skips.append(skip_conv(x))
            x = encoder(x)

        for i in range(self.num_scales - 1, -1, -1):
            x = self.upsamplers[i](x)
            skip = skips[i]
            # Odd dimensions can leave the two branches one pixel apart.
            if x.shape[2:] != skip.shape[2:]:
                height = min(x.size(2), skip.size(2))
                width = min(x.size(3), skip.size(3))
                x = _center_crop(x, height, width)
                skip = _center_crop(skip, height, width)
            x = torch.cat([x, skip], dim=1)
            x = self.concat_bns[i](x)
            x = self.decoders[i](x)

        return self.final(x)
