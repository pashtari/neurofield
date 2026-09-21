# The model zoo

Twelve architectures share one interface: `Model(in_features, out_features, ...)` maps coordinates `(..., in_features)` in $[-1, 1]$ to `(..., out_features)`. They are re-implementations under one training loop, at settings taken from the authors' code, and the [benchmarks](../experiments/benchmarks.md) compare them at equal parameter counts.

## Coordinate networks

MLPs whose prior sits in the first layer or in the activation.

| Model | Class | Idea | Reference |
| --- | --- | --- | --- |
| RFF | `RFF` | Random Fourier features $[\sin(2\pi \boldsymbol{B}\boldsymbol{x}), \cos(2\pi \boldsymbol{B}\boldsymbol{x})]$, $\boldsymbol{B} \sim \mathcal{N}(0, \sigma^2)$, then a ReLU MLP | Tancik et al., NeurIPS 2020 |
| PE-MLP | `PEMLP` | NeRF's positional encoding $[\boldsymbol{x}, \sin(2^k \boldsymbol{x}), \cos(2^k \boldsymbol{x})]$, then a ReLU MLP | Mildenhall et al., ECCV 2020 |
| SIREN | `SIREN` | Sine layers $\sin(\omega \, \boldsymbol{W}\boldsymbol{x} + \boldsymbol{b})$ with SIREN initialization | Sitzmann et al., NeurIPS 2020 |
| Gauss | `Gauss` | Gaussian layers $\exp(-(s \, \boldsymbol{W}\boldsymbol{x} + \boldsymbol{b})^2)$ | Ramasinghe and Lucey, ECCV 2022 |
| WIRE | `WIRE`, `RealWIRE` | Complex, or real, Gabor wavelet layers $\cos(\omega z_1) \exp(-(s z_2)^2)$ | Saragadam et al., CVPR 2023 |
| FINER | `FINER` | Variable-periodic sines $\sin(\omega (\lvert z \rvert + 1) z)$ | Liu et al., CVPR 2024 |
| MFN | `MFN` | Multiplicative Gabor filter network: each stage multiplies a Gabor filter of the coordinates into the hidden state | Fathony et al., ICLR 2021 |

`MLP` underlies most of them: a plain ReLU network by default, or a network of custom layers (`layer_class=nf.SineLayer`, `nf.GaussLayer`, `nf.FinerLayer`, `nf.RealGaborLayer`) with their own activations. `WIRE` is the complex-valued network of the paper; the experiments use `RealWIRE`, the real Gabor variant from the same paper and code, which trains about 1.5× faster at equal parameter count and has real weights, so it also quantizes.

## Grid and tensor models

Learnable feature grids, read by interpolation and decoded by a ReLU MLP.

| Model | Class | Idea | Reference |
| --- | --- | --- | --- |
| Instant-NGP | `InstantNGP`, `HashEncoding` | Multiresolution hash encoding (16 levels, 2 features per level by default) and a bias-free ReLU MLP, mirroring tiny-cuda-nn | Müller et al., SIGGRAPH 2022 |
| TensoRF | `TensoRF` | Factorized grids: CP (products of line grids) or VM (line-plane products, 3D) | Chen et al., ECCV 2022 |
| GA-Planes | `GAPlanes`, `MultiVector`, `FeatureGrid` | Line, plane and volume grids combined by an expression such as `"[e1 o e2, e12]"` | Sivgin et al., ICML 2025 |
| Grid INR | `GridINR` | Independent coordinate networks on a grid of overlapping patches, blended with smoothstep weights | |

`FeatureGrid` is the building block: a learnable grid of any dimension sampled with `align_corners=True` on $[-1, 1]$. `MultiVector` combines several with elementwise products, sums and concatenation, and `GAPlanes` and `TensoRF` add the decoder.

## FUTON

| Model | Class | Idea | Reference |
| --- | --- | --- | --- |
| FUTON | `FUTON` | A fixed basis per axis, a low-rank CP combiner across axes, and a linear or MLP decoder | Ashtari et al., arXiv 2026 |

The benchmarks run it with the sinc basis (`FUTON-sinc`) and the Lanczos basis (`FUTON-lanczos`). The [FUTON page](futon.md) explains every part.

## Deep Image Prior

| Model | Class | Idea | Reference |
| --- | --- | --- | --- |
| DIP | `DIPUNet`, `DIPSkip` | Convolutional networks that map a fixed noise image to the output; the architecture is the prior | Ulyanov et al., CVPR 2018 |

These are image-to-image networks, `(B, C, H, W)` in and out, paired with `DIPImageDataset`, which supplies the fixed noise input and re-adds a little noise on each access.

## Shared conventions

- **Widths and depths.** `hidden_features` is the hidden width and `hidden_layers` the number of hidden layers, so a network with `hidden_layers=h` has `h + 1` weight layers (MFN: `h` Gabor filters, `h - 1` hidden linear layers and one output layer). `WIRE` and `RealWIRE` divide the nominal width by $\sqrt{2}$ to keep the parameter count comparable to real networks.
- **Output activations.** Every model takes `output_activation`, a callable applied last. The benchmarks use `torch.tanh` for targets in $[-1, 1]$ and none inside a `RadianceField`.
- **Equal capacity.** `nf.count_parameters(model, trainable_only=False)` is how the configs were matched. The image models hold about 195k parameters, the occupancy models about 132k, and the NeRF density networks about 68k (75k with the shared colour network).
- **Learning rates.** Each model has its own, drawn from {3e-1, 1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4} and recorded in the configs with the cost of the neighbouring values.
