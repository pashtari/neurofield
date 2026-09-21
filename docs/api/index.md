# API reference

The public API lives in three namespaces. Everything for fitting coordinate/value pairs is exported at the top level; image compression and radiance fields have their own.

| Namespace | Contents |
| --- | --- |
| `neurofield` (`nf`) | Models, datasets, `train`, `evaluate`, `chunked_inference`, losses, metrics, quantization, plotting and utilities |
| `nf.compression` | `encode_futon`, `decode_futon`, `encode_pil`, `decode_pil`, `evaluate_compression`, arithmetic coding |
| `nf.nerf` | `BlenderDataset`, `RadianceField`, renderers, `train`, `evaluate`, `render_image`, `render_views`, camera helpers |

The pages of this section are organized by module. Each documents the public classes and functions with their signatures, tensor shapes and defaults, as written in the source.

- [neurofield.models](models.md): every architecture except FUTON, with its layers and encodings.
- [FUTON, bases and combiners](futon.md): `FUTON`, the six bases, the three combiners, and the sparse `RCSMatrix`.
- [Datasets](datasets.md): coordinate datasets for images, volumes and MRI, and the Deep Image Prior dataset.
- [Training and evaluation](training.md): `train`, `train_epoch`, `evaluate`, `chunked_inference`.
- [Metrics and losses](metrics.md): PSNR, SSIM, MS-SSIM, LPIPS, NMSE, IoU, bits per pixel; rate-distortion, SDF and entropy losses.
- [Quantization](quantization.md): uniform quantization of a model's parameters.
- [Compression](compression.md): the codecs, their evaluation, and arithmetic coding.
- [Radiance fields](nerf.md): the `nf.nerf` package.
- [Utilities](utils.md): module specs, parameter counts, logging, serialization, and the seaborn line plot.

!!! tip "Docstring conventions"
    Shapes are written as `(N, C)` for a flat batch and `(*, C)` or `(..., C)` when any leading dimensions are accepted. `C` is the number of coordinate axes, `K` or `K_c` the number of basis functions per axis, `R` a rank, and `D` the number of output channels.
