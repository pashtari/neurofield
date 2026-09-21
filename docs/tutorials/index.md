# Tutorials

Each tutorial is a complete, runnable workflow for one task, using the same settings as the benchmarks.

| Tutorial | What you build | Needs |
| --- | --- | --- |
| [Fitting an image](image.md) | FUTON and the baselines on a Kodak image, with learning curves | an image |
| [Occupancy volumes](occupancy.md) | A 3D shape as a field on $[-1, 1]^3$, scored by IoU, exported as a mesh | a mesh and the `3d` extra |
| [Radiance fields](nerf.md) | Any model as the density network of a NeRF, trained on posed views, rendered in orbit | a Blender scene |
| [Image compression](compression.md) | FUTON as a codec: rate-distortion training, quantization, bits per pixel | an image |
| [Custom components](custom-components.md) | Your own basis, combiner, decoder, loss, metric and dataset | nothing |
| [Notebooks](notebooks.md) | The three benchmark notebooks and how to run them | the data |
