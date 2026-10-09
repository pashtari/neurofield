# Concepts

Five pages explain the library's design, from the abstraction every model shares to the conventions that make them interchangeable.

- [Neural fields in NeuroField](neural-fields.md): a model is a function of coordinates, a dataset is a set of coordinate/value pairs, and `nf.train` ties them together. Read this to understand what the training loop does and what it returns.
- [FUTON](futon.md): the Fourier Tensor Network, stage by stage: bases, combiners, decoders, the sparse path for local bases, and how to size a model.
- [Error bound of FUTON](error-bound.md): how close a FUTON with a linear decoder can come to a signal at a given spectral resolution and rank, exactly and in computable bounds, with the proof and an experiment.
- [The model zoo](models.md): the twelve architectures, their families, references and constructor arguments.
- [Conventions](conventions.md): coordinate and value ranges, tensor shapes, metrics, checkpoints and seeds.
