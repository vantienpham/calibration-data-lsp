"""Learned subspace projections for low-rank compression of decoder language models.

The pipeline follows Learnable Subspace Projections (Bini et al., 2026): every
compressed unit (one linear layer, or a group of layers reading the same input)
removes a subspace with an orthogonal projector; projectors start from a
whitened truncation and are trained against the dense model's output
distribution with the pretrained weights frozen; trained projectors merge into
plain low-rank factors.

Modules:
    arch      which linear layers form units, and what each costs in parameters
    data      calibration and evaluation corpora
    stats     input / output Gram matrices of every unit
    init      whitened ordered bases (the training-free initialization)
    alloc     uniform and measured-KL rank allocation
    project   projector modules, the projector bank, and merging into factors
    train     KL distillation of the projectors (or of free factors)
    evaluate  perplexity and zero-shot accuracy
    selfgen   calibration text sampled from the dense model
    hub       pinned Hugging Face revisions of every model and corpus
    utils     run directories, provenance, small I/O helpers
"""

__version__ = "0.1.0"
