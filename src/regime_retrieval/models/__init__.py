"""Frozen latent-state models with causal filtering and auditable fit failures."""

from .hmm import HMMFitError, LatentStateModel

__all__ = ["HMMFitError", "LatentStateModel"]
