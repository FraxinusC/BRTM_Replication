"""Model helper re‑exports for a clean import path."""
from .lda_init import init_lda_beta, shared_topics
from .var_em_gpu import var_em_gpu_fixed
from .lbfgs_gpu import lbfgs_gpu

__all__ = [
    "init_lda_beta",
    "shared_topics",
    "var_em_gpu_fixed",
    "lbfgs_gpu",
]