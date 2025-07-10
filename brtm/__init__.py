"""BRTM package top‑level namespace."""
from importlib.metadata import version
from .config import CFG
from .models.var_em_gpu import var_em_gpu_fixed
from .models.lda_init import init_lda_beta, shared_topics
from .models.lbfgs_gpu import lbfgs_gpu
from .evaluation import evaluate_topn, calculate_mrr_ndcg
from .utils import features
from .data import loaders
__all__ = ["CFG", "var_em_gpu_fixed", "lbfgs_gpu", "init_lda_beta", "shared_topics", "evaluate_topn", "calculate_mrr_ndcg", "features", "loaders", "version"]
