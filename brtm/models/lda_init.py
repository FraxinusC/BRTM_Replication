"""Utility wrappers around gensim LDA used for β initialisation."""
from __future__ import annotations
from typing import List, Tuple
import numpy as np
from gensim import corpora, models
from gensim.models.ldamodel import LdaModel
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

def init_lda_beta(docs, dictionary: corpora.Dictionary, CFG: dict) -> Tuple[np.ndarray, LdaModel, List[List[Tuple[int, int]]]]:
    """Use Gensim LDA to obtain the initial topic-word distribution"""
    bow = [dictionary.doc2bow(d) for d in tqdm(docs, desc="Building BOW")]

    print("Training initial LDA...")
    lda = LdaModel(
        bow,
        id2word=dictionary,
        num_topics=CFG["K"],
        passes=CFG["lda_pass"],
        alpha="auto",
        eta=CFG["eta"],
        random_state=CFG["seed"]
    )

    # Get topic-word distribution
    beta = lda.get_topics()  # Shape: (K, V)
    return beta, lda, bow

def shared_topics(b1, b2, n):
    """Find shared topics between two topic spaces"""
    print("Matching shared topics...")
    sim = 1 - cdist(b1, b2, "cosine")
    r, c = linear_sum_assignment(-sim)
    idx = np.argsort(-sim[r, c])[:n]
    return [r[i] for i in idx], [c[i] for i in idx]