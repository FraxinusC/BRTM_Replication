"""I/O helpers for textual & profile CSVs and transaction splits."""
from __future__ import annotations
import json, re
from pathlib import Path
from typing import Dict, Tuple, List
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

STOP = set(stopwords.words("english"))

def tok(text: str) -> List[str]:
    """Lowercase tokenisation + stop‑word filtering (alpha tokens only)."""
    return [w for w in word_tokenize(str(text).lower()) if re.fullmatch(r"[a-z]+", w) and w not in STOP]



def preprocess_docs(docs, dictionary) -> Tuple[List[Dict[int, int]], List[int]]:
    """Convert documents into word frequency format for efficient training"""
    doc_data = []
    doc_lengths = []

    for d in docs:
        word_counts = {}
        doc_len = 0
        for w in d:
            if w in dictionary.token2id:
                wid = dictionary.token2id[w]
                word_counts[wid] = word_counts.get(wid, 0) + 1
                doc_len += 1
        doc_data.append(word_counts)
        doc_lengths.append(doc_len)

    return doc_data, doc_lengths

def make_guest_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a new DataFrame with only 4 numeric columns; index remains guest_id"""
    out = pd.DataFrame(index=df.index)

    # 1) Years since account joined
    now_year = pd.Timestamp.today().year
    out["years_since_join"] = now_year - df["join_year"].fillna(now_year)

    # 2) Number of verified sources
    out["n_verified_src"] = df["verifiedTypes"].fillna("").apply(
        lambda s: len([x for x in s.split(",") if x.strip()])
    )

    # 3) Number of connected accounts
    out["n_connected"] = df["facebookConnected"].fillna(False).astype(int)

    return out.astype(float)

def make_host_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return host profile features"""
    now_year = pd.Timestamp.today().year
    out = pd.DataFrame(index=df.index)

    # 1) Years since account joined
    out["years_since_join"] = now_year - df["host_join_year"].fillna(now_year)

    # 2) Number of verified sources
    out["n_verified_src"] = df["num_verified_sources"].fillna(0).astype(float)

    # 3) Is Superhost (binary)
    out["is_superhost"] = df["is_superhost"].fillna(False).astype(int)

    return out.astype(float)