"""Evaluation and baseline implementations."""
import numpy as np, pandas as pd
from tqdm import tqdm
from typing import List, Dict
from brtm.utils.features import XY
from brtm.config import CFG
# --- Hit‑Rate & ranking helpers -------------------------------------------

def evaluate_topn(model_beta, test_df, CFG, topn_list=[1, 5, 10]) -> Dict[str, float]:
    """Evaluate Top-N recommendation performance"""
    results = {}
    
    # Construct test features
    print("Building test features...")
    X_test, y_test = XY(test_df)
    
    # Predict scores using sigmoid(logistic) function
    scores = 1.0 / (1.0 + np.exp(-X_test.dot(model_beta)))
    
    # Simplified evaluation: randomly sample negative examples for ranking
    np.random.seed(CFG["seed"])
    n_eval = min(10000, len(test_df))
    eval_indices = np.random.choice(len(test_df), n_eval, replace=False)
    
    # Compute Hit Rate
    for n in topn_list:
        hits = 0
        for idx in eval_indices:
            if y_test[idx] == 1:  # Positive sample
                # Randomly select negative samples for comparison
                neg_indices = np.random.choice(
                    np.where(y_test == 0)[0], 
                    min(CFG["neg_k"], np.sum(y_test == 0)), 
                    replace=False
                )
                candidate_indices = np.append(neg_indices, idx)
                candidate_scores = scores[candidate_indices]
                
                # Sort by score
                sorted_indices = np.argsort(candidate_scores)[::-1]
                
                # Check if the positive sample is in top-N
                pos_rank = np.where(sorted_indices == len(candidate_indices) - 1)[0][0]
                if pos_rank < n:
                    hits += 1
        
        hit_rate = hits / np.sum(y_test[eval_indices] == 1)
        results[f'HR@{n}'] = hit_rate
        print(f"Hit Rate @ {n}: {hit_rate:.3f}")
    
    return results
