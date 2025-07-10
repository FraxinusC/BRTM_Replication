from brtm.utils.features import XY
import numpy as np, pandas as pd
from tqdm import tqdm



def calculate_mrr_ndcg(model_beta, test_df, CFG ,θD, θA, θB, guest_p, host_p, order_D, order_A, order_B):
    """Compute MRR and NDCG@10"""
    X_test, y_test = XY(test_df,θD, θA, θB, guest_p, host_p, order_D, order_A, order_B)
    scores = 1.0 / (1.0 + np.exp(-X_test.dot(model_beta)))
    
    mrr_scores = []
    ndcg_scores = []

    # Randomly select positive samples for evaluation
    pos_indices = np.where(y_test == 1)[0]
    neg_indices = np.where(y_test == 0)[0]
    
    np.random.seed(CFG["seed"])
    eval_size = min(1000, len(pos_indices))
    eval_pos_indices = np.random.choice(pos_indices, eval_size, replace=False)
    
    for pos_idx in tqdm(eval_pos_indices, desc="Computing MRR & NDCG"):
        # Sample negative examples
        selected_neg = np.random.choice(neg_indices, CFG["neg_k"], replace=False)
        candidate_indices = np.append(selected_neg, pos_idx)
        candidate_scores = scores[candidate_indices]
        
        # Sort by predicted scores
        sorted_indices = np.argsort(candidate_scores)[::-1]
        
        # Get rank of the positive sample
        pos_rank = np.where(sorted_indices == len(candidate_indices) - 1)[0][0] + 1
        
        # MRR
        mrr_scores.append(1.0 / pos_rank)
        
        # NDCG@10
        if pos_rank <= 10:
            ndcg_scores.append(1.0 / np.log2(pos_rank + 1))
        else:
            ndcg_scores.append(0.0)
    
    return np.mean(mrr_scores), np.mean(ndcg_scores)