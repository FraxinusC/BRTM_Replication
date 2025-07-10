"""Centralised configuration, overridable via CLI or env‑vars."""
from pathlib import Path
CFG = {
    # --- paths (根据实际数据调整) ---
    "files": {
        # 文本数据
        "D_j"  : (r"C:\Users\fraxi\OneDrive\Desktop\code task\Data\dj_documents_unique.csv", "document", "listing_id"),
        "D_li" : (r"C:\Users\fraxi\OneDrive\Desktop\code task\Data\Dli_by_guest.csv", "doc_text", "guest_id"),
        "A_i"  : (r"C:\Users\fraxi\OneDrive\Desktop\code task\Data\A(,i)_reviews.csv", "document", "reviewer_id"),
        "A_j"  : (r"C:\Users\fraxi\OneDrive\Desktop\code task\Data\A(·,j)_listing_reviews.csv", "doc_text", "listing_id"),
        "B_i"  : (r"C:\Users\fraxi\OneDrive\Desktop\code task\Data\B(·,i)_guest_reviews_received.csv", "translation.comments", "guest_id"),
        "B_k"  : (r"C:\Users\fraxi\OneDrive\Desktop\code task\Data\B(k,·)_host_reviews_written.csv", "doc_text", "host_id"),
        # Profile数据
        "C_i"  : (r"C:\Users\fraxi\OneDrive\Desktop\code task\Data\Ci_open_guest_basic_info.csv", None, "guest_id"),
        "C_k"  : (r"C:\Users\fraxi\OneDrive\Desktop\code task\Data\Ck_amsterdam_host_profiles.csv", None, "host_id"),
        # 交易数据
        "train": (r"C:\Users\fraxi\OneDrive\Desktop\code task\Data\transaction_train.csv", None, None),
        "val"  : (r"C:\Users\fraxi\OneDrive\Desktop\code task\Data\transaction_val.csv", None, None),
        "test" : (r"C:\Users\fraxi\OneDrive\Desktop\code task\Data\transaction_test.csv", None, None),
    },
    # --- topics ---
    "K":60,"Dstar":38,"DA":22,"Astar":29,"AB":9,"Bstar":51,
    # --- variational EM (修正后的超参数) ---
    "alpha_k":0.1,"eta":0.01,"lda_pass":10,
    "em_inner":40,"em_outer":5,"gpu_batch":16384,
    "conv_thresh":1e-5,
    # --- LBFGS ---
    "lbfgs_iter":120,"l2_lambda":1e-3,
    # --- eval ---
    "topN":10,"neg_k":19,
    "seed":42,
    "device": "cuda" if Path("/dev/nvidia0").exists() else "cpu",
}