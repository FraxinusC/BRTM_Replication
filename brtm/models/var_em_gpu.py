"""GPU‑accelerated variational EM with fixed topic sets."""
from __future__ import annotations
import torch, numpy as np
from typing import List, Dict, Tuple
from tqdm import tqdm

__all__ = ["var_em_gpu_fixed"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def var_em_gpu_fixed(doc_data, doc_lengths, kept_topics, phi_init, lda_model, bow_docs, domain_name, CFG) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fixed GPU-based Variational EM Algorithm
    """
    K, V = len(kept_topics), phi_init.shape[1]
    D = len(doc_data)

    print(f"Training domain {domain_name}: K={K}, V={V}, D={D}")

    # Initialize φ (topic-word distribution)
    phi = torch.tensor(phi_init, device=DEVICE, dtype=torch.float32) + 1e-8
    phi /= phi.sum(1, keepdim=True)

    # Initialize γ (document-topic distribution)
    gamma_init = torch.tensor([
        [p for _, p in lda_model.get_document_topics(bow, minimum_probability=0.0)]
        for bow in bow_docs
    ], device=DEVICE, dtype=torch.float32)[:, kept_topics]

    # Key fix: ensure γ has a reasonable scale
    gamma = gamma_init * 10 + CFG["alpha_k"]

    if domain_name == "D":  # Debug info for the first domain only
        print(f"Initial gamma stats - min:{gamma.min():.3f}, max:{gamma.max():.3f}, mean:{gamma.mean():.3f}")

    prev_log_likelihood = -np.inf

    # Outer loop: EM algorithm
    for outer in range(CFG["em_outer"]):
        epoch_start_gamma = gamma.clone()

        # Inner loop: Variational inference
        for inner in tqdm(range(CFG["em_inner"]), desc=f"{domain_name} E-step", leave=False):
            # Reset sufficient statistics
            sstats = torch.zeros_like(phi)

            # Precompute log φ
            log_phi = torch.log(phi + 1e-10)

            # Process each document
            for did in range(D):
                if not doc_data[did]:  # Skip empty docs
                    continue

                word_ids = list(doc_data[did].keys())
                word_counts = list(doc_data[did].values())

                if not word_ids:
                    continue

                word_ids = torch.tensor(word_ids, device=DEVICE, dtype=torch.long)
                word_counts = torch.tensor(word_counts, device=DEVICE, dtype=torch.float32)

                # E[log θ] = ψ(γ) - ψ(Σγ)
                dig_gamma = torch.digamma(gamma[did])
                dig_gamma_sum = torch.digamma(gamma[did].sum())
                expected_log_theta = dig_gamma - dig_gamma_sum

                # Variational posterior q(z_dn)
                log_prob = expected_log_theta[:, None] + log_phi[:, word_ids]  # (K, |unique_words|)
                prob = torch.softmax(log_prob, dim=0)

                # Weighted sufficient statistics
                weighted_prob = prob * word_counts[None, :]  # (K, |unique_words|)

                # Update sufficient statistics
                for k in range(K):
                    sstats[k].scatter_add_(0, word_ids, weighted_prob[k])

                # Update γ for the document
                gamma[did] = weighted_prob.sum(1) + CFG["alpha_k"]

            # M-step: update φ
            phi = sstats + CFG["eta"]
            phi /= phi.sum(1, keepdim=True)

            # Inner loop convergence check
            if inner > 0 and inner % 10 == 0:
                gamma_change = torch.norm(gamma - epoch_start_gamma).item()
                if gamma_change < CFG["conv_thresh"]:
                    if domain_name == "D":
                        print(f"Inner loop converged at iteration {inner+1}")
                    break

        # Outer loop convergence check
        epoch_gamma_change = torch.norm(gamma - epoch_start_gamma).item()

        # Approximate log-likelihood computation (sampled subset of documents)
        log_likelihood = 0.0
        sample_size = min(1000, D)
        for did in range(sample_size):
            if not doc_data[did]:
                continue
            word_ids = list(doc_data[did].keys())
            word_counts = list(doc_data[did].values())
            if word_ids:
                word_ids = torch.tensor(word_ids, device=DEVICE)
                word_counts = torch.tensor(word_counts, device=DEVICE, dtype=torch.float32)

                # θ = γ / Σγ
                theta = gamma[did] / gamma[did].sum()
                # p(w|d) = Σ_k θ_k * φ_{k,w}
                word_probs = torch.sum(theta[:, None] * phi[:, word_ids], dim=0)
                # log p(d) = Σ_n count_n * log p(w_n|d)
                doc_likelihood = torch.sum(word_counts * torch.log(word_probs + 1e-10))
                log_likelihood += doc_likelihood.item()

        if domain_name == "D":
            print(f"Outer {outer+1}: γ change={epoch_gamma_change:.6f}, log-likelihood={log_likelihood:.2f}")
            print(f"φ range=[{phi.min():.6f}, {phi.max():.6f}], γ range=[{gamma.min():.2f}, {gamma.max():.2f}]")

        # Check convergence of the outer loop
        if epoch_gamma_change < CFG["conv_thresh"] * 10:
            if domain_name == "D":
                print(f"{domain_name} domain converged in outer loop.")
            break

    # Return normalized θ and final φ
    theta_mat = (gamma / gamma.sum(1, keepdim=True)).cpu().numpy()
    phi_final = phi.detach().cpu().numpy()

    return theta_mat, None, phi_final
