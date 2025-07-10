"""Plain‑vanilla logistic‑loss LBFGS optimiser on GPU."""
import torch, numpy as np
__all__ = ["lbfgs_gpu"]


def lbfgs_gpu(X: np.ndarray, y: np.ndarray, cfg, w0=None, device="cuda"):
    Xt = torch.tensor(X, device=device, dtype=torch.float32)
    yt = torch.tensor(y, device=device, dtype=torch.float32)
    w = torch.zeros(X.shape[1], device=device, dtype=torch.float32) if w0 is None else torch.tensor(w0, device=device, dtype=torch.float32)
    w.requires_grad_(True)
    opt = torch.optim.LBFGS([w], max_iter=cfg["lbfgs_iter"], line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(Xt.mv(w), yt)
        loss += cfg["l2_lambda"] * (w**2).sum()
        loss.backward(); return loss
    opt.step(closure)
    return w.detach().cpu().numpy()