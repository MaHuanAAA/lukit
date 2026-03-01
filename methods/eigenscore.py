import numpy as np
import torch

from .base_method import UncertaintyMethod


class EigenScore(UncertaintyMethod):
    name = "eigenscore"
    requires_data = ["sampling_stats"]
    tags = ["sampling", "representation"]

    def __init__(self, alpha: float = 1e-3) -> None:
        self.alpha = float(alpha)

    def _compute(self, stats):
        sampling_stats = stats["sampling_stats"]
        embeddings = sampling_stats.get("eigenscore_embeddings", [])
        h = np.asarray(embeddings, dtype=np.float64)

        if h.ndim != 2 or h.shape[0] == 0 or h.shape[1] < 2:
            return {"u": 0.0}

        # Match original implementation idea: rows are samples (N), cols are features (d),
        # then torch.cov(H) gives sample covariance in sample space (N x N).
        h_tensor = torch.from_numpy(h)
        cov = torch.cov(h_tensor)
        if cov.ndim == 0:
            cov = cov.reshape(1, 1)
        n = int(cov.shape[0])
        cov = cov + self.alpha * torch.eye(n, dtype=cov.dtype, device=cov.device)

        eigvals = torch.linalg.eigvalsh(cov).real
        eigvals = torch.clamp(eigvals, min=1e-12)
        score = torch.mean(torch.log10(eigvals))
        return {"u": float(score.item())}
