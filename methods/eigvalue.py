import numpy as np
from .base_method import UncertaintyMethod


def get_L_mat(W, symmetric=True):

    D = np.diag(np.sum(W, axis=1))
    

    d_inv_sqrt = np.zeros_like(D, dtype=float)
    non_zero_mask = np.diag(D) > 1e-12
    d_inv_sqrt[np.diag_indices_from(D)[0][non_zero_mask], np.diag_indices_from(D)[1][non_zero_mask]] = 1.0 / np.sqrt(np.diag(D)[non_zero_mask])
    
    if symmetric:
        L = d_inv_sqrt @ (D - W) @ d_inv_sqrt
    else:
        raise NotImplementedError()
    return L.copy()

def get_eig(L, thres=None, eps=1e-4):
    if eps is not None:
        L = (1-eps) * L + eps * np.eye(len(L))
    eigvals, eigvecs = np.linalg.eigh(L)
    
    if thres is not None:
        keep_mask = eigvals < thres
        eigvals, eigvecs = eigvals[keep_mask], eigvecs[:, keep_mask]
    return eigvals, eigvecs

class EigValue(UncertaintyMethod):
    name = "eigvalue"
    requires_data = ["sampling_stats", "similarity_matrix_stats"]
    tags = ["sampling", "semantic"]

    def __init__(self, adjust: bool = True):
        self.adjust = adjust

    def _compute(self, stats):
        sim_stats = stats["similarity_matrix_stats"]
        w = np.asarray(sim_stats.get("W", []), dtype=np.float32)

        if w.ndim != 2 or w.shape[0] < 2:
            return {"u": 0.0}

        L = get_L_mat(w, symmetric=True)
        eigvals, _ = get_eig(L)
        
        spectral_eigvs = 1.0 - eigvals
        if self.adjust:
            spectral_eigvs = np.clip(spectral_eigvs, 0, None)
        else:
            spectral_eigvs = np.clip(spectral_eigvs, -1, None)
            
        u = np.sum(spectral_eigvs)
        
        return {"u": float(u)}
