import numpy as np
import torch
from .base_method import UncertaintyMethod

def get_L_mat(W, symmetric=True):
    # compute the degreee matrix from the weighted adjacency matrix
    D = np.diag(np.sum(W, axis=1))
    
    # Handle single sample or disconnected components producing zeros in D
    d_inv_sqrt = np.zeros_like(D, dtype=float)
    non_zero_mask = np.diag(D) > 1e-12
    d_inv_sqrt[np.diag_indices_from(D)[0][non_zero_mask], np.diag_indices_from(D)[1][non_zero_mask]] = 1.0 / np.sqrt(np.diag(D)[non_zero_mask])
    
    # compute the normalized laplacian matrix from the degree matrix and weighted adjacency matrix
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

class Eccentricity(UncertaintyMethod):
    name = "eccentricity"
    requires_data = ["sampling_stats", "similarity_matrix_stats"]
    tags = ["sampling", "semantic"]

    def __init__(self, eigv_threshold: float = 0.9):
        self.eigv_threshold = eigv_threshold

    def _compute(self, stats):
        sim_stats = stats["similarity_matrix_stats"]
        w = np.asarray(sim_stats.get("W", []), dtype=np.float32)
        
        if w.ndim != 2 or w.shape[0] < 2:
            return {"u": 0.0}


        L = get_L_mat(w, symmetric=True)
        eigvals, eigvecs = get_eig(L, thres=self.eigv_threshold)
        

        if eigvecs.size == 0 or eigvecs.shape[1] == 0:
            return {"u": 0.0}
            
        center = np.mean(eigvecs, axis=0)
        distances = np.linalg.norm(eigvecs - center, ord=2, axis=1)
        u = np.linalg.norm(distances, ord=2)
        
        return {"u": float(u)}
