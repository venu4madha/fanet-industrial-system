"""
Memory Bank module for FANet anomaly detection.
Stores embeddings of normal images using coreset subsampling to reduce
memory usage while preserving representative diversity.
"""
import os
import numpy as np
import torch
import logging
from pathlib import Path
from scipy.spatial.distance import cdist

logger = logging.getLogger(__name__)


class CoreSetSampler:
    """Greedy coreset subsampling for memory bank compression."""

    def __init__(self, ratio=0.1):
        self.ratio = ratio

    def sample(self, embeddings: np.ndarray) -> np.ndarray:
        n = len(embeddings)
        target_n = max(1, int(n * self.ratio))
        if target_n >= n:
            return embeddings

        logger.info(f"CoreSet: {n} -> {target_n} samples (ratio={self.ratio})")
        selected_indices = [0]
        min_distances = np.full(n, np.inf)

        for _ in range(target_n - 1):
            last = embeddings[selected_indices[-1]]
            dists = np.linalg.norm(embeddings - last, axis=1)
            min_distances = np.minimum(min_distances, dists)
            next_idx = int(np.argmax(min_distances))
            selected_indices.append(next_idx)
            min_distances[next_idx] = 0.0

        return embeddings[selected_indices]


class MemoryBank:
    """
    Stores and queries a bank of normal-image feature embeddings.
    Used for nearest-neighbour anomaly scoring at inference time.
    """

    def __init__(self, feature_dim=512, coreset_ratio=0.1):
        self.feature_dim = feature_dim
        self.coreset_ratio = coreset_ratio
        self.bank: np.ndarray | None = None
        self.sampler = CoreSetSampler(ratio=coreset_ratio)
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None

    def add_embeddings(self, embeddings: torch.Tensor):
        arr = embeddings.detach().cpu().numpy()
        if self.bank is None:
            self.bank = arr
        else:
            self.bank = np.vstack([self.bank, arr])

    def compress(self):
        if self.bank is not None and len(self.bank) > 5:
            self.bank = self.sampler.sample(self.bank)
            self._mean = self.bank.mean(axis=0)
            self._std = self.bank.std(axis=0) + 1e-8
            
            # Calculate internal distance stats for robust Z-score normalization
            all_dists = cdist(self.bank, self.bank, metric='euclidean')
            np.fill_diagonal(all_dists, np.inf)
            min_dists = all_dists.min(axis=1)
            self._ref_mean = float(np.mean(min_dists))
            self._ref_std = float(np.std(min_dists)) + 1e-8
            
            logger.info(f"Memory bank compressed to {len(self.bank)} entries. Ref Mean: {self._ref_mean:.4f}")

    def compute_anomaly_score(self, embedding: torch.Tensor, k: int = 5) -> dict:
        """
        Compute anomaly score using Z-score normalization against the normal bank distribution.
        """
        if self.bank is None or len(self.bank) == 0:
            return {'score': 0.0, 'normalised': 0.0, 'top_k': []}

        vec = embedding.detach().cpu().numpy()
        if vec.ndim == 1:
            vec = vec[None, :]

        # 1. Raw distance (mean of k-nearest neighbours)
        dists = cdist(vec, self.bank, metric='euclidean')[0]
        k = min(k, len(dists))
        top_k_dists = np.sort(dists)[:k].tolist()
        score = float(np.mean(top_k_dists))

        # 2. Z-score Normalization
        # Use persisted stats if available, otherwise fallback to reasonable defaults
        ref_mean = getattr(self, '_ref_mean', 0.5) 
        ref_std = getattr(self, '_ref_std', 0.5)
        
        z_score = (score - ref_mean) / (ref_std + 1e-8)
        
        # Map Z-score to 0-1 range (more conservative mapping: 5 sigma = 0.5)
        # This makes the system much more robust to minor texture variations.
        normalised = max(0.0, z_score / 10.0) 
        normalised = min(1.0, normalised)

        return {
            'score': score,
            'normalised': round(normalised, 6),
            'z_score': round(z_score, 4),
            'top_k': [round(d, 6) for d in top_k_dists],
        }

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, {
            'bank': self.bank,
            'mean': self._mean,
            'std': self._std,
            'ref_mean': getattr(self, '_ref_mean', 0.0),
            'ref_std': getattr(self, '_ref_std', 1.0),
            'feature_dim': self.feature_dim,
            'coreset_ratio': self.coreset_ratio,
        })
        logger.info(f"Memory bank saved to {path}")

    def load(self, path: str):
        data = np.load(path, allow_pickle=True).item()
        self.bank = data['bank']
        self._mean = data.get('mean')
        self._std = data.get('std')
        self._ref_mean = data.get('ref_mean', 0.0)
        self._ref_std = data.get('ref_std', 1.0)
        self.feature_dim = data.get('feature_dim', self.feature_dim)
        logger.info(f"Memory bank loaded. Bank Size: {len(self.bank)}, Ref Mean: {self._ref_mean:.4f}")

    @property
    def size(self):
        return len(self.bank) if self.bank is not None else 0

    def is_ready(self):
        return self.bank is not None and len(self.bank) > 0
