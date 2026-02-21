"""gpuicalcc: PyTorch-accelerated locally centered cyclic contrasts.

GPU extension for ICALCC. Same API as sklearn FastICA:
just replace the import and set K.

    from gpuicalcc import GPUICALCC
    est = GPUICALCC(K='ltanh', device='cuda')
    est.fit(X)

Supported K values: 4, 6, 8, 'ltanh', 'lexp', 'tanh', 'exp', 'skew'.

Requirements: icalcc, torch

Reference: T. Saito, "Locally Centered Cyclic Kernels for Higher-Order
Independent Component Analysis," TechRxiv, 2026.
https://doi.org/10.36227/techrxiv.XXXXXXX
"""

import numpy as np
import warnings
import torch
from sklearn.exceptions import ConvergenceWarning
from icalcc import ICALCC


# -------------------------------------------------------------------
#  Polynomial LCC on GPU  (K = 4, 6, 8)
# -------------------------------------------------------------------

def _gpu_lcc_h_gprime(y_np, k, device="cuda"):
    """Polynomial LCC nonlinearity on GPU."""
    y = torch.as_tensor(y_np, dtype=torch.float64, device=device)
    y2 = y * y
    y3 = y2 * y

    if k == 4:
        gy = (-3.0 / 16) * y3
        gpy = (-9.0 / 16) * y2
        return gy.cpu().numpy(), gpy.cpu().numpy()

    m3 = y3.mean().item()
    m4 = (y2 * y2).mean().item()

    if k == 6:
        dJ3 = 145 * m3 / 1944.0
        dJ4 = 115.0 / 2592
        dJ6 = -5.0 / 7776
        y4 = y2 * y2
        gy = dJ3 * 3 * y2 + dJ4 * 4 * y3 + dJ6 * 6 * (y4 * y)
        gpy = dJ3 * 6 * y + dJ4 * 12 * y2 + dJ6 * 30 * y4
        return gy.cpu().numpy(), gpy.cpu().numpy()

    # k == 8
    m5 = (y2 * y3).mean().item()
    m6 = (y3 * y3).mean().item()
    dJ3 = -7665 * m3 / 65536.0 + 497 * m5 / 262144.0
    dJ4 = 2765 * m4 / 1048576.0 - 18795.0 / 524288
    dJ5 = 497 * m3 / 262144.0
    dJ6 = 329.0 / 524288
    dJ8 = -7.0 / 2097152
    y4 = y2 * y2
    gy = (dJ3 * 3 * y2 + dJ4 * 4 * y3
          + dJ5 * 5 * y4 + dJ6 * 6 * (y4 * y)
          + dJ8 * 8 * (y4 * y3))
    gpy = (dJ3 * 6 * y + dJ4 * 12 * y2
           + dJ5 * 20 * y3 + dJ6 * 30 * y4
           + dJ8 * 56 * (y3 * y3))
    return gy.cpu().numpy(), gpy.cpu().numpy()


def _gpu_lcc_contrast(x, K=6, device="cuda"):
    """Polynomial LCC contrast for sklearn fun parameter."""
    if x.ndim == 1:
        return _gpu_lcc_h_gprime(x, K, device=device)
    p, N = x.shape
    gY = np.empty_like(x)
    gpy = np.empty(p, dtype=x.dtype)
    for i in range(p):
        gi, gpi = _gpu_lcc_h_gprime(x[i], K, device=device)
        gY[i] = gi
        gpy[i] = gpi.mean()
    return gY, gpy


# -------------------------------------------------------------------
#  Bounded LCC on GPU  (K = 'ltanh', 'lexp')
# -------------------------------------------------------------------

def _get_gpu_mem_bytes(device, gpu_mem_limit):
    """Resolve effective GPU memory limit in bytes.

    Auto-detects free memory.  If gpu_mem_limit is set and exceeds
    97% of total, warns and caps at 97%.
    """
    if not device.startswith("cuda"):
        return 4 * 1024**3  # 4GB fallback for CPU
    free, total = torch.cuda.mem_get_info(device)
    safe = int(total * 0.97)
    if gpu_mem_limit is None:
        return min(free, safe)
    requested = int(gpu_mem_limit * 1024**3)
    if requested > safe:
        warnings.warn(
            f"gpu_mem_limit={gpu_mem_limit}GB exceeds 97% of "
            f"GPU memory ({total/1024**3:.1f}GB). "
            f"Capping at {safe/1024**3:.1f}GB.",
            RuntimeWarning)
        return safe
    return requested


def _gpu_bounded_contrast(x, G="tanh", batch_size=500,
                          device="cuda", gpu_mem_limit=None):
    """Pairwise bounded LCC contrast on GPU.

    Handles both deflation (1-D) and parallel (2-D).
    Auto-detects GPU memory; user can override via gpu_mem_limit (GB).
    Halves chunk size on OOM.
    """
    if x.ndim == 1:
        return _gpu_bounded_1d(x, G, batch_size, device)

    p, N = x.shape
    B = min(batch_size, N)
    step = max(1, N // B)
    idx = np.arange(0, N, step)[:B]

    mem = _get_gpu_mem_bytes(device, gpu_mem_limit)
    max_chunk = max(1, mem // (p * B * 8 * 6))

    X_batch = torch.as_tensor(x[:, idx], dtype=torch.float64,
                              device=device)
    gY = np.empty_like(x)
    gpY = np.empty_like(x)

    start = 0
    while start < N:
        end = min(start + max_chunk, N)
        try:
            X_chunk = torch.as_tensor(
                x[:, start:end], dtype=torch.float64, device=device)
            diff = X_chunk.unsqueeze(2) - X_batch.unsqueeze(1)

            if G == "tanh":
                t = torch.tanh(diff)
                gY[:, start:end] = t.mean(dim=2).cpu().numpy()
                gpY[:, start:end] = (1.0 - t * t).mean(
                    dim=2).cpu().numpy()
            else:  # exp
                e = torch.exp(-0.5 * diff * diff)
                gY[:, start:end] = (diff * e).mean(
                    dim=2).cpu().numpy()
                gpY[:, start:end] = ((1.0 - diff * diff) * e).mean(
                    dim=2).cpu().numpy()

            del diff, X_chunk
            torch.cuda.empty_cache()
            start = end

        except torch.cuda.OutOfMemoryError:
            del X_chunk
            torch.cuda.empty_cache()
            max_chunk = max(1, max_chunk // 2)

    del X_batch
    return gY, gpY.mean(axis=1)


def _gpu_bounded_1d(y_np, G, batch_size, device):
    """1-D bounded LCC for deflation algorithm."""
    N = len(y_np)
    B = min(batch_size, N)
    step = max(1, N // B)
    idx = np.arange(0, N, step)[:B]

    y = torch.as_tensor(y_np, dtype=torch.float64, device=device)
    y_batch = y[idx]
    diff = y.unsqueeze(1) - y_batch.unsqueeze(0)

    if G == "tanh":
        t = torch.tanh(diff)
        gy = t.mean(dim=1)
        gpy = (1.0 - t * t).mean(dim=1)
    else:
        e = torch.exp(-0.5 * diff * diff)
        gy = (diff * e).mean(dim=1)
        gpy = ((1.0 - diff * diff) * e).mean(dim=1)

    return gy.cpu().numpy(), gpy.cpu().numpy()


# -------------------------------------------------------------------
#  Main class
# -------------------------------------------------------------------

class GPUICALCC(ICALCC):
    """GPU-accelerated ICALCC.

    Accelerates all LCC contrasts (polynomial and bounded) via
    PyTorch.  Classical contrasts ('tanh', 'exp', 'skew') and
    FastICA(k) contrasts fall back to the parent class.

    Parameters
    ----------
    K : contrast selector (same as ICALCC)
    device : str, default='cuda'
        PyTorch device.  Falls back to 'cpu' if CUDA unavailable.
    batch_size : int, default=500
        Subsample size for bounded pairwise computation.
    **kwargs : passed to ICALCC

    Examples
    --------
    >>> from gpuicalcc import GPUICALCC
    >>> est = GPUICALCC(n_components=4, K='ltanh', device='cuda')
    >>> S_hat = est.fit_transform(X)
    >>> est6 = GPUICALCC(n_components=4, K=6, device='cuda')
    >>> S6 = est6.fit_transform(X)
    """

    def __init__(self, n_components=None, *, K=6, device="cuda",
                 batch_size=500, gpu_mem_limit=None,
                 algorithm="parallel",
                 whiten="unit-variance", max_iter=200, tol=1e-4,
                 w_init=None, whiten_solver="svd",
                 random_state=None, clear_gpu=True):

        if device.startswith("cuda") and not torch.cuda.is_available():
            warnings.warn(
                "CUDA not available, falling back to device='cpu'.",
                RuntimeWarning)
            device = "cpu"
        self.device = device
        self.batch_size = batch_size
        self.gpu_mem_limit = gpu_mem_limit
        self.clear_gpu = clear_gpu

        super().__init__(
            n_components=n_components, K=K,
            algorithm=algorithm, whiten=whiten,
            max_iter=max_iter, tol=tol, w_init=w_init,
            whiten_solver=whiten_solver, random_state=random_state)

        # override with GPU versions
        if K in self._LCC_BOUNDED_MAP:
            G = self._LCC_BOUNDED_MAP[K]
            self.fun = _gpu_bounded_contrast
            self.fun_args = dict(G=G, batch_size=batch_size,
                                 device=self.device,
                                 gpu_mem_limit=gpu_mem_limit)
        elif isinstance(K, int) and K in (4, 6, 8):
            self.fun = _gpu_lcc_contrast
            self.fun_args = dict(K=K, device=self.device)

    def get_params(self, deep=True):
        params = super().get_params(deep=deep)
        params["device"] = self.device
        params["batch_size"] = self.batch_size
        params["gpu_mem_limit"] = self.gpu_mem_limit
        params["clear_gpu"] = self.clear_gpu
        return params

    def set_params(self, **params):
        device = params.pop("device", None)
        batch_size = params.pop("batch_size", None)
        gpu_mem_limit = params.pop("gpu_mem_limit", None)
        if device is not None:
            self.device = device
        if batch_size is not None:
            self.batch_size = batch_size
        if gpu_mem_limit is not None:
            self.gpu_mem_limit = gpu_mem_limit
        super().set_params(**params)
        K = self.K
        if K in self._LCC_BOUNDED_MAP:
            G = self._LCC_BOUNDED_MAP[K]
            self.fun = _gpu_bounded_contrast
            self.fun_args = dict(G=G, batch_size=self.batch_size,
                                 device=self.device,
                                 gpu_mem_limit=self.gpu_mem_limit)
        elif isinstance(K, int) and K in (4, 6, 8):
            self.fun = _gpu_lcc_contrast
            self.fun_args = dict(K=K, device=self.device)
        return self

    def _clear(self):
        if self.clear_gpu and self.device.startswith("cuda"):
            import gc
            gc.collect()
            torch.cuda.empty_cache()

    def fit(self, X, y=None):
        result = super().fit(X, y)
        self._clear()
        return result

    def fit_transform(self, X, y=None):
        result = super().fit_transform(X, y)
        self._clear()
        return result
