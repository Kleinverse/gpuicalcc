# gpuicalcc

GPU-accelerated locally centered cyclic contrasts for FastICA.

PyTorch extension of [icalcc](https://github.com/Kleinverse/icalcc). Same API, drop-in replacement with CUDA acceleration for polynomial and bounded LCC contrasts.

```python
from gpuicalcc import GPUICALCC
ica = GPUICALCC(n_components=4, K=6, device='cuda')
S_hat = ica.fit_transform(X)
```
## See Also

- [Kleinverse Open Research Repository (KORR)](https://github.com/Kleinverse/research/tree/main/lcc) — research and experiment code

## Installation

```bash
pip install gpuicalcc
```

Requires PyTorch with CUDA. See [pytorch.org](https://pytorch.org) for installation instructions.

## Usage

```python
from gpuicalcc import GPUICALCC

# LCC order 6 on GPU
ica = GPUICALCC(n_components=4, K=6, device='cuda', random_state=0)
S_hat = ica.fit_transform(X)

# Bounded LCC-tanh with memory limit
ica = GPUICALCC(n_components=4, K='ltanh', device='cuda',
                batch_size=500, gpu_mem_limit=8, random_state=0)
S_hat = ica.fit_transform(X)

# Falls back to CPU automatically if CUDA unavailable
ica = GPUICALCC(n_components=4, K=8, device='cuda', random_state=0)
```

## Supported K Values

| K | Description |
|---|---|
| `4` | LCC polynomial order 4 |
| `6` | LCC polynomial order 6 (recommended) |
| `8` | LCC polynomial order 8 |
| `'fast4'` | FastICA(4), g(y) = y³ |
| `'fast6'` | FastICA(6), g(y) = y⁵ |
| `'fast8'` | FastICA(8), g(y) = y⁷ |
| `'tanh'` | logcosh contrast |
| `'exp'` | Gaussian contrast |
| `'ltanh'` | Locally centered tanh (GPU pairwise) |
| `'lexp'` | Locally centered exp (GPU pairwise) |
| `'skew'` | Skewness contrast (experimental) |

Polynomial LCC (K=4,6,8) and bounded LCC (K='ltanh','lexp') are GPU-accelerated. Classical contrasts ('tanh', 'exp', 'skew') and FastICA(k) fall back to the CPU implementation in `icalcc`.

## Parameters

| Parameter | Default | Description |
|---|---|---|
| `device` | `'cuda'` | PyTorch device. Falls back to `'cpu'` if CUDA unavailable |
| `batch_size` | `500` | Subsample size for bounded pairwise computation |
| `gpu_mem_limit` | `None` | GPU memory limit in GB. Auto-detected if None |
| `clear_gpu` | `True` | Clear GPU cache after fit |

## Requirements

- Python ≥ 3.9
- numpy ≥ 1.24
- scikit-learn ≥ 1.3
- icalcc ≥ 0.1.0
- torch ≥ 2.0

## Citation

```bibtex
@misc{saito2026lcc,
  author    = {Saito, Tetsuya},
  title     = {Locally Centered Cyclic Kernels for Higher-Order Independent Component Analysis},
  year      = {2026},
  publisher = {TechRxiv},
  doi       = {10.36227/techrxiv.XXXXXXX}
}
```

## License

[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
