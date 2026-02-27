# gpuicalcc
GPU-accelerated locally centered contrast functions for FastICA.
PyTorch extension of [icalcc](https://github.com/Kleinverse/icalcc).
Same API, drop-in replacement with CUDA acceleration for bounded
and polynomial LCC contrasts.
```python
from gpuicalcc import GPUICALCC
ica = GPUICALCC(n_components=4, K='ltanh', device='cuda', random_state=0)
S_hat = ica.fit_transform(X)
```

## Installation
```bash
pip install gpuicalcc
```
Requires PyTorch with CUDA. See [pytorch.org](https://pytorch.org)
for installation instructions.

## Supported K Values
| K | Description |
|---|---|
| `'ltanh'` | Bounded LCC-tanh — 40–48× GPU speedup |
| `'lexp'` | Bounded LCC-exp — 40–48× GPU speedup |
| `4` | Polynomial LCC order 4 |
| `6` | Polynomial LCC order 6, couples m₃, m₄, m₆ |
| `8` | Polynomial LCC order 8, up to 2.4× GPU speedup at N≥500k |
| `'tanh'` | Classical logcosh contrast (CPU fallback) |
| `'exp'` | Classical Gaussian contrast (CPU fallback) |
| `'skew'` | Classical cube contrast (CPU fallback) |

Bounded contrasts (`'ltanh'`, `'lexp'`) achieve 40–48× speedup
across all dataset sizes due to the O(NB) pairwise computation
being embarrassingly parallel. Polynomial contrasts benefit from
GPU acceleration at N ≥ 500,000. Classical contrasts fall back
to the CPU implementation in `icalcc`.

## Usage
```python
from gpuicalcc import GPUICALCC

# Bounded LCC-tanh (recommended for heavy-tailed or skewed sources)
ica = GPUICALCC(n_components=4, K='ltanh', device='cuda', random_state=0)
S_hat = ica.fit_transform(X)

# Bounded LCC-tanh with memory limit
ica = GPUICALCC(n_components=4, K='ltanh', device='cuda',
                batch_size=500, gpu_mem_limit=8, random_state=0)
S_hat = ica.fit_transform(X)

# Polynomial LCC order 8 (near-Gaussian sources)
ica = GPUICALCC(n_components=4, K=8, device='cuda', random_state=0)
S_hat = ica.fit_transform(X)

# Falls back to CPU automatically if CUDA unavailable
ica = GPUICALCC(n_components=4, K='ltanh', device='cuda', random_state=0)
```

## Parameters
| Parameter | Default | Description |
|---|---|---|
| `device` | `'cuda'` | PyTorch device. Falls back to `'cpu'` if CUDA unavailable |
| `batch_size` | `500` | Subsample size B for bounded pairwise computation |
| `gpu_mem_limit` | `None` | GPU memory limit in GB. Auto-detected if None |
| `clear_gpu` | `True` | Clear GPU cache after fit |

## Benchmark
![GPU Benchmark](https://github.com/Kleinverse/research/blob/main/icalcc/img/benchmark.png)

Bounded contrasts (`ltanh`, `lexp`) achieve 40–48× speedup across
all dataset sizes. Polynomial contrasts (K=4,6,8) benefit from GPU
acceleration at N ≥ 500,000, reaching up to 2.4× at K=8. All runs
use batch_size=500. Full benchmark table:
[Experiment code](https://github.com/Kleinverse/research/tree/main/icalcc).

## Requirements
- Python ≥ 3.9
- numpy ≥ 1.24
- scikit-learn ≥ 1.3
- icalcc ≥ 0.1.0
- torch ≥ 2.0

## See Also
- [icalcc](https://github.com/Kleinverse/icalcc) — CPU version
- [Experiment code](https://github.com/Kleinverse/research/tree/main/icalcc)

## Citation

If you use this package, please cite both the software paper and
the underlying LCC kernel paper:
```bibtex
@article{saito2026icalcc,
  author  = {Saito, Tetsuya},
  title   = {{ICALCC}: Locally Centered Contrast Functions for
             {FastICA} with {GPU} Acceleration},
  journal = {TechRxiv},
  year    = {2026},
  doi     = {10.36227/techrxiv.177220376.62411390}
}

@article{saito2026lcc,
  author  = {Saito, Tetsuya},
  title   = {Locally Centered Cyclic Kernels for Higher-Order
             Independent Component Analysis},
  journal = {TechRxiv},
  year    = {2026},
  doi     = {10.36227/techrxiv.177203264.46969730}
}
```

## License
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
