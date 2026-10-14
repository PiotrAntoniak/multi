"""gpu_limit.py -- cap this process's PyTorch GPU memory use.

``limit_gpu(gb=None)`` caps the PyTorch CUDA caching allocator for device 0 to
``gb`` GiB via ``torch.cuda.set_per_process_memory_fraction``.  The default
``gb`` is read from the env var ``TORCH_GPU_LIMIT_GB`` (default 10).

Safe to call on CPU-only hosts or when CUDA is unavailable: torch is imported
lazily and the function returns quietly if CUDA cannot be used.  Call it once,
early -- before any model or tensor touches CUDA, e.g. at the top of ``main``:

    try:
        from gpu_limit import limit_gpu; limit_gpu()
    except Exception:
        pass
"""
import os


def limit_gpu(gb=None):
    """Cap this process's PyTorch GPU memory to ``gb`` GiB (no-op on CPU / no CUDA).

    ``gb`` defaults to the ``TORCH_GPU_LIMIT_GB`` env var, else 10 GiB.
    """
    if gb is None:
        try:
            gb = float(os.environ.get("TORCH_GPU_LIMIT_GB", "10"))
        except (TypeError, ValueError):
            gb = 10.0
    try:
        gb = float(gb)
    except (TypeError, ValueError):
        return
    if gb <= 0:
        return

    try:
        import torch
    except Exception:
        return
    try:
        if not torch.cuda.is_available():
            return
        total = torch.cuda.get_device_properties(0).total_memory
        if not total:
            return
        fraction = min(1.0, gb * (1024 ** 3) / float(total))
        torch.cuda.set_per_process_memory_fraction(fraction, 0)
    except Exception:
        return
