import gc
import logging
import os
import threading
import time
from collections.abc import Callable

import psutil
import torch

import cufy.backends.torch.config as config

logger = logging.getLogger(__name__)


def _is_oom_error(e: Exception) -> bool:
    if isinstance(e, (MemoryError, torch.cuda.OutOfMemoryError)):
        return True
    msg = str(e).lower()
    return ("out of memory" in msg) or ("cublas_status_alloc_failed" in msg)


def _available_memory_bytes(available_memory_fraction: float) -> int:
    d = config.device
    if d.type == "cuda":
        total = int(torch.cuda.get_device_properties(d).total_memory)
    else:
        total = int(psutil.virtual_memory().total)

    return int(total * available_memory_fraction)


def _cuda_peak_delta_bytes(f: Callable[[], None], device: torch.device) -> int:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    baseline = torch.cuda.memory_allocated(device)
    f()
    torch.cuda.synchronize(device)
    peak = torch.cuda.max_memory_allocated(device)
    return max(1, peak - baseline)


def _cpu_peak_delta_bytes(f: Callable[[], None]) -> int:
    interval = 0.001
    proc = psutil.Process(os.getpid())
    baseline = int(proc.memory_info().rss)
    peak = baseline
    stopped = False

    def sampler():
        nonlocal peak
        while not stopped:
            rss = int(proc.memory_info().rss)
            if rss > peak:
                peak = rss
            time.sleep(interval)

    t = threading.Thread(target=sampler, daemon=True)
    t.start()
    try:
        f()
    finally:
        stopped = True
        t.join(timeout=1.0)
    return max(1, peak - baseline)


def _probe_peak_bytes(make_f: Callable[[int, int], Callable[[], None]], m: int, n: int) -> int:
    device = config.device
    oom = False
    try:
        if device.type == "cuda":
            return _cuda_peak_delta_bytes(make_f(m, n), device)
        return _cpu_peak_delta_bytes(make_f(m, n))
    except Exception as e:
        if _is_oom_error(e):
            oom = True
        else:
            raise RuntimeError(
                f"Memory probing failed while running the representative workload. ({m=}, {n=}, device={device.type}). "
                f"Original error: {type(e).__name__}: {e}"
            ) from e
    finally:
        if oom:
            if device.type == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
    return -1


def _probe_grid(max_n: int, max_m: int) -> list[tuple[int, int]]:
    n_min = max_n // 8
    m_min = max_m // 32

    n_levels = [x for x in (n_min, 2 * n_min, 4 * n_min, max_n)]
    m_levels = [x for x in (m_min, 2 * m_min, 4 * m_min, 8 * m_min)]

    points = [(m, n) for n in n_levels for m in m_levels]
    points.sort(key=lambda t: t[0] * t[1])
    return points


def probe_param_batch_sizes(
    *, make_f: Callable[[int, int], Callable[[], None]], max_n: int, max_m: int, available_memory_fraction: float
) -> Callable[[int], int]:
    available_bytes = _available_memory_bytes(available_memory_fraction)
    if max_n < 8 or max_m < 32:
        raise ValueError("max_n must be at least 8 and max_m must be at least 32")

    points = _probe_grid(max_n, max_m)
    samples: list[tuple[int, int, int]] = []
    for m, n in points:
        peak = _probe_peak_bytes(make_f, m, n)
        if int(peak) < 0:
            break
        samples.append((m, n, peak))

    if len(samples) < 4:
        logger.warning(
            f"memory probe collected only {len(samples)} sample(s) before OOM (need at least 4 to fit the model)."
            f"Falling back to param_batch_size=1."
        )
        return lambda _n: 1

    X = torch.tensor([[1.0, n, m, m * n] for (m, n, _) in samples], dtype=config.dtype)
    y = torch.tensor([p for (_, _, p) in samples], dtype=config.dtype).reshape(-1, 1)
    a0, a1, b0, b1 = torch.linalg.lstsq(X, y).solution.reshape(-1)

    a0 = float(max(0.0, float(a0.item())))
    a1 = float(max(0.0, float(a1.item())))
    b0 = float(max(1.0, float(b0.item())))
    b1 = float(max(0.0, float(b1.item())))
    logger.info(
        f"memory probe: device={config.device.type}, {max_n=}, {max_m=}, {available_bytes=}. "
        f"Fitted peak(m,n) ≈ a0 + a1*n + b0*m + b1*m*n with {a0=:.2f}, {a1=:.2f}, {b0=:.2f}, {b1=:.2f}."
    )

    def batch_size(n: int) -> int:
        a = a0 + a1 * max(1, n)
        b = b0 + b1 * max(1, n)
        return max(1, int((available_bytes - a) // b))

    return batch_size
