from dataclasses import dataclass

import numpy as np

from cufy.abc.data import OptionBatch

@dataclass(frozen=True, slots=True)
class WithinBatchSplit:
    train: OptionBatch
    validation: OptionBatch
    validation_frac: float
    strategy: str
    validation_rows: int
    train_rows: int

def _rank_bucket(values: np.ndarray, bins: int, missing_label: str) -> list[str]:
    out = [missing_label] * len(values)
    finite = np.isfinite(values)
    if not finite.any():
        return out
    vals = values[finite].astype(float)
    ranks = np.argsort(np.argsort(vals)).astype(float)                      
    pct = ranks / max(len(vals) - 1, 1)
    bucket = np.minimum(np.floor(pct * bins).astype(int), bins - 1)
    indices = np.where(finite)[0]
    for i, b in zip(indices, bucket):
        out[i] = f"b{b}"
    return out

def _make_strata(batch: OptionBatch, *, moneyness_bins: int, ttm_bins: int) -> np.ndarray:
    m = batch.K / batch.F
    m_bucket = _rank_bucket(m, bins=max(1, moneyness_bins), missing_label="m_na")
    t_bucket = _rank_bucket(batch.T, bins=max(1, ttm_bins), missing_label="t_na")
    cp = np.where(batch.is_call.astype(bool), "call", "put")
    return np.array([f"{c}|{mb}|{tb}" for c, mb, tb in zip(cp, m_bucket, t_bucket)])

def _select_evenly_spaced_positions(size: int, count: int) -> np.ndarray:
    if count <= 0:
        return np.empty((0,), dtype=int)
    if count >= size:
        return np.arange(size, dtype=int)
    return np.unique(np.linspace(0, size - 1, num=count, dtype=int).astype(int))

def make_within_batch_split(
    batch: OptionBatch,
    *,
    validation_frac: float = 0.2,
    min_validation_rows: int = 8,
    moneyness_bins: int = 6,
    ttm_bins: int = 4,
) -> WithinBatchSplit:
    if not 0.0 < float(validation_frac) < 1.0:
        raise ValueError(f"validation_frac must be in (0, 1), got {validation_frac}")
    n = len(batch)
    if n < 2:
        raise ValueError("Within-batch split requires at least 2 rows")

    strata = _make_strata(batch, moneyness_bins=moneyness_bins, ttm_bins=ttm_bins)
    target_val = max(1, min(int(round(validation_frac * n)), n - 1))
    val_positions: list[int] = []

    unique_strata = np.unique(strata)
    for s in unique_strata:
        group_idx = np.where(strata == s)[0]
        g = len(group_idx)
        if g < 2:
            continue
        count = int(round(validation_frac * g))
        count = min(max(count, 1), g - 1)
        local_pos = _select_evenly_spaced_positions(size=g, count=count)
        val_positions.extend(group_idx[local_pos].tolist())

    if len(val_positions) < min(target_val, max(1, min_validation_rows)):
        val_positions = _select_evenly_spaced_positions(size=n, count=target_val).tolist()

    val_positions_clean = sorted(set(int(x) for x in val_positions))
    if len(val_positions_clean) >= n:
        val_positions_clean = val_positions_clean[:-1]
    if not val_positions_clean:
        val_positions_clean = [n - 1]

    val_mask = np.zeros(n, dtype=bool)
    val_mask[np.asarray(val_positions_clean, dtype=int)] = True
    if val_mask.all():
        val_mask[-1] = False
    if not (~val_mask).any():
        raise RuntimeError("Within-batch split failed to leave any training rows")

    val_idx = np.where(val_mask)[0]
    train_idx = np.where(~val_mask)[0]

    validation = batch.slice(val_idx)
    train = batch.slice(train_idx)

    if len(validation) == 0 or len(train) == 0:
        raise RuntimeError("Within-batch split produced an empty train or validation subset")

    return WithinBatchSplit(
        train=train,
        validation=validation,
        validation_frac=float(validation_frac),
        strategy="deterministic_stratified_is_call_moneyness_ttm",
        validation_rows=len(validation),
        train_rows=len(train),
    )
