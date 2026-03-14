import math
from typing import Callable

import numpy as np

from cufy.abc.data import OptionBatch, PricedBatch

MetricFn = Callable[[PricedBatch, OptionBatch], dict[str, float]]


def iv_metrics(pred: PricedBatch, eval_batch: OptionBatch) -> dict[str, float]:
    pred_iv = pred.col("iv")
    actual_iv = eval_batch.close_IV

    if len(pred_iv) != len(actual_iv):
        raise ValueError(
            f"pred has {len(pred_iv)} rows but eval_batch has {len(actual_iv)} rows"
        )

    ok = np.isfinite(actual_iv) & np.isfinite(pred_iv)
    if not bool(ok.any()):
        return {
            "n_obs": 0.0,
            "mae_iv": float("nan"),
            "rmse_iv": float("nan"),
            "weighted_rmse_iv": float("nan"),
            "max_abs_iv": float("nan"),
        }

    actual_ok = actual_iv[ok]
    pred_ok = pred_iv[ok]
    err = pred_ok - actual_ok
    abs_err = np.abs(err)

    raw_w = np.asarray(eval_batch.w, dtype=float)[ok]
    raw_w = np.where(np.isfinite(raw_w) & (raw_w > 0.0), raw_w, 0.0)
    total = float(np.sum(raw_w))
    if not math.isfinite(total) or total <= 0.0:
        n = int(ok.sum())
        w = np.full(n, 1.0 / max(1, n), dtype=float)
    else:
        w = raw_w / total

    return {
        "n_obs": float(actual_ok.size),
        "mae_iv": float(np.mean(abs_err)),
        "rmse_iv": float(np.sqrt(np.mean(err * err))),
        "weighted_rmse_iv": float(np.sqrt(np.sum(w * err * err))),
        "max_abs_iv": float(np.max(abs_err)),
    }
