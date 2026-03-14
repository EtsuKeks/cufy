from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from cufy.abc.data import OptionBatch, PricedBatch
from cufy.backtest.trial import Trial


def _predictions_table(
    priced: PricedBatch,
    *,
    trial: Trial,
    fold_idx: int,
    fit_timestamp: str,
    stage: str,
) -> pa.Table:
    n = len(priced)
    arrays: dict[str, list[object]] = {
        "timestamp": [priced.timestamp] * n,
        "fit_timestamp": [fit_timestamp] * n,
        "fold_idx": [fold_idx] * n,
        "trial_id": [trial.id] * n,
        "stage": [stage] * n,
    }
    for tag_key, tag_val in trial.tags.items():
        arrays[f"tag_{tag_key}"] = [tag_val] * n
    for col_name, arr in priced.columns.items():
        arrays[col_name] = arr.tolist()
    arrays["F"] = priced.source.F.tolist()
    arrays["K"] = priced.source.K.tolist()
    arrays["T"] = priced.source.T.tolist()
    arrays["is_call"] = priced.source.is_call.tolist()
    arrays["close"] = priced.source.close.tolist()
    arrays["close_IV"] = priced.source.close_IV.tolist()
    arrays["df"] = priced.source.df.tolist()
    arrays["w"] = priced.source.w.tolist()
    for extra_key, extra_arr in priced.source.extras.items():
        arrays[f"extra_{extra_key}"] = extra_arr.tolist()
    return pa.table({k: pa.array(v) for k, v in arrays.items()})


def _metrics_table(
    metrics: dict[str, float],
    *,
    trial: Trial,
    fold_idx: int,
    fit_timestamp: str,
    eval_timestamp: str,
    fit_elapsed_sec: float,
    predict_elapsed_sec: float,
    numeric_state: dict[str, float],
    string_state: dict[str, str],
    stage: str,
) -> pa.Table:
    row: dict[str, object] = {
        "fold_idx": fold_idx,
        "trial_id": trial.id,
        "fit_timestamp": fit_timestamp,
        "eval_timestamp": eval_timestamp,
        "fit_elapsed_sec": fit_elapsed_sec,
        "predict_elapsed_sec": predict_elapsed_sec,
        "stage": stage,
    }
    for tag_key, tag_val in trial.tags.items():
        row[f"tag_{tag_key}"] = tag_val
    for k, v in metrics.items():
        row[f"metric_{k}"] = v
    for k, v in numeric_state.items():
        row[f"state_num_{k}"] = v
    for k, v in string_state.items():
        row[f"state_str_{k}"] = v
    return pa.table({k: pa.array([v]) for k, v in row.items()})


def write_fold(
    output_dir: Path,
    trial: Trial,
    fold_idx: int,
    priced: PricedBatch,
    eval_batch: OptionBatch,
    metrics: dict[str, float],
    fit_elapsed_sec: float,
    predict_elapsed_sec: float,
    stage: str,
) -> None:
    trial_dir = output_dir / trial.id
    
    # We create partitioned directories for each fold to avoid O(N^2) I/O overhead
    # from constantly appending to a single growing parquet file.
    fold_dir = trial_dir / f"fold={fold_idx}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    pred_table = _predictions_table(
        priced,
        trial=trial,
        fold_idx=fold_idx,
        fit_timestamp=priced.source.timestamp,
        stage=stage,
    )
    # Write directly to a new file for this fold and stage
    pq.write_table(pred_table, fold_dir / f"predictions_{stage}.parquet", compression="snappy")

    metrics_table = _metrics_table(
        metrics,
        trial=trial,
        fold_idx=fold_idx,
        fit_timestamp=priced.source.timestamp,
        eval_timestamp=eval_batch.timestamp,
        fit_elapsed_sec=fit_elapsed_sec,
        predict_elapsed_sec=predict_elapsed_sec,
        numeric_state=priced.numeric_state,
        string_state=priced.string_state,
        stage=stage,
    )
    pq.write_table(metrics_table, fold_dir / f"metrics_{stage}.parquet", compression="snappy")
