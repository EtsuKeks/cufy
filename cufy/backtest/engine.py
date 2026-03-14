import time
from pathlib import Path

from cufy.abc.data import DataSource
from cufy.backtest.metrics import MetricFn
from cufy.backtest.reporting import write_fold
from cufy.backtest.trial import Trial


class BacktestEngine:
    def __init__(
        self,
        trials: list[Trial],
        metrics: list[MetricFn],
    ) -> None:
        if not trials:
            raise ValueError("trials must not be empty")
        ids = [t.id for t in trials]
        if len(set(ids)) != len(ids):
            raise ValueError(f"Trial ids must be unique, got duplicates: {ids}")
        if not metrics:
            raise ValueError("metrics must not be empty")
        self.trials = trials
        self.metrics = metrics

    def run(self, source: DataSource, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize runners once per trial to preserve state across folds (Warm Start)
        runners = {trial.id: trial.make_runner() for trial in self.trials}

        # Use itertools.pairwise to stream batches lazily instead of loading all into RAM
        # This prevents Out-Of-Memory (OOM) errors on large datasets
        iterator = iter(source)
        try:
            first_batch = next(iterator)
        except StopIteration:
            raise ValueError("BacktestEngine.run requires at least 2 batches from source")

        def _pairwise_generator():
            prev = first_batch
            for current in iterator:
                yield prev, current
                prev = current

        pairs = _pairwise_generator()
        
        has_run = False
        for fold_idx, (fit_batch, eval_batch) in enumerate(pairs):
            has_run = True
            for trial in self.trials:
                runner = runners[trial.id]

                # --- 1. Initial Search Phase ---
                t0 = time.perf_counter()
                prepared_fit = runner.prepare_batch(fit_batch)
                prepared_eval = runner.prepare_batch(eval_batch)
                
                # Only run full initial search if the model is not initialized yet.
                # This handles both the first fold (cold start) and models that
                # were explicitly initialized with parameters by the user.
                ran_initial = False
                if not runner.is_initialized():
                    runner.find_initial_params(prepared_fit)
                    ran_initial = True
                initial_fit_elapsed = time.perf_counter() - t0

                if ran_initial:
                    t0 = time.perf_counter()
                    initial_priced = runner.price(prepared_eval)
                    initial_predict_elapsed = time.perf_counter() - t0

                    initial_metrics: dict[str, float] = {}
                    for fn in self.metrics:
                        initial_metrics.update(fn(initial_priced, prepared_eval))

                    write_fold(
                        output_dir=output_dir,
                        trial=trial,
                        fold_idx=fold_idx,
                        priced=initial_priced,
                        eval_batch=prepared_eval,
                        metrics=initial_metrics,
                        fit_elapsed_sec=initial_fit_elapsed,
                        predict_elapsed_sec=initial_predict_elapsed,
                        stage="initial",
                    )

                # --- 2. Calibration Phase ---
                t0 = time.perf_counter()
                runner.calibrate(prepared_fit)
                calibrate_fit_elapsed = time.perf_counter() - t0

                t0 = time.perf_counter()
                calibrated_priced = runner.price(prepared_eval)
                calibrate_predict_elapsed = time.perf_counter() - t0

                calibrated_metrics: dict[str, float] = {}
                for fn in self.metrics:
                    calibrated_metrics.update(fn(calibrated_priced, prepared_eval))

                write_fold(
                    output_dir=output_dir,
                    trial=trial,
                    fold_idx=fold_idx,
                    priced=calibrated_priced,
                    eval_batch=prepared_eval,
                    metrics=calibrated_metrics,
                    fit_elapsed_sec=calibrate_fit_elapsed,
                    predict_elapsed_sec=calibrate_predict_elapsed,
                    stage="calibrate",
                )
                
        if not has_run:
            raise ValueError("BacktestEngine.run requires at least 2 batches from source")
