from abc import ABC, abstractmethod
from typing import Mapping

import numpy as np
import pandas as pd
import torch

from src.abc.model import Model
from src.abc.parameterized_model import ParameterizedModel
from src.config.config import settings
from src.utils.implied_vol import implied_vol_newton_bs
from src.utils.torch_utils import Batch1D


def split_df(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    required = ("current_time", "underlying_price", "strike", "ttm", "is_call", "close", "close_IV")
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Invalid options batch dataframe: missing required columns: {missing}")
    return (
        df["underlying_price"].to_numpy(),
        df["strike"].to_numpy(),
        df["ttm"].to_numpy(),
        df["is_call"].to_numpy(),
        df["close"].to_numpy(),
        df["close_IV"].to_numpy(),
    )


class Runner(ABC):
    @property
    @abstractmethod
    def running_pairs(self) -> Mapping[str, Model]:
        raise NotImplementedError

    @abstractmethod
    def filter_df(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def compute_weights(self, df: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    def _to_utc_date(self, s: pd.Series) -> pd.Series:
        dt = pd.to_datetime(s, utc=True, errors="coerce")
        if dt.isna().any():
            raise ValueError(f"Failed to parse values in s['current_time'] as datetimes")
        return dt.dt.normalize()

    def _load_interest_rate_series(self) -> pd.Series:
        if hasattr(self, "_interest_rates_s") and getattr(self, "_interest_rates_s") is not None:
            return getattr(self, "_interest_rates_s")

        rates = pd.read_csv(settings.ppl.interest_rate_csv)
        if "current_time" not in rates.columns or "interest_rate" not in rates.columns:
            raise ValueError("interest_rate_csv must have columns: 'current_time' and 'interest_rate'")

        rates["current_time"] = self._to_utc_date(rates["current_time"])
        rates["interest_rate"] = pd.to_numeric(rates["interest_rate"], errors="coerce")
        rates = rates.dropna(subset=["current_time", "interest_rate"]).reset_index(drop=True)

        s = rates.set_index("current_time")["interest_rate"]
        if s.index.duplicated().any():
            dup = s.index[s.index.duplicated()].unique()
            raise ValueError(
                f"interest_rate_csv has duplicate dates: {list(dup[:5])}{'...' if len(dup) > 5 else ''}"
            )

        setattr(self, "_interest_rates_s", s)
        return s

    def _resolve_r(self, df: pd.DataFrame) -> np.ndarray:
        rates = self._load_interest_rate_series()
        ct = self._to_utc_date(df["current_time"])
        r = ct.map(rates)
        if r.isna().any():
            raise ValueError(
                f"Missing interest rate(s) for some option observation times: {r.index[r.isna()].tolist()}"
                "You must provide interest_rate_csv rows for ALL dates/times where you want to price options."
            )

        return r.to_numpy()

    def _validate_filtered(self, df: pd.DataFrame, df_f: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
        if len(df_f) == 0:
            raise ValueError("filter_df(df) must not filter out all options in the batch")

        pos = df.index.get_indexer(df_f.index)
        if (pos < 0).any():
            raise ValueError("filter_df must return a subset of df (its result's index must be contained in df.index)")

        return df_f, pos

    def _validate_w(self, df: pd.DataFrame, w: np.ndarray) -> np.ndarray:
        if w.ndim != 1 or w.dtype != float or len(w) != len(df):
            raise ValueError(f"weights must be 1D of shape (len(df),), got shape={w.shape}, n={len(df)}")
        return w

    def _prepare_fit_batch(
        self, df_f: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if len(df_f) == 0:
            raise ValueError("Filtered dataframe must not be empty")
        S, K, T, is_call, close, close_IV = split_df(df_f)
        r = self._resolve_r(df_f)
        w = self._validate_w(df_f, self.compute_weights(df_f))
        return S, K, T, is_call, close, close_IV, r, w

    def _prepare_fit_batch_maybe_filter(
        self, df: pd.DataFrame, to_filter: bool
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if to_filter:
            df, _ = self._validate_filtered(df, self.filter_df(df))
        return self._prepare_fit_batch(df)

    def find_initial_params(self, df: pd.DataFrame, to_filter: bool = True) -> None:
        S, K, T, is_call, close, close_IV, r, w = self._prepare_fit_batch_maybe_filter(df, to_filter)
        for _, model in self.running_pairs.items():
            model.find_initial_params(S=S, K=K, T=T, is_call=is_call, close=close, close_IV=close_IV, r=r, w=w)

    def calibrate(self, df: pd.DataFrame, to_filter: bool = True) -> None:
        S, K, T, is_call, close, close_IV, r, w = self._prepare_fit_batch_maybe_filter(df, to_filter)
        for _, model in self.running_pairs.items():
            model.calibrate(S=S, K=K, T=T, is_call=is_call, close=close, close_IV=close_IV, r=r, w=w)

    def price(self, df: pd.DataFrame, to_filter: bool = True) -> pd.DataFrame:
        if to_filter:
            df_f, pos = self._validate_filtered(df, self.filter_df(df))
        else:
            if len(df) == 0:
                raise ValueError("Filtered dataframe must not be empty")
            df_f = df
            pos = None

        S, K, T, is_call, _, _ = split_df(df_f)
        r = self._resolve_r(df_f)

        def _assign_column(name: str, values: np.ndarray | object) -> None:
            if not to_filter:
                df[name] = values
                return

            if isinstance(values, np.ndarray):
                if values.dtype == object:
                    out = np.full(df.shape[0], None, dtype=object)
                else:
                    out = np.full(df.shape[0], np.nan)
                out[pos] = values
            else:
                out = np.full(
                    df.shape[0],
                    None if isinstance(values, str) else np.nan,
                    dtype=object if isinstance(values, str) else None
                )
                out[pos] = values
            df[name] = out

        for tag, model in self.running_pairs.items():
            pred = model.price(S=S, K=K, T=T, is_call=is_call, r=r)

            with torch.no_grad():
                price_t = torch.as_tensor(pred, device=settings.device, dtype=settings.dtype).reshape(1, -1)
                dummy_w = np.ones_like(S, dtype=np.float64)
                dummy_iv = np.zeros_like(S, dtype=np.float64)
                batch = Batch1D.from_numpy(S=S, K=K, T=T, is_call=is_call, close_IV=dummy_iv, r=r, w=dummy_w)
                iv_t = implied_vol_newton_bs(price=price_t, data=batch)[0]
                iv = iv_t.detach().cpu().numpy()

            _assign_column(f"{tag}_price", pred)
            _assign_column(f"{tag}_iv", iv)

            if isinstance(model, ParameterizedModel):
                for label, params in model.get_params().items():
                    for name, value in params.items():
                        _assign_column(f"{tag}_{label}_{name}", value)

                for label, facts in model.get_facts().items():
                    for name, value in facts.items():
                        _assign_column(f"{tag}_{label}_{name}", value)
        return df
