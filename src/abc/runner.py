from abc import ABC, abstractmethod
from typing import Mapping

import numpy as np
import pandas as pd
import torch

from src.abc.model import Model
from src.abc.parameterized_model import ParameterizedModel
from src.config.config import settings
from src.utils.implied_vol import implied_vol_newton_bs
from src.utils.torch_utils import inputs_1d


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
        self, df: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        df_f, _ = self._validate_filtered(df, self.filter_df(df))
        S, K, T, is_call, close, close_IV = split_df(df_f)
        r = self._resolve_r(df_f)
        w = self._validate_w(df_f, self.compute_weights(df_f))
        return S, K, T, is_call, close, close_IV, r, w

    def find_initial_params(self, df: pd.DataFrame) -> None:
        S, K, T, is_call, close, close_IV, r, w = self._prepare_fit_batch(df)
        for _, model in self.running_pairs.items():
            model.find_initial_params(S=S, K=K, T=T, is_call=is_call, close=close, close_IV=close_IV, r=r, w=w)

    def calibrate(self, df: pd.DataFrame) -> None:
        S, K, T, is_call, close, close_IV, r, w = self._prepare_fit_batch(df)
        for _, model in self.running_pairs.items():
            model.calibrate(S=S, K=K, T=T, is_call=is_call, close=close, close_IV=close_IV, r=r, w=w)

    def price(self, df: pd.DataFrame) -> pd.DataFrame:
        df_f, pos = self._validate_filtered(df, self.filter_df(df))
        S, K, T, is_call, _, _= split_df(df_f)
        r = self._resolve_r(df_f)
        S_t, K_t, T_t, is_call_t, r_t = inputs_1d(S=S, K=K, T=T, is_call=is_call, r=r)
        for tag, model in self.running_pairs.items():
            out = np.full(df.shape[0], np.nan)
            pred = model.price(S=S, K=K, T=T, is_call=is_call, r=r)
            out[pos] = pred
            df[f"{tag}_price"] = out

            with torch.no_grad():
                price_t = torch.as_tensor(pred, device=settings.device, dtype=settings.dtype).reshape(1, -1)
                iv_t = implied_vol_newton_bs(price=price_t, S=S_t, K=K_t, T=T_t, is_call=is_call_t, r=r_t)[0]
                iv = iv_t.detach().cpu().numpy()
            out_iv = np.full(df.shape[0], np.nan)
            out_iv[pos] = iv
            df[f"{tag}_iv"] = out_iv

            if isinstance(model, ParameterizedModel):
                for label, params in model.get_params().items():
                    for name, value in params.items():
                        out_param = np.full(df.shape[0], np.nan)
                        out_param[pos] = value
                        df[f"{tag}_{label}_{name}"] = out_param

                for label, facts in model.get_facts().items():
                    for name, value in facts.items():
                        out_fact = np.full(df.shape[0], None, dtype=object)
                        out_fact[pos] = value
                        df[f"{tag}_{label}_{name}"] = out_fact

        return df
