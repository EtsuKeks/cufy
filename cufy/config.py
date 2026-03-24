import os

device: str = os.environ.get("CUFY_DEVICE", "auto").strip().lower()
dtype: str = os.environ.get("CUFY_DTYPE", "float32").strip().lower()

if dtype not in {"float32", "float64"}:
    raise ValueError(f"CUFY_DTYPE must be 'float32' or 'float64', got '{dtype}'")

eps: float = float(os.environ.get("CUFY_EPS", "1e-8"))
