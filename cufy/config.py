import os

import torch

_device_str: str = os.environ.get("CUFY_DEVICE", "auto").strip().lower()
dtype_str: str = os.environ.get("CUFY_DTYPE", "float32").strip().lower()

if dtype_str not in {"float32", "float64"}:
    raise ValueError(f"CUFY_DTYPE must be 'float32' or 'float64', got '{dtype_str}'")

device: torch.device = (
    (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    if _device_str == "auto"
    else torch.device(_device_str)
)

dtype: torch.dtype = torch.float32 if dtype_str == "float32" else torch.float64

eps_float: float = float(os.environ.get("CUFY_EPS", "1e-8"))
eps: torch.Tensor = torch.tensor(eps_float, device=device, dtype=dtype)

mlflow_enabled: bool = os.environ.get("CUFY_MLFLOW", "1").strip() not in {"0", "false", "no"}
