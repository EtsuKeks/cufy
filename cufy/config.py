import os

import torch

_device_str = os.environ.get("CUFY_DEVICE", "auto").strip().lower()
device: torch.device = (
    (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    if _device_str == "auto"
    else torch.device(_device_str)
)

_dtype_str = os.environ.get("CUFY_DTYPE", "float32").strip().lower()
_dtype_map: dict[str, torch.dtype] = {"float32": torch.float32, "float64": torch.float64}
if _dtype_str not in _dtype_map:
    raise ValueError(f"CUFY_DTYPE must be 'float32' or 'float64', got '{_dtype_str}'")
dtype: torch.dtype = _dtype_map[_dtype_str]

eps: float = float(os.environ.get("CUFY_EPS", "1e-10"))

torch.set_default_dtype(dtype)
if device.type == "cpu":
    torch.set_num_threads(os.cpu_count() or 1)
