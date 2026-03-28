import torch

import cufy.config as _core

device: torch.device = (
    (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    if _core.device == "auto"
    else torch.device(_core.device)
)

dtype: torch.dtype = torch.float32 if _core.dtype == "float32" else torch.float64

eps: torch.Tensor = torch.tensor(_core.eps, device=device, dtype=dtype)
