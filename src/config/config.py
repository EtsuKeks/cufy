import os
from pathlib import Path

import torch
from pydantic import BaseModel, ConfigDict, Field  # type: ignore

from src.config.ppl_config import PipelineSettings

PROJECT_ROOT = Path(__file__).parent.parent.parent
INPUTS_DIR = PROJECT_ROOT / "data" / "inputs"
OUTPUTS_DIR = PROJECT_ROOT / "data" / "outputs"


class GlobalSettings(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    device: torch.device = Field(
        default_factory=lambda: (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    )
    dtype: torch.dtype = torch.float32
    ppl: PipelineSettings = PipelineSettings()


settings = GlobalSettings()

if settings.dtype not in (torch.float32, torch.float64):
    raise ValueError("settings.dtype must be torch.float32 or torch.float64")

torch.set_default_dtype(settings.dtype)

if settings.device.type == "cpu":
    torch.set_num_threads(os.cpu_count() or 1)
