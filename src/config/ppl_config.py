from pathlib import Path
from pydantic import BaseModel, ConfigDict, model_validator  # type: ignore


class PipelineSettings(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    input_csv: Path
    output_csv: Path
    interest_rate_csv: Path
    epsilon: float = 1e-10
    max_steps: int

    @model_validator(mode="after")
    def validate_paths(self) -> "PipelineSettings":
        if not self.input_csv.exists():
            raise ValueError(f"input_csv does not exist: {self.input_csv}")
        if not self.interest_rate_csv.exists():
            raise ValueError(f"interest_rate_csv does not exist: {self.interest_rate_csv}")
        if not self.output_csv.parent.exists():
            raise ValueError(f"output_csv parent directory does not exist: {self.output_csv.parent}")
        return self
