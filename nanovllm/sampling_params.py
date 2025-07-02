from dataclasses import dataclass
from typing import Optional, Union


@dataclass
class SamplingParams:
    temperature: float = 1.0
    max_tokens: int = 64
    ignore_eos: bool = False
    repetition_penalty: float = 1.0
    top_p: float = 1.0
    stop: Optional[Union[str, list[str]]] = None
    stop_token_ids: Optional[list[int]] = None
    
    def __post_init__(self):
        """Validate parameters after initialization."""
        if self.repetition_penalty <= 0:
            raise ValueError(f"repetition_penalty must be > 0, got {self.repetition_penalty}")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p}")
        if self.temperature < 0:
            raise ValueError(f"temperature must be >= 0, got {self.temperature}")
        
        # Normalize stop parameter
        if self.stop is not None:
            if isinstance(self.stop, str):
                self.stop = [self.stop]
        
        # Initialize stop_token_ids if None
        if self.stop_token_ids is None:
            self.stop_token_ids = []
