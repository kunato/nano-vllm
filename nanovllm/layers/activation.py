import torch
from torch import nn
import torch.nn.functional as F


class SiluAndMul(nn.Module):

    def __init__(self):
        super().__init__()

    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, y = x.chunk(2, -1)
        return F.silu(x) * y


class GeluAndMul(nn.Module):
    """Gelu activation function with multiplication for Gemma3 models.
    
    Following exact vLLM implementation.
    """

    def __init__(self, approximate: str = "none"):
        super().__init__()
        self.approximate = approximate

    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, y = x.chunk(2, -1)
        x = F.gelu(x, approximate=self.approximate)
        return x * y
