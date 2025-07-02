import torch
from torch import nn
from typing import Optional


class Sampler(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, 
                logits: torch.Tensor, 
                temperatures: torch.Tensor,
                repetition_penalties: Optional[torch.Tensor] = None,
                top_ps: Optional[torch.Tensor] = None,
                prompt_tokens: Optional[torch.Tensor] = None,
                output_tokens: Optional[torch.Tensor] = None):
        """
        Args:
            logits: (batch_size, vocab_size) logits from the model
            temperatures: (batch_size,) temperature values
            repetition_penalties: (batch_size,) repetition penalty values
            top_ps: (batch_size,) top_p values
            prompt_tokens: (batch_size, max_prompt_len) prompt token ids (padded with vocab_size)
            output_tokens: (batch_size, max_output_len) output token ids (padded with vocab_size)
        """
        logits = logits.to(torch.float)
        
        # Apply repetition penalties if provided
        if (repetition_penalties is not None and 
            prompt_tokens is not None and 
            output_tokens is not None):
            logits = self._apply_repetition_penalties(
                logits, prompt_tokens, output_tokens, repetition_penalties
            )
        
        # Apply temperature scaling
        greedy_tokens = logits.argmax(dim=-1)
        logits = logits / temperatures.unsqueeze(dim=1).clamp(min=1e-8)
        
        # Apply top_p if provided
        if top_ps is not None:
            logits = self._apply_top_p(logits, top_ps)
        
        # Sample from the logits
        probs = torch.softmax(logits, dim=-1, dtype=torch.float)
        
        # Use gumbel-max trick for sampling (more stable than multinomial)
        epsilon = 1e-10  
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1) + epsilon).argmax(dim=-1)
        
        # Return greedy tokens when temperature is 0, otherwise sample
        return torch.where(temperatures == 0, greedy_tokens, sample_tokens)
    
    def _apply_repetition_penalties(self, 
                                  logits: torch.Tensor,
                                  prompt_tokens: torch.Tensor,
                                  output_tokens: torch.Tensor,
                                  repetition_penalties: torch.Tensor) -> torch.Tensor:
        """Apply repetition penalties to logits based on prompt and output tokens."""
        batch_size, vocab_size = logits.shape
        
        # Get token frequencies from prompt and output
        prompt_mask = self._get_token_mask(prompt_tokens, vocab_size, batch_size)
        output_mask = self._get_token_mask(output_tokens, vocab_size, batch_size)
        
        # Combine prompt and output masks
        combined_mask = prompt_mask | output_mask
        
        # Apply repetition penalty: 
        # if logits[i] > 0 and token appeared before: logits[i] /= penalty
        # if logits[i] <= 0 and token appeared before: logits[i] *= penalty
        penalty_mask = combined_mask.float()
        penalty = repetition_penalties.unsqueeze(1).expand(-1, vocab_size)
        
        # Create penalty factor: penalty for positive logits, 1/penalty for negative logits
        positive_mask = (logits > 0).float()
        negative_mask = (logits <= 0).float()
        
        penalty_factor = (positive_mask / penalty + negative_mask * penalty) * penalty_mask + (1 - penalty_mask)
        
        return logits * penalty_factor
    
    def _get_token_mask(self, tokens: torch.Tensor, vocab_size: int, batch_size: int) -> torch.Tensor:
        """Create a mask indicating which tokens have appeared in the sequence."""
        # Create a mask for each token in the vocabulary
        mask = torch.zeros((batch_size, vocab_size), dtype=torch.bool, device=tokens.device)
        
        # Mark tokens that appear in the sequence (vocab_size is used as padding)
        valid_tokens = tokens[tokens < vocab_size]
        if valid_tokens.numel() > 0:
            batch_indices = torch.arange(batch_size, device=tokens.device).unsqueeze(1).expand_as(tokens)
            valid_mask = tokens < vocab_size
            batch_idx = batch_indices[valid_mask]
            token_idx = tokens[valid_mask]
            mask[batch_idx, token_idx] = True
        
        return mask
    
    def _apply_top_p(self, logits: torch.Tensor, top_ps: torch.Tensor) -> torch.Tensor:
        """Apply top-p (nucleus) sampling by masking tokens below cumulative probability threshold."""
        # Sort logits in descending order
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        
        # Get probabilities of sorted logits
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        
        # Calculate cumulative probabilities
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        
        # Create mask for tokens to remove
        # We want to remove tokens where cumulative_probs - current_prob >= top_p
        # This means we include tokens while cumsum <= top_p
        sorted_indices_to_remove = cumulative_probs - sorted_probs > top_ps.unsqueeze(1)
        
        # Keep at least one token (the most likely one)
        sorted_indices_to_remove[:, 0] = False
        
        # Create a new logits tensor with filtered values
        filtered_sorted_logits = sorted_logits.clone()
        filtered_sorted_logits[sorted_indices_to_remove] = float('-inf')
        
        # Scatter back to original order
        filtered_logits = torch.zeros_like(logits)
        filtered_logits.scatter_(dim=-1, index=sorted_indices, src=filtered_sorted_logits)
        
        return filtered_logits
