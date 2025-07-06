import torch
from torch import nn
import nanovllm.utils.distributed as dist
from transformers import Gemma3TextConfig

from nanovllm.layers.activation import GeluAndMul
from nanovllm.layers.attention import Attention
from nanovllm.layers.layernorm import GemmaRMSNorm
from nanovllm.layers.linear import QKVParallelLinear, MergedColumnParallelLinear, RowParallelLinear
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead


class Gemma3MLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_activation: str,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
        )
        if hidden_activation != "gelu_pytorch_tanh":
            raise ValueError(
                "Gemma3 uses `gelu_pytorch_tanh` as the hidden activation "
                "function. Please set `hidden_act` and `hidden_activation` to "
                "`gelu_pytorch_tanh`.")
        self.act_fn = GeluAndMul(approximate="tanh")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x = self.down_proj(x)
        return x


class Gemma3Attention(nn.Module):

    def __init__(self,
                 config: Gemma3TextConfig,
                 hidden_size: int,
                 num_heads: int,
                 num_kv_heads: int,
                 head_dim: int,
                 max_position_embeddings: int,
                 layer_idx: int,
                 attn_logits_soft_cap: float | None = None) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = hidden_size
        tp_size = dist.get_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = config.query_pre_attn_scalar**-0.5

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=config.attention_bias,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=config.attention_bias,
        )

        self.q_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)

        # Interleaved sliding window attention pattern following exact vLLM implementation
        # Check if this layer uses sliding window (local) attention
        self.interleaved_sliding_window = getattr(config, "sliding_window", None)
        self.sliding_window_pattern = getattr(config, "sliding_window_pattern", None)
        
        self.is_sliding = (self.interleaved_sliding_window is not None and 
                          self.sliding_window_pattern is not None and
                          bool((layer_idx + 1) % self.sliding_window_pattern))
        
        # Initialize the rotary embedding following exact vLLM patterns
        if self.is_sliding:
            # Local attention. Override the values in config.json.
            self.rope_theta = getattr(config, "rope_local_base_freq", 10000.0)
            self.rope_scaling = {"rope_type": "default"}
            self.sliding_window = self.interleaved_sliding_window
        else:
            # Global attention. Use the values in config.json.
            self.rope_theta = getattr(config, "rope_theta", 1000000.0)
            self.rope_scaling = getattr(config, "rope_scaling", None)
            self.sliding_window = None
            
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=self.rope_theta,
            is_neox_style=True,
            rope_scaling=self.rope_scaling,
        )

        # Initialize the attention.
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
        )
        
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        q = q.unflatten(-1, (self.num_heads, self.head_dim))
        q = self.q_norm(q)
        q = q.flatten(-2, -1)
        k = k.unflatten(-1, (self.num_kv_heads, self.head_dim))
        k = self.k_norm(k)
        k = k.flatten(-2, -1)

        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        output = self.o_proj(attn_output)
        return output


class Gemma3DecoderLayer(nn.Module):

    def __init__(
        self,
        config: Gemma3TextConfig,
        layer_idx: int,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Gemma3Attention(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            layer_idx=layer_idx,
            attn_logits_soft_cap=None,
        )
        self.mlp = Gemma3MLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_activation=config.hidden_activation,
        )
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states = self.post_attention_layernorm(hidden_states)

        hidden_states, residual = self.pre_feedforward_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        return hidden_states, residual


class Gemma3Model(nn.Module):

    def __init__(
        self,
        config: Gemma3TextConfig,
    ) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Gemma3DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Normalize the embedding by sqrt(hidden_size)
        # The normalizer's data type should be downcasted to the model's
        # data type such as bfloat16, not float32.
        # See https://github.com/huggingface/transformers/pull/29402
        normalizer = self.config.hidden_size**0.5
        self.register_buffer("normalizer",
                             torch.tensor(normalizer),
                             persistent=False)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        # NOTE(woosuk): Only apply the normalizer to the output of
        # vocab embedding. Don't apply it to the vision embedding.
        return self.embed_tokens(input_ids) * self.normalizer

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.get_input_embeddings(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Gemma3ForCausalLM(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(
        self,
        config: Gemma3TextConfig
    ) -> None:
        super().__init__()
        self.config = config
        self.model = Gemma3Model(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        # Currently all existing Gemma models have `tie_word_embeddings` enabled
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, positions)
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        logits = self.lm_head(hidden_states)
        return logits 