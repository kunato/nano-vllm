# 📚 Complete Guide: Adding New Model Support to nano-vLLM

Based on successful implementations of Llama, Qwen2, Qwen3, and Gemma3 support in nano-vLLM, this guide provides a step-by-step process for adding any new model architecture to the nano-vLLM inference engine.

## 📋 Table of Contents

1. [Quick Start Workflow](#quick-start-workflow)
2. [Pre-Implementation Analysis](#pre-implementation-analysis)
3. [Implementation Steps](#implementation-steps)
4. [Complex Model Features](#complex-model-features)
5. [Common Issues & Solutions](#common-issues--solutions)
6. [Testing Guide](#testing-guide)
7. [Best Practices](#best-practices)

---

## 🚀 Quick Start Workflow

1. **Analyze model config** (`pretrained/[model]/config.json`) for unique features
2. **Study official vLLM implementation** (`serving/vllm/vllm/model_executor/models/[model_name].py`)
3. **Copy exact architecture** to `nanovllm/models/[model_name].py`
4. **Add model loading** to `nanovllm/engine/model_runner.py`
5. **Add test function** to `test_models.py`
6. **Test with real model** (if available)

---

## 🔍 Pre-Implementation Analysis

### Step 0: Examine Model Configuration

**CRITICAL**: Always check the model's `config.json` first to identify unique features:

```bash
cat pretrained/[model]/config.json | jq .
```

**Key features to look for:**
- **Attention patterns**: `sliding_window`, `sliding_window_pattern`, `interleaved_sliding_window`
- **RoPE configurations**: `rope_theta`, `rope_scaling`, `rope_local_base_freq`
- **Normalization**: Different models may require specific norm types
- **Activation functions**: `hidden_activation`, `hidden_act`
- **Special scalars**: `query_pre_attn_scalar`, `attn_logit_softcapping`, `final_logit_softcapping`
- **Layer patterns**: Layer-specific configurations or alternating behaviors

**Example Gemma3 config analysis:**
```json
{
  "sliding_window": 1024,
  "sliding_window_pattern": 6,
  "rope_local_base_freq": 10000.0,
  "rope_theta": 1000000.0,
  "query_pre_attn_scalar": 256,
  "hidden_activation": "gelu_pytorch_tanh"
}
```
This indicates: interleaved sliding window attention, layer-specific RoPE configs, and GELU activation.

---

## 🔧 Implementation Steps

### Step 1: Create Model File

**File**: `nanovllm/models/[model_name].py`  
**Reference**: `serving/vllm/vllm/model_executor/models/[model_name].py`

**Template Structure**:
```python
import torch
from torch import nn
import torch.distributed as dist
from transformers import ModelConfig

from nanovllm.layers.activation import SiluAndMul  # Or GeluAndMul for Gemma
from nanovllm.layers.attention import Attention
from nanovllm.layers.layernorm import RMSNorm, GemmaRMSNorm  # Choose correct norm!
from nanovllm.layers.linear import QKVParallelLinear, MergedColumnParallelLinear, RowParallelLinear
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead

class ModelAttention(nn.Module):
    def __init__(self, config, ..., layer_idx=None):  # layer_idx for complex models
        # Copy exact vLLM implementation
        pass

class ModelMLP(nn.Module):
    # Copy exact vLLM implementation
    pass

class ModelDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx=None):  # Pass layer_idx for complex models
        # Copy exact vLLM implementation
        pass

class ModelNameModel(nn.Module):
    def __init__(self, config):
        # For complex models, pass layer_idx to decoder layers
        self.layers = nn.ModuleList([
            ModelDecoderLayer(config, layer_idx) 
            for layer_idx in range(config.num_hidden_layers)
        ])

class ModelForCausalLM(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = ModelNameModel(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(self, input_ids, positions):
        return self.model(input_ids, positions)

    def compute_logits(self, hidden_states):
        return self.lm_head(hidden_states)
```

### Step 2: Add Model Loading

**File**: `nanovllm/engine/model_runner.py`

```python
# Add import
from nanovllm.models.newmodel import NewModelForCausalLM

# Update model selection
if hf_config.model_type == "newmodel":
    self.model = NewModelForCausalLM(hf_config)
elif hf_config.model_type == "gemma3_text":  # Note: exact model_type from config
    self.model = Gemma3ForCausalLM(hf_config)
# ... other models

# Update error message
else:
    raise ValueError(f"Unsupported model type: {hf_config.model_type}. "
                   "Supported types: qwen3, qwen2, llama, gemma3_text, newmodel")
```

---

## 🏗️ Complex Model Features

### Interleaved Sliding Window Attention (Gemma3)

Some models use different attention patterns per layer:

```python
class ModelAttention(nn.Module):
    def __init__(self, config, layer_idx, ...):
        # Check if this layer uses sliding window attention
        self.sliding_window_pattern = getattr(config, "sliding_window_pattern", None)
        self.is_sliding = (
            getattr(config, "sliding_window", None) is not None and 
            self.sliding_window_pattern is not None and
            bool((layer_idx + 1) % self.sliding_window_pattern)
        )
        
        # Layer-specific RoPE configuration
        if self.is_sliding:
            self.rope_theta = getattr(config, "rope_local_base_freq", 10000.0)
            self.rope_scaling = {"rope_type": "default"}
            self.sliding_window = config.sliding_window
        else:
            self.rope_theta = getattr(config, "rope_theta", 1000000.0)
            self.rope_scaling = getattr(config, "rope_scaling", None)
            self.sliding_window = None
```

### Model-Specific Normalization

**Critical**: Use the correct normalization for each model family:

```python
# For Gemma models - REQUIRED for numerical accuracy
from nanovllm.layers.layernorm import GemmaRMSNorm
self.norm = GemmaRMSNorm(hidden_size, eps=config.rms_norm_eps)

# For other models  
from nanovllm.layers.layernorm import RMSNorm
self.norm = RMSNorm(hidden_size, eps=config.rms_norm_eps)
```

**GemmaRMSNorm vs RMSNorm differences:**
- **Weight init**: `torch.zeros()` vs `torch.ones()`
- **Formula**: `x * (1 + weight)` vs `x * weight`
- **Type casting**: `(x * w).to(orig_dtype)` vs `x.to(orig_dtype) * w`

### Embedding Normalization

Some models normalize embeddings (Gemma3):

```python
class ModelNameModel(nn.Module):
    def __init__(self, config):
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        
        # Gemma3-style embedding normalization
        normalizer = config.hidden_size**0.5
        self.register_buffer("normalizer", torch.tensor(normalizer), persistent=False)
    
    def get_input_embeddings(self, input_ids):
        return self.embed_tokens(input_ids) * self.normalizer
```

### Activation Functions

Add new activations as needed:

```python
# nanovllm/layers/activation.py
class GeluAndMul(nn.Module):
    def __init__(self, approximate: str = "none"):
        super().__init__()
        self.approximate = approximate

    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, y = x.chunk(2, -1)
        x = F.gelu(x, approximate=self.approximate)
        return x * y
```

---

## ⚠️ Common Issues & Solutions

### Issue 1: Incorrect Normalization Layer
**Error**: Model generates garbage tokens (e.g., `<unused62>` repeatedly)

**Root Cause**: Using wrong normalization type (RMSNorm instead of GemmaRMSNorm for Gemma models)

**Solution**: Always use the exact normalization from vLLM:
```python
# For Gemma models - CRITICAL
from nanovllm.layers.layernorm import GemmaRMSNorm
self.q_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)

# For other models
from nanovllm.layers.layernorm import RMSNorm  
self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
```

### Issue 2: tie_word_embeddings Runtime Error
**Error**: `Expected tensor for argument #1 'indices' to have one of the following scalar types: Long, Int; but got CUDABFloat16Type instead`

**Root Cause**: Incorrect implementation assigning embedding layer to lm_head
```python
# WRONG - Makes lm_head an embedding layer expecting int indices
if config.tie_word_embeddings:
    self.lm_head = self.model.embed_tokens
```

**Solution**: Always create `ParallelLMHead` and share weights properly
```python
# CORRECT - Always create ParallelLMHead, then share weights
self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
if config.tie_word_embeddings:
    self.lm_head.weight.data = self.model.embed_tokens.weight.data
```

### Issue 3: Missing Layer-Specific Features
**Error**: Model loads but generates wrong outputs

**Root Cause**: Not implementing layer-specific attention patterns or RoPE configs

**Solution**: Pass `layer_idx` through the model hierarchy:
```python
class ModelDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx):
        self.self_attn = ModelAttention(config, layer_idx=layer_idx, ...)

class ModelNameModel(nn.Module):
    def __init__(self, config):
        self.layers = nn.ModuleList([
            ModelDecoderLayer(config, layer_idx) 
            for layer_idx in range(config.num_hidden_layers)
        ])
```

### Issue 4: RoPE Configuration Errors
**Error**: `AssertionError` in `rotary_embedding.py` or wrong attention patterns

**Solution**: Check for layer-specific RoPE configurations and implement exactly:
```python
# Global vs local attention with different RoPE settings
if self.is_sliding:
    self.rope_theta = config.rope_local_base_freq  # Usually 10000.0
    self.rope_scaling = {"rope_type": "default"}
else:
    self.rope_theta = config.rope_theta  # Usually much larger like 1000000.0
    self.rope_scaling = config.rope_scaling  # Complex scaling like linear
```

### Issue 5: Model Architecture Differences
**Problem**: Subtle differences between model generations (e.g., Qwen2 vs Qwen3)

**Key Differences Found**:
- **Q/K Normalization**: Qwen2 has NO q_norm/k_norm layers, Qwen3 has RMSNorm, Gemma3 has GemmaRMSNorm
- **QKV Bias**: Qwen2 always uses `bias=True`, Qwen3 uses configurable bias
- **Activation Functions**: Llama/Qwen use SiLU, Gemma3 uses GELU with tanh approximation
- **Attention Scaling**: Standard models use `head_dim**-0.5`, Gemma3 uses `query_pre_attn_scalar**-0.5`

**Solution**: Carefully compare official vLLM implementations line by line

---

## 🧪 Testing Guide

### Add Test to test_models.py

```python
def test_newmodel_model():
    """Test function to verify NewModel loading and inference"""
    model_path = os.path.expanduser("pretrained/NewModel-1B-Instruct")
    
    if not os.path.exists(model_path):
        print(f"Skipping NewModel test - model path {model_path} does not exist.")
        return True
    
    try:
        config = AutoConfig.from_pretrained(model_path)
        if config.model_type != "newmodel":
            print(f"Error: Model at {model_path} is not a NewModel (type: {config.model_type})")
            return False
        
        print(f"Loading NewModel from: {model_path}")
        print(f"Model type: {config.model_type}")
        print(f"Hidden size: {config.hidden_size}")
        print(f"Num attention heads: {config.num_attention_heads}")
        print(f"Num key-value heads: {getattr(config, 'num_key_value_heads', config.num_attention_heads)}")
        print(f"Hidden activation: {getattr(config, 'hidden_activation', 'N/A')}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        llm = LLM(model_path, enforce_eager=False, tensor_parallel_size=1)
        
        sampling_params = SamplingParams(temperature=0.6, max_tokens=50)
        prompts = ["Hello, how are you?"]
        prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for prompt in prompts
        ]
        outputs = llm.generate(prompts, sampling_params)
        
        print(f"NewModel result: {outputs[0]['text']!r}")
        llm.exit()
        del llm
        gc.collect()
        return True
        
    except Exception as e:
        print(f"❌ Error during NewModel test: {e}")
        import traceback
        traceback.print_exc()
        return False

# Update main() function
def main():
    newmodel_success = test_newmodel_model()
    print(f"NewModel test: {'✅ PASSED' if newmodel_success else '❌ FAILED'}")
```

### Run Tests
```bash
cd serving/nano-vllm
python test_models.py
```

---

## 🎯 Best Practices

### 1. Implementation Principles
- **Always check model config first** for unique features before implementing
- **Copy exact vLLM implementations** for numerical accuracy - do NOT simplify
- **Use exact normalization layers** (GemmaRMSNorm vs RMSNorm matters critically)
- **Implement layer-specific configurations** when needed
- **Follow existing nano-vLLM patterns** exactly

### 2. Model Sensitivity
**CRITICAL INSIGHT**: Modern models are extremely sensitive to implementation details:
- A single wrong normalization layer can cause complete generation failure
- Missing layer-specific features leads to wrong outputs
- Approximate implementations cause subtle but critical errors

### 3. Error Handling
```python
# Add helpful error messages with config analysis
def __init__(self, config):
    if not hasattr(config, 'num_attention_heads'):
        raise ValueError(f"Config missing required attribute: num_attention_heads")
    
    # Validate complex model features
    if hasattr(config, 'sliding_window_pattern'):
        assert hasattr(config, 'sliding_window'), "sliding_window_pattern requires sliding_window"
        assert hasattr(config, 'rope_local_base_freq'), "Sliding window models need rope_local_base_freq"
    
    # Validate configuration
    assert config.hidden_size % config.num_attention_heads == 0, \
        f"hidden_size must be divisible by num_attention_heads"
```

### 4. Model-Specific Considerations
- **Attention mechanisms**: Standard, GQA, MQA, sliding window, interleaved patterns
- **Normalization**: LayerNorm, RMSNorm, GemmaRMSNorm - **exact type is critical**
- **Activation functions**: SiLU, GELU (none/tanh), Swish - **approximation matters**
- **Bias usage**: Some models use bias everywhere, others nowhere, others selectively
- **RoPE configurations**: Different theta values, scaling methods, layer-specific configs
- **Embedding normalization**: Some models (Gemma3) normalize by sqrt(hidden_size)
- **Attention scaling**: Standard vs query_pre_attn_scalar patterns

### 5. Debugging Strategy
1. **Compare config.json** with vLLM implementation 
2. **Test each component** individually (attention, MLP, normalization)
3. **Check layer patterns** - print layer_idx and configs during initialization
4. **Verify numerical outputs** match expected ranges
5. **Test generation quality** - garbage tokens indicate fundamental issues

---

## 🎉 Summary

Successfully implemented models:
- **Llama**: Full architecture with advanced RoPE scaling and GQA
- **Qwen2**: Correct attention without Q/K normalization, fixed tie_word_embeddings  
- **Qwen3**: Original implementation with Q/K normalization
- **Gemma3**: Complex interleaved sliding window attention, GemmaRMSNorm, embedding normalization

**Key Success Factors**:
1. **Analyze model config first** to identify unique features
2. **Study official vLLM implementation** carefully - copy exactly
3. **Use correct normalization layers** - GemmaRMSNorm vs RMSNorm is critical
4. **Implement layer-specific features** when present (sliding window, etc.)
5. **Test with real models** using test_models.py
6. **Debug systematically** when generation fails
7. **Never simplify or approximate** - modern models are extremely sensitive

**Critical Lesson from Gemma3**: Even tiny implementation differences (wrong normalization, missing layer features) can cause complete generation failure. Always implement exactly as vLLM does.

Following this guide ensures new model implementations maintain the quality, performance, and compatibility standards of nano-vLLM while avoiding common pitfalls discovered during real implementations. 