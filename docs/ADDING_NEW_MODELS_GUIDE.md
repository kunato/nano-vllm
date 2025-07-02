# 📚 Complete Guide: Adding New Model Support to nano-vLLM

Based on the successful implementation of Llama support in nano-vLLM, this guide provides a comprehensive step-by-step process for adding any new model architecture to the nano-vLLM inference engine.

## 📋 Table of Contents

1. [Overview & Architecture Understanding](#overview--architecture-understanding)
2. [Prerequisites & Setup](#prerequisites--setup)
3. [Step-by-Step Implementation Process](#step-by-step-implementation-process)
4. [File Structure & References](#file-structure--references)
5. [Common Issues & Solutions](#common-issues--solutions)
6. [Testing & Validation](#testing--validation)
7. [Best Practices](#best-practices)

---

## 🏗️ Overview & Architecture Understanding

### nano-vLLM Architecture
nano-vLLM is a trimmed-down version of the official vLLM inference engine, designed for:
- **Simplicity**: Fewer abstractions, easier to understand
- **Performance**: Focus on core inference functionality
- **Compatibility**: Following vLLM patterns for numerical accuracy

### Key Components
```
nano-vllm/
├── nanovllm/
│   ├── models/           # Model implementations
│   ├── layers/           # Reusable layer components
│   ├── engine/           # Core inference engine
│   └── utils/            # Utilities and helpers
```

### Model Integration Points
1. **Model Architecture** (`nanovllm/models/`)
2. **Layer Components** (`nanovllm/layers/`)
3. **Model Loading** (`nanovllm/engine/model_runner.py`)
4. **Configuration** (`nanovllm/config.py`)

---

## 🔧 Prerequisites & Setup

### 1. Environment Setup
```bash
cd serving/nano-vllm
pip install -e .
```

### 2. Required Knowledge
- PyTorch and transformer architectures
- Understanding of attention mechanisms
- Familiarity with vLLM codebase structure
- Knowledge of the target model architecture

### 3. Reference Materials
- **Official vLLM repo**: `serving/vllm/`
- **Target model's HuggingFace implementation**
- **Model configuration files**

---

## 🚀 Step-by-Step Implementation Process

### Step 1: Analysis Phase

#### 1.1 Study the Target Model
```bash
# Example: Analyze model config
python -c "
from transformers import AutoConfig
config = AutoConfig.from_pretrained('path/to/model')
print('Model type:', config.model_type)
print('Architecture:', config.architectures)
print('Key parameters:', {k: v for k, v in config.__dict__.items() if not k.startswith('_')})
"
```

#### 1.2 Find vLLM Reference Implementation
**Key vLLM files to examine:**
- `serving/vllm/vllm/model_executor/models/[model_name].py`
- `serving/vllm/vllm/model_executor/layers/`
- `serving/vllm/vllm/config.py` (for configuration methods)

#### 1.3 Identify Required Components
Common components for most models:
- **Attention mechanism** (Self-attention, GQA, MQA)
- **Feed-forward network** (MLP with activation)
- **Normalization layers** (LayerNorm, RMSNorm)
- **Positional encoding** (RoPE, absolute, etc.)
- **Embedding layers**

### Step 2: Layer Implementation

#### 2.1 Implement Core Layers

Create reusable layers in `nanovllm/layers/` if they don't exist:

**Example: RoPE Implementation** (if needed)
```python
# File: nanovllm/layers/rotary_embedding.py
# Reference: serving/vllm/vllm/model_executor/layers/rotary_embedding.py

class NewModelRotaryEmbedding(nn.Module):
    def __init__(self, head_size, rotary_dim, max_position_embeddings, base, rope_scaling=None):
        # Copy exact vLLM implementation
        super().__init__()
        # ... exact vLLM code
```

**Key principles:**
- **Copy exact vLLM implementation** for numerical accuracy
- Use existing nano-vLLM layers when possible
- Add new layers only when necessary

#### 2.2 Update Existing Layers

If existing layers need modification (like we did with rotary embedding):

```python
# Example: Adding new rope scaling types
def get_rope(head_size, rotary_dim, max_position_embeddings, base, rope_scaling, is_neox_style=True):
    if rope_scaling is None:
        return RotaryEmbedding(head_size, rotary_dim, max_position_embeddings, base, is_neox_style)
    
    # Add new scaling types
    rope_type = rope_scaling.get("rope_type", rope_scaling.get("type"))
    if rope_type == "llama3":
        return Llama3RotaryEmbedding(head_size, rotary_dim, max_position_embeddings, base, rope_scaling, is_neox_style)
    # ... other types
```

### Step 3: Model Architecture Implementation

#### 3.1 Create Model File

**File**: `nanovllm/models/[model_name].py`  
**Reference**: `serving/vllm/vllm/model_executor/models/[model_name].py`

**Template Structure:**
```python
# File: nanovllm/models/llama.py
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from transformers import LlamaConfig

from ..layers.activation import SiluAndMul
from ..layers.attention import PagedAttention
from ..layers.layernorm import RMSNorm
from ..layers.linear import (
    LinearMethodBase, 
    MergedColumnParallelLinear,
    QKVParallelLinear, 
    RowParallelLinear
)
from ..layers.rotary_embedding import get_rope
from ..layers.sampler import Sampler
from ..utils.weight_utils import (
    convert_pyslice_to_tensor,
    default_weight_loader,
    hf_model_weights_iterator
)

class ModelNameAttention(nn.Module):
    """Copy exact implementation from vLLM"""
    def __init__(self, config, linear_method=None):
        # Exact vLLM implementation
        pass

class ModelNameMLP(nn.Module):
    """Copy exact implementation from vLLM"""
    pass

class ModelNameDecoderLayer(nn.Module):
    """Copy exact implementation from vLLM"""
    pass

class ModelNameModel(nn.Module):
    """Copy exact implementation from vLLM"""
    pass

class ModelNameForCausalLM(nn.Module):
    """Main model class"""
    pass
```

### Step 4: Model Loading Integration

#### 4.1 Update Model Runner

**File**: `nanovllm/engine/model_runner.py`

```python
def load_model(self):
    config = self.config
    hf_config = config.hf_config
    
    # Add new model type detection
    if hf_config.model_type == "llama":
        from ..models.llama import LlamaForCausalLM
        model = LlamaForCausalLM(hf_config)
    elif hf_config.model_type == "qwen3":
        from ..models.qwen3 import Qwen3ForCausalLM  
        model = Qwen3ForCausalLM(hf_config)
    else:
        raise ValueError(f"Unsupported model type: {hf_config.model_type}")
    
    model.load_weights(config.model_path)
    return model.cuda().eval()
```

#### 4.2 Update Cache Allocation

Implement vLLM-compatible cache calculation:

```python
class NanoVLLMModelConfig:
    """Helper class that implements vLLM's ModelConfig methods for cache calculation."""
    
    def __init__(self, hf_config, tensor_parallel_size=1):
        self.hf_config = hf_config
        self.tensor_parallel_size = tensor_parallel_size
    
    def get_head_size(self) -> int:
        """Get head size following exact vLLM implementation."""
        if getattr(self.hf_config, "head_dim", None) is not None:
            return self.hf_config.head_dim
        return (self.hf_config.hidden_size // self.hf_config.num_attention_heads)
    
    def get_total_num_kv_heads(self) -> int:
        """Returns the total number of KV heads following exact vLLM implementation."""
        attributes = ["n_head_kv", "num_kv_heads", "num_key_value_heads", "multi_query_group_num"]
        for attr in attributes:
            num_kv_heads = getattr(self.hf_config, attr, None)
            if num_kv_heads is not None:
                return num_kv_heads
        return self.hf_config.num_attention_heads
```

---

## 📁 File Structure & References

### Files You Need to Create/Modify

| File Path | Purpose | vLLM Reference |
|-----------|---------|----------------|
| `nanovllm/models/[model].py` | Main model implementation | `vllm/model_executor/models/[model].py` |
| `nanovllm/engine/model_runner.py` | Model loading and cache allocation | `vllm/worker/cache_engine.py`, `vllm/config.py` |
| `nanovllm/layers/rotary_embedding.py` | RoPE implementation (if needed) | `vllm/model_executor/layers/rotary_embedding.py` |
| `nanovllm/layers/attention.py` | Attention mechanisms (if needed) | `vllm/attention/` |

### Key vLLM Reference Files

1. **Model Implementation**: 
   - `serving/vllm/vllm/model_executor/models/llama.py`
   - `serving/vllm/vllm/model_executor/models/qwen.py`

2. **Layer Components**:
   - `serving/vllm/vllm/model_executor/layers/rotary_embedding.py`
   - `serving/vllm/vllm/model_executor/layers/attention.py`
   - `serving/vllm/vllm/model_executor/layers/activation.py`
   - `serving/vllm/vllm/model_executor/layers/linear.py`

3. **Configuration**:
   - `serving/vllm/vllm/config.py` (lines 1127-1225 for cache calculations)
   - `serving/vllm/vllm/worker/cache_engine.py`

---

## ⚠️ Common Issues & Solutions

### Issue 1: RoPE Scaling Assertion Error
**Error**: `AssertionError` in `rotary_embedding.py`
```python
assert rope_scaling is None
```

**Solution**: Implement full RoPE scaling support following vLLM patterns

### Issue 2: KV Cache Allocation Failure
**Error**: `AssertionError: config.num_kvcache_blocks > 0`

**Solution**: Implement vLLM-compatible cache calculation with proper head dimension handling

### Issue 3: Weight Loading Mismatches
**Error**: Shape mismatches during weight loading

**Solution**: Implement proper parameter mapping for merged weights (QKV, gate_up, etc.)

---

## 🧪 Testing & Validation

### Test Script Template
```python
# test_new_model.py
from nanovllm import LLM

def test_new_model():
    print("Testing new model...")
    
    # Test model loading
    llm = LLM('path/to/model', enforce_eager=True, tensor_parallel_size=1)
    
    # Test inference
    prompts = ["Hello, how are you?", "What is AI?"]
    results = llm.generate(prompts, max_tokens=50)
    
    for prompt, result in zip(prompts, results):
        print(f"Prompt: {prompt}")
        print(f"Result: {result.outputs[0].text}")

if __name__ == "__main__":
    test_new_model()
```

### Validation Checklist

- [ ] **Model loads without errors**
- [ ] **KV cache allocation succeeds**
- [ ] **Weight loading completes**
- [ ] **Forward pass works**
- [ ] **Text generation produces coherent output**
- [ ] **Existing models still work (backward compatibility)**

---

## 🎯 Best Practices

### 1. Follow vLLM Patterns Exactly
- **Copy implementations directly** from vLLM for numerical accuracy
- **Don't simplify or optimize** unless absolutely necessary
- **Maintain exact parameter names** and shapes

### 2. Incremental Development
1. Start with basic model structure
2. Add layer by layer
3. Test each component separately
4. Integrate gradually

### 3. Error Handling
```python
# Add helpful error messages
def __init__(self, config):
    if not hasattr(config, 'num_attention_heads'):
        raise ValueError(f"Config missing required attribute: num_attention_heads")
    
    # Validate configuration
    assert config.hidden_size % config.num_attention_heads == 0, \
        f"hidden_size must be divisible by num_attention_heads"
```

---

## 🎉 Summary

The key to successfully adding new models to nano-vLLM is:

1. **Study the official vLLM implementation carefully**
2. **Copy exact implementations for numerical accuracy**
3. **Test incrementally at each step**
4. **Handle edge cases and configuration variations**
5. **Maintain backward compatibility**

This process was successfully demonstrated with the Llama implementation, which now supports:
- ✅ Full Llama architecture (Attention, MLP, Decoder layers)
- ✅ Advanced RoPE scaling (llama3, linear, ntk, dynamic)
- ✅ Grouped Query Attention (GQA)
- ✅ Proper KV cache allocation
- ✅ Weight loading with parameter mapping
- ✅ Backward compatibility with existing Qwen3 models

Following this guide ensures that new model implementations maintain the quality, performance, and compatibility standards of nano-vLLM. 