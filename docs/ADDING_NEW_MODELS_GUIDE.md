# 📚 Complete Guide: Adding New Model Support to nano-vLLM

Based on successful implementations of Llama and Qwen2 support in nano-vLLM, this guide provides a step-by-step process for adding any new model architecture to the nano-vLLM inference engine.

## 📋 Table of Contents

1. [Quick Start Workflow](#quick-start-workflow)
2. [Implementation Steps](#implementation-steps)
3. [Common Issues & Solutions](#common-issues--solutions)
4. [Testing Guide](#testing-guide)
5. [Best Practices](#best-practices)

---

## 🚀 Quick Start Workflow

1. **Study official vLLM implementation** (`serving/vllm/vllm/model_executor/models/[model_name].py`)
2. **Copy exact architecture** to `nanovllm/models/[model_name].py`
3. **Add model loading** to `nanovllm/engine/model_runner.py`
4. **Add test function** to `test_models.py`
5. **Test with real model** (if available)

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

from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.attention import Attention
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import QKVParallelLinear, MergedColumnParallelLinear, RowParallelLinear
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead

class ModelAttention(nn.Module):
    # Copy exact vLLM implementation
    pass

class ModelMLP(nn.Module):
    # Copy exact vLLM implementation
    pass

class ModelDecoderLayer(nn.Module):
    # Copy exact vLLM implementation
    pass

class ModelNameModel(nn.Module):
    # Copy exact vLLM implementation
    pass

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
elif hf_config.model_type == "qwen3":
    self.model = Qwen3ForCausalLM(hf_config)
# ... other models

# Update error message
else:
    raise ValueError(f"Unsupported model type: {hf_config.model_type}. "
                   "Supported types: qwen3, qwen2, llama, newmodel")
```

---

## ⚠️ Common Issues & Solutions

### Issue 1: tie_word_embeddings Runtime Error
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

### Issue 2: RoPE Scaling Assertion Error
**Error**: `AssertionError` in `rotary_embedding.py`
```python
assert rope_scaling is None
```

**Solution**: Implement full RoPE scaling support following vLLM patterns

### Issue 3: KV Cache Allocation Failure
**Error**: `AssertionError: config.num_kvcache_blocks > 0`

**Solution**: Implement vLLM-compatible cache calculation with proper head dimension handling

### Issue 4: Model Architecture Differences
**Problem**: Subtle differences between model generations (e.g., Qwen2 vs Qwen3)

**Key Differences Found**:
- **Q/K Normalization**: Qwen2 has NO q_norm/k_norm layers, Qwen3 has RMSNorm
- **QKV Bias**: Qwen2 always uses `bias=True`, Qwen3 uses configurable bias
- **Tensor Parallel Logic**: Different KV head distribution patterns

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
- **Copy exact vLLM implementations** for numerical accuracy
- **Don't simplify or optimize** unless absolutely necessary
- **Follow existing nano-vLLM patterns** exactly

### 2. Error Handling
```python
# Add helpful error messages
def __init__(self, config):
    if not hasattr(config, 'num_attention_heads'):
        raise ValueError(f"Config missing required attribute: num_attention_heads")
    
    # Validate configuration
    assert config.hidden_size % config.num_attention_heads == 0, \
        f"hidden_size must be divisible by num_attention_heads"
```

### 3. Key Implementation Insights
1. **Architecture Precision Matters**: Even small differences like normalization layers can completely change model behavior
2. **tie_word_embeddings is Tricky**: Always use the pattern of creating separate layers and copying weights
3. **Follow Existing Patterns**: nano-vLLM has established patterns that work - follow them exactly, while the implementation is exactly official vllm

### 4. Model-Specific Considerations
- **Attention mechanisms**: Different models have vastly different attention implementations
- **Normalization**: LayerNorm vs RMSNorm, pre vs post, Q/K normalization variations  
- **Bias usage**: Some models use bias everywhere, others nowhere, others selectively
- **RoPE configurations**: Different theta values, scaling methods, partial rotation
- **KV head configurations**: GQA, MQA, standard attention variations

---

## 🎉 Summary

Successfully implemented models:
- **Llama**: Full architecture with advanced RoPE scaling and GQA
- **Qwen2**: Correct attention without Q/K normalization, fixed tie_word_embeddings
- **Qwen3**: Original implementation with Q/K normalization

**Key Success Factors**:
1. Study official vLLM implementation carefully
2. Copy exact implementations for numerical accuracy
3. Use test_models.py for comprehensive testing
4. Debug tie_word_embeddings carefully
5. Maintain backward compatibility

Following this guide ensures new model implementations maintain the quality, performance, and compatibility standards of nano-vLLM while avoiding common pitfalls discovered during real implementations. 