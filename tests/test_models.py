#!/usr/bin/env python3

"""
Test script for model implementation in nano-vLLM.
This script validates that the model implemented in nanovllm can be loaded and run inference correctly.
"""

import gc
import os
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer, AutoConfig


def test_qwen2_model():
    """Test function to verify Qwen2 model loading and inference"""
    
    # Example Qwen2 model path - user should modify this to their actual model path
    # You can download a Qwen2 model like "Qwen/Qwen2-0.5B" or "Qwen/Qwen2-1.5B" from HuggingFace
    qwen2_path = os.path.expanduser("pretrained/Qwen2.5-1.5B-Instruct")
    
    if not os.path.exists(qwen2_path):
        print(f"Error: Model path {qwen2_path} does not exist.")
        print("Please download a Qwen2 model and update the path in this script.")
        print("Example: Qwen/Qwen2-0.5B-Instruct, Qwen/Qwen2-1.5B-Instruct, etc.")
        return False
    
    try:
        # Check if it's actually a Qwen2 model
        config = AutoConfig.from_pretrained(qwen2_path)
        if config.model_type != "qwen2":
            print(f"Error: Model at {qwen2_path} is not a Qwen2 model (type: {config.model_type})")
            return False
        
        print(f"Loading Qwen2 model from: {qwen2_path}")
        print(f"Model type: {config.model_type}")
        print(f"Hidden size: {config.hidden_size}")
        print(f"Num attention heads: {config.num_attention_heads}")
        print(f"Num key-value heads: {getattr(config, 'num_key_value_heads', config.num_attention_heads)}")
        print(f"Rope theta: {getattr(config, 'rope_theta', 1000000)}")
        
        # Initialize tokenizer and LLM
        tokenizer = AutoTokenizer.from_pretrained(qwen2_path)
        llm = LLM(qwen2_path, enforce_eager=False, tensor_parallel_size=1)
        
        # Test simple generation
        sampling_params = SamplingParams(temperature=0.6, max_tokens=50)
        prompts = [
            "The capital of France is",
            "2 + 2 equals",
            "Please give me a simple Python function to add two numbers"
        ]
        
        # Apply chat template for Qwen2
        prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for prompt in prompts
        ]
        
        print("\nRunning inference...")
        outputs = llm.generate(prompts, sampling_params)
        
        print("\nResults:")
        for prompt, output in zip(prompts, outputs):
            print(f"Prompt: {prompt!r}")
            print(f"Completion: {output['text']!r}")
            print()
        
        llm.exit()
        del llm
        gc.collect()
        print("✅ Qwen2 model test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Error during Qwen2 model test: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_qwen3_model():
    """Test function to verify Qwen3 model still works"""
    qwen3_path = os.path.expanduser("pretrained/Qwen3-0.6B")
    
    if not os.path.exists(qwen3_path):
        print(f"Skipping Qwen3 test - model path {qwen3_path} does not exist.")
        return True
    
    try:
        print(f"Loading Qwen3 model from: {qwen3_path}")
        config = AutoConfig.from_pretrained(qwen3_path)
        print(f"Model type: {config.model_type}")
        
        tokenizer = AutoTokenizer.from_pretrained(qwen3_path)
        llm = LLM(qwen3_path, enforce_eager=False, tensor_parallel_size=1)
        
        sampling_params = SamplingParams(temperature=0.6, max_tokens=30)
        prompts = tokenizer.apply_chat_template(
            [{"role": "user", "content": "Hello, how are you?"}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True
        )
        
        outputs = llm.generate([prompts], sampling_params)
        print(f"Qwen3 result: {outputs[0]['text']!r}")
        print("✅ Qwen3 model test passed!")
        llm.exit()
        del llm
        gc.collect()
        return True
        
    except Exception as e:
        print(f"❌ Error during Qwen3 model test: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_llama_model():
    """Test function to verify Llama model still works"""
    llama_path = os.path.expanduser("pretrained/Llama-3.2-1B-Instruct")
    
    if not os.path.exists(llama_path):
        print(f"Skipping Llama test - model path {llama_path} does not exist.")
        return True
    
    try:
        print(f"Loading Llama model from: {llama_path}")
        config = AutoConfig.from_pretrained(llama_path)
        print(f"Model type: {config.model_type}")
        
        tokenizer = AutoTokenizer.from_pretrained(llama_path)
        llm = LLM(llama_path, enforce_eager=False, tensor_parallel_size=1)
        
        sampling_params = SamplingParams(temperature=0.6, max_tokens=30)
        
        prompts = [
            "The capital of France is",
            "2 + 2 equals",
            "Please give me a roast chicken recipe"
        ]
        
        prompts = [tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        ) for prompt in prompts]
        
        outputs = llm.generate(prompts, sampling_params)
        print("\nResults:")
        for prompt, output in zip(prompts, outputs):
            print(f"Prompt: {prompt!r}")
            print(f"Completion: {output['text']!r}")
            print()
        llm.exit()
        del llm
        gc.collect()
        return True
        
    except Exception as e:
        print(f"❌ Error during Llama model test: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_gemma3_model():
    """Test function to verify Gemma3 loading and inference"""
    model_path = os.path.expanduser("pretrained/gemma-3-4b-it-text-only")
    
    if not os.path.exists(model_path):
        print(f"Skipping Gemma3 test - model path {model_path} does not exist.")
        return True
    
    try:
        config = AutoConfig.from_pretrained(model_path)
        if config.model_type != "gemma3_text":
            print(f"Error: Model at {model_path} is not a gemma3_text (type: {config.model_type})")
            return False
        
        print(f"Loading Gemma3 from: {model_path}")
        print(f"Model type: {config.model_type}")
        print(f"Hidden size: {config.hidden_size}")
        print(f"Num attention heads: {config.num_attention_heads}")
        print(f"Num key-value heads: {getattr(config, 'num_key_value_heads', config.num_attention_heads)}")
        print(f"Hidden activation: {getattr(config, 'hidden_activation', 'N/A')}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        llm = LLM(model_path, enforce_eager=True, tensor_parallel_size=1)
        
        sampling_params = SamplingParams(temperature=0.6, max_tokens=50)
        prompts = [
            "Hello, how are you?",
            "What is the capital of France?",
            "Write a simple Python function to add two numbers"
        ]
        prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for prompt in prompts
        ]
        outputs = llm.generate(prompts, sampling_params)
        
        print("\nResults:")
        for prompt, output in zip(prompts, outputs):
            print(f"Prompt: {prompt!r}")
            print(f"Completion: {output['text']!r}")
            print()
        
        print(f"✅ Gemma3 result: {outputs[0]['text']!r}")
        llm.exit()
        del llm
        gc.collect()
        return True
        
    except Exception as e:
        print(f"❌ Error during Gemma3 test: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("Testing nano-vllm with Gemma3 support...")
    print("=" * 50)
    
    # Test existing functionality first to ensure we didn't break anything
    print("1. Testing Qwen3 model (existing functionality):")
    qwen3_success = test_qwen3_model()
    
    print("\n" + "=" * 50)
    print("2. Testing Llama model (existing functionality):")
    llama_success = test_llama_model()
    
    print("\n" + "=" * 50)
    print("3. Testing Qwen2 model (existing functionality):")
    qwen2_success = test_qwen2_model()
    
    print("\n" + "=" * 50)
    print("4. Testing Gemma3 model (new functionality):")
    gemma3_success = test_gemma3_model()
    
    print("\n" + "=" * 50)
    print("Test Summary:")
    print(f"Qwen3 test: {'✅ PASSED' if qwen3_success else '❌ FAILED'}")
    print(f"Llama test: {'✅ PASSED' if llama_success else '❌ FAILED'}")
    print(f"Qwen2 test: {'✅ PASSED' if qwen2_success else '❌ FAILED'}")
    print(f"Gemma3 test: {'✅ PASSED' if gemma3_success else '❌ FAILED'}")
    


if __name__ == "__main__":
    main() 