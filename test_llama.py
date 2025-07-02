import gc
import os
import sys
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer, AutoConfig


def test_llama_model():
    """Test function to verify Llama model loading and inference"""
    
    # Example Llama model path - user should modify this to their actual model path
    # You can download a Llama model like "meta-llama/Llama-3.2-1B" from HuggingFace
    llama_path = os.path.expanduser("pretrained/Llama-3.2-1B-Instruct")
    
    if not os.path.exists(llama_path):
        print(f"Error: Model path {llama_path} does not exist.")
        print("Please download a Llama model and update the path in this script.")
        print("Example: meta-llama/Llama-3.2-1B or any other Llama model")
        return False
    
    try:
        # Check if it's actually a Llama model
        config = AutoConfig.from_pretrained(llama_path)
        if config.model_type != "llama":
            print(f"Error: Model at {llama_path} is not a Llama model (type: {config.model_type})")
            return False
        
        print(f"Loading Llama model from: {llama_path}")
        print(f"Model type: {config.model_type}")
        print(f"Hidden size: {config.hidden_size}")
        print(f"Num attention heads: {config.num_attention_heads}")
        print(f"Num key-value heads: {getattr(config, 'num_key_value_heads', config.num_attention_heads)}")
        
        # Initialize tokenizer and LLM
        tokenizer = AutoTokenizer.from_pretrained(llama_path)
        llm = LLM(llama_path, enforce_eager=False, tensor_parallel_size=1)
        
        # Test simple generation
        sampling_params = SamplingParams(temperature=0.6, max_tokens=50)
        prompts = [
            "The capital of France is",
            "2 + 2 equals",
            "Please give me a roast chicken recipe"
        ]
        
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
        print("✅ Llama model test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Error during Llama model test: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_qwen3_model():
    """Test function to verify Qwen3 model still works"""
    qwen_path = os.path.expanduser("pretrained/Qwen3-0.6B")
    
    if not os.path.exists(qwen_path):
        print(f"Skipping Qwen3 test - model path {qwen_path} does not exist.")
        return True
    
    try:
        print(f"Loading Qwen3 model from: {qwen_path}")
        config = AutoConfig.from_pretrained(qwen_path)
        print(f"Model type: {config.model_type}")
        
        tokenizer = AutoTokenizer.from_pretrained(qwen_path)
        llm = LLM(qwen_path, enforce_eager=False, tensor_parallel_size=1)
        
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


def main():
    print("Testing nano-vllm with Llama support...")
    print("=" * 50)
    
    # Test Qwen3 first to ensure we didn't break existing functionality
    print("1. Testing Qwen3 model (existing functionality):")
    qwen_success = test_qwen3_model()
    
    print("\n" + "=" * 50)
    print("2. Testing Llama model (new functionality):")
    llama_success = test_llama_model()
    
    print("\n" + "=" * 50)
    print("Test Summary:")
    print(f"Qwen3 test: {'✅ PASSED' if qwen_success else '❌ FAILED'}")
    print(f"Llama test: {'✅ PASSED' if llama_success else '❌ FAILED'}")
    
    if llama_success:
        print("\n🎉 Llama support successfully implemented in nano-vllm!")
    else:
        print("\n💥 Llama support implementation needs fixes.")
        sys.exit(1)


if __name__ == "__main__":
    main() 