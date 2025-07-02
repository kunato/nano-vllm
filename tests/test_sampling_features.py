#!/usr/bin/env python3
"""
Test script for new sampling features in nano-vllm:
- repetition_penalty
- top_p 
- stop sequences (both strings and token IDs)
"""

from nanovllm import LLM, SamplingParams


def test_repetition_penalty():
    """Test repetition penalty feature."""
    print("=== Testing Repetition Penalty ===")
    
    llm = LLM("pretrained/gemma-3-4b-it-text-only")  # Small model for testing
    tokenizer = llm.tokenizer
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "The weather today is"}],
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Test with different repetition penalties
    test_cases = [
        (1.0, "No repetition penalty"),
        (1.2, "Moderate repetition penalty"),
        (1.5, "High repetition penalty"),
    ]
    
    for penalty, description in test_cases:
        print(f"\n{description} (penalty={penalty}):")
        sampling_params = SamplingParams(
            temperature=0.8,
            max_tokens=50,
            repetition_penalty=penalty
        )
        outputs = llm.generate([prompt], sampling_params)
        print(f"Output: {outputs[0]['text']}")
    llm.exit()
    del llm


def test_top_p():
    """Test top-p (nucleus) sampling feature."""
    print("\n=== Testing Top-P Sampling ===")
    
    llm = LLM("pretrained/gemma-3-4b-it-text-only")
    tokenizer = llm.tokenizer
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Once upon a time"}],
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Test with different top_p values
    test_cases = [
        (1.0, "All tokens considered"),
        (0.9, "Top 90% probability mass"),
        (0.5, "Top 50% probability mass"),
        (0.1, "Top 10% probability mass"),
    ]
    
    for top_p, description in test_cases:
        print(f"\n{description} (top_p={top_p}):")
        sampling_params = SamplingParams(
            temperature=0.8,
            max_tokens=50,
            top_p=top_p
        )
        outputs = llm.generate([prompt], sampling_params)
        print(f"Output: {outputs[0]['text']}")
    llm.exit()
    del llm


def test_stop_strings():
    """Test stop string functionality."""
    print("\n=== Testing Stop Strings ===")
    
    llm = LLM("pretrained/gemma-3-4b-it-text-only")
    tokenizer = llm.tokenizer
    
    # Test single stop string
    print("\nTesting single stop string:")
    prompt = "Count from 1 to 10: 1, 2, 3, 4, 5"
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True
    )
    sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=100,
        stop="10"
    )
    outputs = llm.generate([prompt], sampling_params)
    print(f"Prompt: {prompt}")
    print(f"Output: {outputs[0]['text']}")
    print(f"Should stop before '10'")
    
    # Test multiple stop strings
    print("\nTesting multiple stop strings:")
    prompt = "Write a short story. It was a dark and stormy night"
    sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=200,
        stop=[".", "!", "?"]
    )
    outputs = llm.generate([prompt], sampling_params)
    print(f"Prompt: {prompt}")
    print(f"Output: {outputs[0]['text']}")
    print(f"Should stop at first punctuation mark")
    llm.exit()
    del llm


def test_stop_token_ids():
    """Test stop token IDs functionality."""
    print("\n=== Testing Stop Token IDs ===")
    
    llm = LLM("pretrained/gemma-3-4b-it-text-only")
    tokenizer = llm.tokenizer
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Hello world"}],
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Get some common token IDs
    try:
        period_tokens = tokenizer.encode(".", add_special_tokens=False)
        period_id = period_tokens[0] if period_tokens else None
    except:
        period_id = None
    
    if period_id is not None:
        print(f"\nTesting stop token ID (period token: {period_id}):")
        sampling_params = SamplingParams(
            temperature=0.7,
            max_tokens=100,
            stop_token_ids=[period_id]
        )
        outputs = llm.generate([prompt], sampling_params)
        print(f"Output: {outputs[0]['text']}")
        print(f"Should stop when period token is generated")
    llm.exit()
    del llm


def test_combined_features():
    """Test combining multiple features."""
    print("\n=== Testing Combined Features ===")
    
    llm = LLM("pretrained/gemma-3-4b-it-text-only")
    tokenizer = llm.tokenizer
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "The quick brown fox jumps over the lazy dog. The fox"}],
        tokenize=False,
        add_generation_prompt=True
    )
    
    sampling_params = SamplingParams(
        temperature=0.8,
        max_tokens=100,
        repetition_penalty=1.3,  # Discourage repetition
        top_p=0.9,              # Use nucleus sampling
        stop=["cat", "dog"]     # Stop on animal words
    )
    
    outputs = llm.generate([prompt], sampling_params)
    print(f"Prompt: {prompt}")
    print(f"Output: {outputs[0]['text']}")
    print("Features: repetition_penalty=1.3, top_p=0.9, stop=['cat', 'dog']")
    llm.exit()
    del llm


def test_validation():
    """Test parameter validation."""
    print("\n=== Testing Parameter Validation ===")
    
    # Test invalid repetition_penalty
    try:
        SamplingParams(repetition_penalty=0.0)
        print("ERROR: Should have failed with repetition_penalty=0.0")
    except ValueError as e:
        print(f"✓ Correctly rejected repetition_penalty=0.0: {e}")
    
    # Test invalid top_p
    try:
        SamplingParams(top_p=1.5)
        print("ERROR: Should have failed with top_p=1.5")
    except ValueError as e:
        print(f"✓ Correctly rejected top_p=1.5: {e}")
    
    try:
        SamplingParams(top_p=0.0)
        print("ERROR: Should have failed with top_p=0.0")
    except ValueError as e:
        print(f"✓ Correctly rejected top_p=0.0: {e}")


if __name__ == "__main__":
    print("Testing new nano-vllm sampling features...")
    
    # Test validation first (doesn't require model loading)
    test_validation()
    
    # Test individual features
    try:
        test_repetition_penalty()
        test_top_p()
        test_stop_strings()
        test_stop_token_ids()
        test_combined_features()
        
        print("\n✓ All tests completed successfully!")
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        print("Make sure you have a compatible model downloaded or adjust the model name in the tests.") 