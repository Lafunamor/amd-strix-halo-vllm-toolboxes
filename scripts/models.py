MODEL_TABLE = {
    # 1. Llama 3.1 8B Instruct
    # MAD uses 131k tokens. We scale to 32k for 32GB VRAM safety.
    "meta-llama/Meta-Llama-3.1-8B-Instruct": {
        "trust_remote": False,
        "valid_tp": [1, 2],
        "max_num_seqs": "64", # Strix Halo Optimized (Bus Batch Scale)
        "max_tokens": "32768" 
    },
    
    "google/gemma-3-12b-it": {
        "trust_remote": False,
        "enforce_eager": True,
        "valid_tp": [1, 2],
        "max_num_seqs": "64",
        "max_tokens": "32768" 
    },
    # 2. GPT-OSS 20B (MXFP4)
    # MAD Row 0 uses 8192. We match this exactly.
    "openai/gpt-oss-20b": {
        "trust_remote": True,
        "valid_tp": [1, 2],
        "max_num_seqs": "64",
        "max_tokens": "8192"
    },
    
    "openai/gpt-oss-120b": {
        "trust_remote": True,
        "valid_tp": [1],
        "max_num_seqs": "16",
        "max_tokens": "8192"
    },

    "Qwen/Qwen3.5-35B-A3B": {
        "trust_remote": True,
        "valid_tp": [1],
        "max_num_seqs": "16",
        "max_tokens": "8192"
    },

    "Qwen/Qwen3-14B-AWQ": {
        "trust_remote": True,
        "valid_tp": [1], # Too big for single GPU
        "max_num_seqs": "64", # Strix Halo Optimized
        "max_tokens": "16384", # Lower batch size because Eager mode is CPU intensive
        "enforce_eager": False, 
        "env": {"VLLM_USE_TRITON_AWQ": "1"} # Fixes "Unsupported Hardware" error
    },

    "cyankiwi/Qwen3.5-122B-A10B-AWQ-4bit": {
        "trust_remote": True,
        "valid_tp": [1,2], # Too big for single GPU
        "max_num_seqs": "64", # Strix Halo Optimized
        "max_tokens": "16384", # Lower batch size because Eager mode is CPU intensive
        "enforce_eager": True, 
        "env": {"VLLM_USE_TRITON_AWQ": "1"} # Fixes "Unsupported Hardware" error
    },


    # 4. Qwen 30B 4-bit
    "btbtyler09/Qwen3-Coder-30B-A3B-Instruct-gptq-4bit": {
        "trust_remote": True,
        "enforce_eager": False, 
        "valid_tp": [1, 2],
        "max_num_seqs": "64",
        "max_tokens": "32768"
    },

    "btbtyler09/Qwen3-Coder-30B-A3B-Instruct-gptq-8bit": {
        "trust_remote": True,
        "enforce_eager": False, 
        "valid_tp": [1, 2],
        "max_num_seqs": "64",
        "max_tokens": "32768"
    },

    "zai-org/GLM-4.7-Flash": {
        "trust_remote": True,
        "enforce_eager": False, 
        "valid_tp": [1, 2],
        "max_num_seqs": "64",
        "max_tokens": "32768",
    },

    # 5. Qwen 80B AWQ
    # Size: ~48GB. Fits on 2x32GB (64GB). Leftover for Cache: ~16GB.
    # Config: 20k ctx fits in that cache. Eager mode required for stability.
    "dazipe/Qwen3-Next-80B-A3B-Instruct-GPTQ-Int4A16": {
        "trust_remote": True,
        "valid_tp": [1], # Too big for single GPU
        "max_num_seqs": "64", # Large Model / Bandwidth Constrained
        "max_tokens": "16384", # Lower batch size because Eager mode is CPU intensive
        "enforce_eager": True, 
        "env": {"VLLM_USE_TRITON_AWQ": "1"} # Fixes "Unsupported Hardware" error
    },

    "mratsim/MiniMax-M2.5-BF16-INT4-AWQ": {
        "trust_remote": True,
        "valid_tp": [2],
        "max_num_seqs": "64",
        "max_tokens": "16384",
        "enforce_eager": False,
        "env": {"VLLM_USE_TRITON_AWQ": "1"} # Fixes "Unsupported Hardware" error
    },

}

MODELS_TO_RUN = [
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "google/gemma-3-12b-it",
    "Qwen/Qwen3-14B-AWQ",
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
    "zai-org/GLM-4.7-Flash",
    "btbtyler09/Qwen3-Coder-30B-A3B-Instruct-gptq-4bit",
    "btbtyler09/Qwen3-Coder-30B-A3B-Instruct-gptq-8bit",
    "dazipe/Qwen3-Next-80B-A3B-Instruct-GPTQ-Int4A16",
    "Qwen/Qwen3.5-35B-A3B",
    "cyankiwi/Qwen3.5-122B-A10B-AWQ-4bit"
]

# Hardware / Global Defaults
GPU_UTIL = "0.90"
OFF_NUM_PROMPTS = 200 # Increased for Strix Halo (Steady State Saturation)
OFF_FORCED_OUTPUT = "512"
DEFAULT_BATCH_TOKENS = "8192"
