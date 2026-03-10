"""
Local Model Configuration for cal_emb_flexible.py
Generated automatically by local_model_config.py

This file contains configurations for using locally cached models.
"""

import os
from pathlib import Path

# Local model configurations
# Qwen2.5-VL (2024-2025): 3B, 7B (0.5B/1.5B are text-only; no VL variants on HF).
# Qwen3.5 series (2026): 800M, 2B, 4B — add local_path when cached.
LOCAL_MODELS = {
    # Qwen2-VL
    "qwen2-vl-2b": {
        "local_path": "/Users/f0s03xp/Debias_VLMs/models_cache/qwen2-vl-2b",
        "hf_name": "Qwen/Qwen2-VL-2B-Instruct",
        "hidden_size": 1280,
        "recommended": False
    },
    "qwen2-vl-7b": {
        "local_path": "/Users/f0s03xp/Debias_VLMs/models_cache/qwen2-vl-7b",
        "hf_name": "Qwen/Qwen2-VL-7B-Instruct",
        "hidden_size": 3584,
        "recommended": False
    },
    # Qwen2.5-VL (default series): 3B, 7B
    "qwen2.5-vl-3b": {
        "local_path": "/Users/f0s03xp/Debias_VLMs/models_cache/qwen2.5-vl-3b",
        "hf_name": "Qwen/Qwen2.5-VL-3B-Instruct",
        "hidden_size": 2048,
        "recommended": True
    },
    "qwen2.5-vl-7b": {
        "local_path": "/Users/f0s03xp/Debias_VLMs/models_cache/qwen2.5-vl-7b",
        "hf_name": "Qwen/Qwen2.5-VL-7B-Instruct",
        "hidden_size": 3584,
        "recommended": False
    },
    # Qwen3.5 series (2026): 800M, 2B, 4B
    "qwen3.5-vl-800m": {
        "local_path": "/Users/f0s03xp/Debias_VLMs/models_cache/qwen3.5-vl-800m",
        "hf_name": "Qwen/Qwen3.5-VL-0.8B-Instruct",
        "hidden_size": 2048,
        "recommended": False
    },
    "qwen3.5-vl-2b": {
        "local_path": "/Users/f0s03xp/Debias_VLMs/models_cache/qwen3.5-vl-2b",
        "hf_name": "Qwen/Qwen3.5-VL-2B-Instruct",
        "hidden_size": 2048,
        "recommended": False
    },
    "qwen3.5-vl-4b": {
        "local_path": "/Users/f0s03xp/Debias_VLMs/models_cache/qwen3.5-vl-4b",
        "hf_name": "Qwen/Qwen3.5-VL-4B-Instruct",
        "hidden_size": 2560,
        "recommended": False
    },
}

def get_local_model_path(model_name: str) -> str:
    """Get local path for a model if available, otherwise return original name"""
    
    # Check if it's already a local path
    if os.path.exists(model_name):
        return model_name
    
    # Check HuggingFace names
    for model_id, config in LOCAL_MODELS.items():
        if model_name == config.get("hf_name"):
            local_path = config["local_path"]
            if os.path.exists(local_path):
                print(f"Using local model: {local_path}")
                return local_path
    
    # Check local model IDs
    if model_name in LOCAL_MODELS:
        local_path = LOCAL_MODELS[model_name]["local_path"]
        if os.path.exists(local_path):
            print(f"Using local model: {local_path}")
            return local_path
    
    # Return original if not found locally
    print(f"Model {model_name} not found locally, using HuggingFace")
    return model_name

def get_recommended_local_model() -> str:
    """Get the best available local model (default: Qwen2.5-VL 3B)."""
    # Prefer default: Qwen2.5-VL 3B
    if "qwen2.5-vl-3b" in LOCAL_MODELS:
        return LOCAL_MODELS["qwen2.5-vl-3b"]["local_path"]
    if "qwen2.5-vl-7b" in LOCAL_MODELS:
        return LOCAL_MODELS["qwen2.5-vl-7b"]["local_path"]
    if "qwen2-vl-2b" in LOCAL_MODELS:
        return LOCAL_MODELS["qwen2-vl-2b"]["local_path"]
    if "qwen2-vl-7b" in LOCAL_MODELS:
        return LOCAL_MODELS["qwen2-vl-7b"]["local_path"]
    for config in LOCAL_MODELS.values():
        if config.get("recommended", False):
            return config["local_path"]
    return "Qwen/Qwen2.5-VL-3B-Instruct"

# Quick access functions
def list_local_models():
    """List all available local models"""
    print("Available Local Models:")
    print("=" * 50)
    for model_id, config in LOCAL_MODELS.items():
        status = "✅" if os.path.exists(config["local_path"]) else "❌"
        recommended = "⭐" if config.get("recommended", False) else "  "
        print(f"{status} {recommended} {model_id}")
        print(f"    Path: {config['local_path']}")
        print(f"    HF Name: {config['hf_name']}")
        print(f"    Hidden Size: {config['hidden_size']}")
        print()
