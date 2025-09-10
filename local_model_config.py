"""
Local Model Configuration for cal_emb_flexible.py
Generated automatically by local_model_config.py

This file contains configurations for using locally cached models.
"""

import os
from pathlib import Path

# Local model configurations
LOCAL_MODELS = {
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
        "recommended": True
    },
    "qwen2.5-vl-3b": {
        "local_path": "/Users/f0s03xp/Debias_VLMs/models_cache/qwen2.5-vl-3b",
        "hf_name": "Qwen/Qwen2.5-VL-3B-Instruct",
        "hidden_size": 2048,
        "recommended": False
    },
    "qwen2.5-vl-7b": {
        "local_path": "/Users/f0s03xp/Debias_VLMs/models_cache/qwen2.5-vl-7b",
        "hf_name": "Qwen/Qwen2.5-VL-7B-Instruct",
        "hidden_size": 3584,
        "recommended": False
    }
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
    """Get the best available local model"""
    # Prefer qwen2-vl-7b
    if "qwen2-vl-7b" in LOCAL_MODELS:
        return LOCAL_MODELS["qwen2-vl-7b"]["local_path"]
    
    # Then qwen2-vl-2b  
    if "qwen2-vl-2b" in LOCAL_MODELS:
        return LOCAL_MODELS["qwen2-vl-2b"]["local_path"]
    
    # Any available model
    for config in LOCAL_MODELS.values():
        if config.get("recommended", False):
            return config["local_path"]
    
    # Fallback to HuggingFace
    return "Qwen/Qwen2-VL-7B-Instruct"

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
