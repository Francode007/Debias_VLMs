#!/usr/bin/env python3
"""
Model Download and Management Script for Qwen Vision-Language Models

This script downloads, caches, and manages Qwen VL models locally to avoid
repeated downloads and version conflicts.
"""

import os
import sys
import json
import shutil
import argparse
from pathlib import Path
from typing import Dict, List, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from huggingface_hub import snapshot_download, hf_hub_download
    from transformers import AutoConfig, AutoProcessor
    import torch
except ImportError as e:
    logger.error(f"Required packages not installed: {e}")
    logger.error("Please install: pip install transformers torch huggingface_hub")
    sys.exit(1)

class ModelManager:
    """Manages downloading and caching of Qwen VL models"""
    
    def __init__(self, cache_dir: str = "./models_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Model configurations
        # Qwen2.5-VL (2024-2025): 3B, 7B (default series). 0.5B/1.5B are text-only on HF.
        # Qwen3.5 series (2026): 800M, 2B, 4B.
        self.model_configs = {
            "qwen2-vl-2b": {
                "hf_name": "Qwen/Qwen2-VL-2B-Instruct",
                "model_class": "Qwen2VLForConditionalGeneration",
                "processor_class": "Qwen2VLProcessor",
                "hidden_size": 1536,
                "description": "Qwen2-VL 2B - smaller, faster",
                "recommended": False,
                "size_gb": 4.5
            },
            "qwen2-vl-7b": {
                "hf_name": "Qwen/Qwen2-VL-7B-Instruct",
                "model_class": "Qwen2VLForConditionalGeneration",
                "processor_class": "Qwen2VLProcessor",
                "hidden_size": 3584,
                "description": "Qwen2-VL 7B - stable and well-supported",
                "recommended": False,
                "size_gb": 14.2
            },
            "qwen2.5-vl-3b": {
                "hf_name": "Qwen/Qwen2.5-VL-3B-Instruct",
                "model_class": "Qwen2_5VLForConditionalGeneration",
                "processor_class": "Qwen2VLProcessor",
                "hidden_size": 2048,
                "description": "Qwen2.5-VL 3B - default; balanced size",
                "recommended": True,
                "size_gb": 6.1
            },
            "qwen2.5-vl-7b": {
                "hf_name": "Qwen/Qwen2.5-VL-7B-Instruct",
                "model_class": "Qwen2_5VLForConditionalGeneration",
                "processor_class": "Qwen2VLProcessor",
                "hidden_size": 3584,
                "description": "Qwen2.5-VL 7B - do not use for testing unless explicitly mentioned",
                "recommended": False,
                "size_gb": 14.2
            },
            "qwen3.5-vl-800m": {
                "hf_name": "Qwen/Qwen3.5-VL-0.8B-Instruct",
                "model_class": "Qwen3_5VLForConditionalGeneration",
                "processor_class": "Qwen2VLProcessor",
                "hidden_size": 2048,
                "description": "Qwen3.5-VL 800M (2026)",
                "recommended": False,
                "size_gb": 2.0
            },
            "qwen3.5-vl-2b": {
                "hf_name": "Qwen/Qwen3.5-VL-2B-Instruct",
                "model_class": "Qwen3_5VLForConditionalGeneration",
                "processor_class": "Qwen2VLProcessor",
                "hidden_size": 2048,
                "description": "Qwen3.5-VL 2B (2026)",
                "recommended": False,
                "size_gb": 4.5
            },
            "qwen3.5-vl-4b": {
                "hf_name": "Qwen/Qwen3.5-VL-4B-Instruct",
                "model_class": "Qwen3_5VLForConditionalGeneration",
                "processor_class": "Qwen2VLProcessor",
                "hidden_size": 2560,
                "description": "Qwen3.5-VL 4B (2026)",
                "recommended": False,
                "size_gb": 8.0
            },
        }
    
    def list_available_models(self):
        """List all available models with their details"""
        print("\n" + "="*80)
        print("AVAILABLE QWEN VL MODELS")
        print("="*80)
        
        for model_id, config in self.model_configs.items():
            status = "✅ RECOMMENDED" if config["recommended"] else "⚠️  EXPERIMENTAL"
            cached = "📁 CACHED" if self.is_model_cached(model_id) else "🌐 REMOTE"
            
            print(f"\n{model_id.upper()}")
            print(f"  Status: {status}")
            print(f"  Storage: {cached}")
            print(f"  HF Name: {config['hf_name']}")
            print(f"  Size: {config['size_gb']} GB")
            print(f"  Hidden Size: {config['hidden_size']}")
            print(f"  Description: {config['description']}")
    
    def is_model_cached(self, model_id: str) -> bool:
        """Check if a model is already cached locally"""
        if model_id not in self.model_configs:
            return False
        
        model_path = self.cache_dir / model_id
        config_file = model_path / "config.json"
        return config_file.exists()
    
    def get_model_info(self, model_id: str) -> Dict:
        """Get detailed information about a model"""
        if model_id not in self.model_configs:
            raise ValueError(f"Model {model_id} not found. Available: {list(self.model_configs.keys())}")
        
        config = self.model_configs[model_id].copy()
        config["cached"] = self.is_model_cached(model_id)
        config["local_path"] = str(self.cache_dir / model_id)
        
        return config
    
    def download_model(self, model_id: str, force_download: bool = False) -> str:
        """Download a model and its components to local cache"""
        if model_id not in self.model_configs:
            raise ValueError(f"Model {model_id} not found. Available: {list(self.model_configs.keys())}")
        
        config = self.model_configs[model_id]
        model_path = self.cache_dir / model_id
        
        # Check if already cached
        if self.is_model_cached(model_id) and not force_download:
            logger.info(f"Model {model_id} already cached at {model_path}")
            return str(model_path)
        
        # Create model directory
        model_path.mkdir(exist_ok=True)
        
        logger.info(f"Downloading {model_id} ({config['size_gb']} GB)...")
        logger.info(f"HuggingFace: {config['hf_name']}")
        logger.info(f"Local path: {model_path}")
        
        try:
            # Download all model files
            snapshot_download(
                repo_id=config["hf_name"],
                local_dir=str(model_path),
                local_dir_use_symlinks=False,  # Use actual files, not symlinks
                resume_download=True,
                ignore_patterns=["*.git*", "README.md", "*.msgpack"]  # Skip unnecessary files
            )
            
            # Create a metadata file
            metadata = {
                "model_id": model_id,
                "hf_name": config["hf_name"],
                "hidden_size": config["hidden_size"],
                "model_class": config["model_class"],
                "processor_class": config["processor_class"],
                "download_date": str(Path(__file__).stat().st_mtime),
                "description": config["description"]
            }
            
            with open(model_path / "model_metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Successfully downloaded {model_id} to {model_path}")
            return str(model_path)
            
        except Exception as e:
            logger.error(f"Failed to download {model_id}: {e}")
            # Clean up partial download
            if model_path.exists():
                shutil.rmtree(model_path)
            raise
    
    def verify_model(self, model_id: str) -> bool:
        """Verify that a cached model is complete and functional"""
        if not self.is_model_cached(model_id):
            logger.error(f"Model {model_id} is not cached")
            return False
        
        model_path = self.cache_dir / model_id
        config = self.model_configs[model_id]
        
        # Check essential files
        essential_files = [
            "config.json",
            "model_metadata.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "processor_config.json"
        ]
        
        missing_files = []
        for file in essential_files:
            if not (model_path / file).exists():
                missing_files.append(file)
        
        if missing_files:
            logger.error(f"Model {model_id} is missing files: {missing_files}")
            return False
        
        # Try to load config
        try:
            model_config = AutoConfig.from_pretrained(str(model_path))
            logger.info(f"Model {model_id} config loaded successfully")
            logger.info(f"  - Architecture: {model_config.architectures}")
            logger.info(f"  - Hidden size: {model_config.hidden_size}")
            logger.info(f"  - Model type: {model_config.model_type}")
            return True
        except Exception as e:
            logger.error(f"Failed to load config for {model_id}: {e}")
            return False
    
    def get_model_files_info(self, model_id: str) -> Dict:
        """Get detailed information about model files"""
        if not self.is_model_cached(model_id):
            raise ValueError(f"Model {model_id} is not cached")
        
        model_path = self.cache_dir / model_id
        files_info = {}
        
        # Define file categories and their purposes
        file_categories = {
            "config.json": {
                "category": "Model Configuration",
                "description": "Contains model architecture configuration, hidden sizes, number of layers, etc.",
                "required": True
            },
            "tokenizer.json": {
                "category": "Tokenizer",
                "description": "Fast tokenizer implementation with vocabulary and encoding rules",
                "required": True
            },
            "tokenizer_config.json": {
                "category": "Tokenizer Configuration", 
                "description": "Tokenizer configuration including special tokens, padding, etc.",
                "required": True
            },
            "processor_config.json": {
                "category": "Processor Configuration",
                "description": "Vision-language processor configuration for handling images and text",
                "required": True
            },
            "preprocessor_config.json": {
                "category": "Image Preprocessor",
                "description": "Image preprocessing configuration (resize, normalize, etc.)",
                "required": False
            },
            "generation_config.json": {
                "category": "Generation Configuration",
                "description": "Text generation parameters (max_length, temperature, etc.)",
                "required": False
            },
            "model_metadata.json": {
                "category": "Custom Metadata",
                "description": "Our custom metadata with model information and download details",
                "required": False
            }
        }
        
        # Add model weights files
        for file_path in model_path.glob("*.safetensors"):
            file_categories[file_path.name] = {
                "category": "Model Weights",
                "description": "SafeTensors format model weights (recommended format)",
                "required": True
            }
        
        for file_path in model_path.glob("*.bin"):
            file_categories[file_path.name] = {
                "category": "Model Weights (Legacy)",
                "description": "PyTorch format model weights (legacy format)",
                "required": True
            }
        
        # Scan actual files
        for file_path in model_path.iterdir():
            if file_path.is_file():
                file_name = file_path.name
                file_size = file_path.stat().st_size / (1024 * 1024)  # MB
                
                category_info = file_categories.get(file_name, {
                    "category": "Other",
                    "description": "Additional file",
                    "required": False
                })
                
                files_info[file_name] = {
                    "size_mb": round(file_size, 2),
                    "category": category_info["category"],
                    "description": category_info["description"],
                    "required": category_info["required"],
                    "path": str(file_path)
                }
        
        return files_info
    
    def clean_cache(self, model_id: Optional[str] = None):
        """Clean cached models"""
        if model_id:
            model_path = self.cache_dir / model_id
            if model_path.exists():
                shutil.rmtree(model_path)
                logger.info(f"Cleaned cache for {model_id}")
            else:
                logger.info(f"Model {model_id} not found in cache")
        else:
            if self.cache_dir.exists():
                shutil.rmtree(self.cache_dir)
                self.cache_dir.mkdir()
                logger.info("Cleaned entire model cache")

def main():
    parser = argparse.ArgumentParser(description="Qwen VL Model Download and Management")
    parser.add_argument("--cache-dir", default="./models_cache", help="Directory to cache models")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List available models")
    
    # Download command
    download_parser = subparsers.add_parser("download", help="Download a model")
    download_parser.add_argument("model_id", help="Model ID to download")
    download_parser.add_argument("--force", action="store_true", help="Force re-download")
    
    # Verify command
    verify_parser = subparsers.add_parser("verify", help="Verify a cached model")
    verify_parser.add_argument("model_id", help="Model ID to verify")
    
    # Info command
    info_parser = subparsers.add_parser("info", help="Get detailed model information")
    info_parser.add_argument("model_id", help="Model ID to get info for")
    
    # Files command
    files_parser = subparsers.add_parser("files", help="List model files and their purposes")
    files_parser.add_argument("model_id", help="Model ID to list files for")
    
    # Clean command
    clean_parser = subparsers.add_parser("clean", help="Clean model cache")
    clean_parser.add_argument("model_id", nargs="?", help="Model ID to clean (or all if not specified)")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Initialize model manager
    manager = ModelManager(args.cache_dir)
    
    try:
        if args.command == "list":
            manager.list_available_models()
            
        elif args.command == "download":
            path = manager.download_model(args.model_id, force_download=args.force)
            print(f"\n✅ Model downloaded successfully to: {path}")
            
        elif args.command == "verify":
            is_valid = manager.verify_model(args.model_id)
            if is_valid:
                print(f"\n✅ Model {args.model_id} is valid and complete")
            else:
                print(f"\n❌ Model {args.model_id} has issues")
                sys.exit(1)
                
        elif args.command == "info":
            info = manager.get_model_info(args.model_id)
            print(f"\n📋 MODEL INFORMATION: {args.model_id}")
            print("="*50)
            for key, value in info.items():
                print(f"{key.replace('_', ' ').title()}: {value}")
                
        elif args.command == "files":
            files_info = manager.get_model_files_info(args.model_id)
            print(f"\n📁 MODEL FILES: {args.model_id}")
            print("="*80)
            
            # Group by category
            categories = {}
            for file_name, info in files_info.items():
                category = info["category"]
                if category not in categories:
                    categories[category] = []
                categories[category].append((file_name, info))
            
            for category, files in categories.items():
                print(f"\n{category}:")
                for file_name, info in files:
                    required = "✅ Required" if info["required"] else "🔧 Optional"
                    print(f"  {file_name} ({info['size_mb']} MB) - {required}")
                    print(f"    {info['description']}")
                    
        elif args.command == "clean":
            manager.clean_cache(args.model_id)
            
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
