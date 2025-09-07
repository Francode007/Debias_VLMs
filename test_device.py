#!/usr/bin/env python3
"""
Device compatibility test script
Run this to check what devices and features are available on your system
"""

import sys
import warnings
warnings.filterwarnings("ignore")

def test_torch():
    """Test PyTorch installation and device availability"""
    print("Testing PyTorch...")
    try:
        import torch
        print(f"✓ PyTorch version: {torch.__version__}")
        
        # Test CUDA
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.device_count()} device(s)")
            print(f"  Current device: {torch.cuda.current_device()}")
            print(f"  Device name: {torch.cuda.get_device_name()}")
        else:
            print("✗ CUDA not available")
        
        # Test MPS (Apple Silicon)
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print("✓ MPS (Apple Silicon) available")
            # Test MPS functionality
            try:
                x = torch.randn(2, 3, device='mps')
                y = x @ x.T
                print("✓ MPS basic operations work")
            except Exception as e:
                print(f"✗ MPS test failed: {e}")
        else:
            print("✗ MPS not available")
        
        # Test CPU
        try:
            x = torch.randn(2, 3)
            y = x @ x.T
            print("✓ CPU operations work")
        except Exception as e:
            print(f"✗ CPU test failed: {e}")
        
        return True
    except ImportError:
        print("✗ PyTorch not installed")
        return False

def test_transformers():
    """Test transformers installation"""
    print("\nTesting Transformers...")
    try:
        import transformers
        print(f"✓ Transformers version: {transformers.__version__}")
        
        # Test AutoProcessor
        try:
            from transformers import AutoProcessor
            print("✓ AutoProcessor available")
        except ImportError:
            print("✗ AutoProcessor not available")
        
        # Test Qwen2VL (might not be available in older versions)
        try:
            from transformers import Qwen2VLForConditionalGeneration
            print("✓ Qwen2VL model available")
        except ImportError:
            print("✗ Qwen2VL model not available (may need newer transformers version)")
        
        return True
    except ImportError:
        print("✗ Transformers not installed")
        return False

def test_flash_attention():
    """Test flash attention availability"""
    print("\nTesting Flash Attention...")
    try:
        import flash_attn
        print(f"✓ Flash attention available: {flash_attn.__version__}")
        return True
    except ImportError:
        print("✗ Flash attention not available (this is fine for MPS/CPU)")
        return False

def test_other_dependencies():
    """Test other required dependencies"""
    print("\nTesting other dependencies...")
    
    deps = [
        ("accelerate", "accelerate"),
        ("datasets", "datasets"),
        ("evaluate", "evaluate"),
        ("peft", "peft"),
        ("trl", "trl"),
        ("pandas", "pandas"),
        ("numpy", "numpy"),
        ("PIL", "Pillow"),
        ("tqdm", "tqdm"),
    ]
    
    all_good = True
    for module, package in deps:
        try:
            __import__(module)
            print(f"✓ {package} available")
        except ImportError:
            print(f"✗ {package} not available")
            all_good = False
    
    return all_good

def test_data_types():
    """Test different data types on available devices"""
    print("\nTesting data types...")
    try:
        import torch
        
        # Test different dtypes
        dtypes_to_test = [torch.float32, torch.float16]
        if hasattr(torch, 'bfloat16'):
            dtypes_to_test.append(torch.bfloat16)
        
        devices_to_test = ["cpu"]
        if torch.cuda.is_available():
            devices_to_test.append("cuda")
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            devices_to_test.append("mps")
        
        for device in devices_to_test:
            print(f"\n  Device: {device}")
            for dtype in dtypes_to_test:
                try:
                    x = torch.randn(2, 3, dtype=dtype, device=device)
                    y = x @ x.T
                    print(f"    ✓ {dtype} works")
                except Exception as e:
                    print(f"    ✗ {dtype} failed: {e}")
        
    except ImportError:
        print("✗ PyTorch not available for dtype testing")

def get_recommended_config():
    """Get recommended configuration based on available hardware"""
    print("\n" + "="*50)
    print("RECOMMENDED CONFIGURATION")
    print("="*50)
    
    try:
        import torch
        
        if torch.cuda.is_available():
            print("🚀 CUDA detected - Optimal performance mode:")
            print("  - Device: cuda")
            print("  - Precision: bfloat16 (if supported) or float16")
            print("  - Flash Attention: Recommended (install flash-attn)")
            print("  - Batch size: 2-4 (depending on GPU memory)")
            print("  - DataLoader batch size: 4-8")
            print("\nCommand to run:")
            print("  python run_flexible.py --device cuda --batch-size 2 --dataloader-batch-size 4")
            
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print("🍎 Apple Silicon (MPS) detected - Good performance mode:")
            print("  - Device: mps")
            print("  - Precision: float16")
            print("  - Flash Attention: Not supported (will use eager)")
            print("  - Batch size: 1-2 (depending on unified memory)")
            print("  - DataLoader batch size: 1-2")
            print("\nCommand to run:")
            print("  python run_flexible.py --device mps --max-length 512 --batch-size 1")
            
        else:
            print("💻 CPU mode - Compatibility mode:")
            print("  - Device: cpu")
            print("  - Precision: float32")
            print("  - Flash Attention: Not supported")
            print("  - Batch size: 1 (keep minimal for memory)")
            print("  - DataLoader batch size: 1")
            print("  - Reduce max_length for memory")
            print("\nCommand to run:")
            print("  python run_flexible.py --device cpu --reduce-memory --small-dataset")
            
    except ImportError:
        print("Cannot determine optimal configuration - PyTorch not available")

def main():
    print("Device Compatibility Test")
    print("=" * 50)
    
    # Run all tests
    torch_ok = test_torch()
    transformers_ok = test_transformers()
    flash_ok = test_flash_attention()
    deps_ok = test_other_dependencies()
    
    if torch_ok:
        test_data_types()
    
    # Summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    
    if torch_ok and transformers_ok and deps_ok:
        print("✅ All essential dependencies are available!")
        get_recommended_config()
    else:
        print("❌ Some dependencies are missing:")
        if not torch_ok:
            print("  - Install PyTorch: pip install torch torchvision")
        if not transformers_ok:
            print("  - Install Transformers: pip install transformers>=4.35.0")
        if not deps_ok:
            print("  - Install other dependencies: pip install -r requirements.txt")
        
        print("\nAfter installing missing dependencies, run this test again.")

if __name__ == "__main__":
    main()
