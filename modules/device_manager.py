"""
Device Management Module

This module handles device selection, compatibility settings, and backend configuration
for optimal performance across different hardware (CUDA, MPS, CPU).
"""

import os
import torch
import logging

logger = logging.getLogger(__name__)


class DeviceManager:
    """
    Manages device selection and compatibility settings for different hardware configurations.
    
    This class provides utilities to automatically detect the best available device,
    configure optimal data types and attention implementations, and set up PyTorch
    backends for maximum performance.
    """
    
    @staticmethod
    def get_optimal_device():
        """
        Detect and return the best available device for computation.
        
        Input:
            None
        
        Output:
            str: Device string - "cuda", "mps", or "cpu"
        
        Process:
            1. Check CUDA availability first (highest priority)
            2. Check MPS (Apple Silicon) availability
            3. Fallback to CPU
        
        Purpose:
            Automatically select the most performant device available on the system
            to avoid manual device configuration errors.
        """
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    @staticmethod
    def get_optimal_dtype(device: str):
        """
        Get optimal data type based on device capabilities.
        
        Input:
            device (str): Target device ("cuda", "mps", "cpu")
        
        Output:
            torch.dtype: Optimal precision type for the device
        
        Process:
            1. CUDA: Use bfloat16 for best performance with modern GPUs
            2. MPS: Use float16 for Apple Silicon optimization
            3. CPU: Use float32 for stability
        
        Purpose:
            Optimize memory usage and computation speed while maintaining
            numerical stability for each device type.
        """
        if device == "cuda":
            # CUDA supports bfloat16
            return torch.bfloat16
        elif device == "mps":
            # MPS has better support for float16
            return torch.float32
        else:
            # CPU fallback to float32
            return torch.float32
    
    @staticmethod
    def get_attention_implementation(device: str):
        """
        Get optimal attention implementation based on device capabilities.
        
        Input:
            device (str): Target device ("cuda", "mps", "cpu")
        
        Output:
            str: Attention implementation ("flash_attention_2" or "eager")
        
        Process:
            1. For CUDA: Try to use Flash Attention 2 if available
            2. For MPS/CPU: Use eager attention (Flash Attention not supported)
            3. Fallback to eager if Flash Attention import fails
        
        Purpose:
            Optimize attention computation speed for transformer models.
            Flash Attention can significantly reduce memory usage and increase speed.
        """
        if device == "cuda":
            try:
                import flash_attn
                return "flash_attention_2"
            except ImportError:
                logger.warning("Flash attention not available, falling back to eager attention")
                return "eager"
        else:
            # MPS and CPU don't support flash attention
            return "eager"
    
    @staticmethod
    def configure_torch_backends(device: str):
        """
        Configure PyTorch backends for optimal performance on specific devices.
        
        Input:
            device (str): Target device ("cuda", "mps", "cpu")
        
        Output:
            None (modifies global PyTorch settings)
        
        Process:
            1. CUDA: Enable TensorFloat-32 (TF32) for faster computation
            2. MPS: Set fallback environment variable for compatibility
            3. CPU: No special configuration needed
        
        Purpose:
            Enable device-specific optimizations to maximize computational performance
            while maintaining compatibility and stability.
        """
        if device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        elif device == "mps":
            # MPS-specific optimizations
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
