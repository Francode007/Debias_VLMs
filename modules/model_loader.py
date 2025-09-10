"""
Model Loading Module

This module handles the loading and initialization of vision-language models
with device compatibility and fallback mechanisms.
"""

import logging
import torch
import torch.nn as nn
from transformers import AutoProcessor
from .device_manager import DeviceManager
from .config import ScriptArguments

logger = logging.getLogger(__name__)


class ModelLoader:
    """
    Handles model loading with device compatibility and fallback mechanisms.
    
    This class manages the complex process of loading vision-language models
    with appropriate device settings, data types, and attention implementations.
    It provides robust fallback mechanisms when primary models fail to load.
    """
    
    def __init__(self, script_args: ScriptArguments, device_manager: DeviceManager):
        """
        Initialize ModelLoader with configuration and device manager.
        
        Input:
            script_args (ScriptArguments): Configuration parameters
            device_manager (DeviceManager): Device management utilities
        
        Output:
            None (initializes instance)
        
        Process:
            1. Store configuration and device manager references
            2. Set up device, dtype, and attention implementation
            3. Try to import local model configuration utilities
            4. Set up fallback functions if local config unavailable
        
        Purpose:
            Prepare model loading environment with optimal settings
            and provide fallback mechanisms for robustness.
        """
        self.script_args = script_args
        self.device_manager = device_manager
        self.device = self._setup_device()
        self.dtype = self._setup_dtype()
        self.attention_impl = self._setup_attention()
        
        # Try to import local model configuration
        self.local_model_manager = None
        try:
            from local_model_config import get_local_model_path, get_recommended_local_model, list_local_models
            self.get_local_model_path = get_local_model_path
            self.get_recommended_local_model = get_recommended_local_model
            self.list_local_models = list_local_models
            logger.info("✅ Local model configuration loaded")
        except ImportError:
            logger.info("⚠️  Local model configuration not found. Using HuggingFace models.")
            self.get_local_model_path = lambda x: x
            self.get_recommended_local_model = lambda: "Qwen/Qwen2-VL-7B-Instruct"
            self.list_local_models = lambda: print("No local model configuration available")
        
    def _setup_device(self):
        """
        Setup device based on configuration arguments.
        
        Input:
            None (uses self.script_args.device)
        
        Output:
            str: Selected device ("cuda", "mps", or "cpu")
        
        Process:
            1. If device is "auto", use DeviceManager to detect optimal device
            2. Otherwise, use the specified device from configuration
            3. Log the selected device for debugging
        
        Purpose:
            Determine the computational device to use for model operations,
            allowing both automatic detection and manual override.
        """
        if self.script_args.device == "auto":
            device = self.device_manager.get_optimal_device()
        else:
            device = self.script_args.device
            
        logger.info(f"Using device: {device}")
        return device
    
    def _setup_dtype(self):
        """
        Setup data type based on device and configuration.
        
        Input:
            None (uses self.script_args.force_fp32 and self.device)
        
        Output:
            torch.dtype: Selected data type for model operations
        
        Process:
            1. If force_fp32 is True, use float32 regardless of device
            2. Otherwise, use DeviceManager to get optimal dtype for device
            3. Log the selected dtype for debugging
        
        Purpose:
            Determine the numerical precision to use for model operations,
            balancing performance and numerical stability.
        """
        if self.script_args.force_fp32:
            dtype = torch.float32
        else:
            dtype = self.device_manager.get_optimal_dtype(self.device)
        
        logger.info(f"Using dtype: {dtype}")
        return dtype
    
    def _setup_attention(self):
        """
        Setup attention implementation based on device capabilities.
        
        Input:
            None (uses self.device)
        
        Output:
            str: Attention implementation ("flash_attention_2" or "eager")
        
        Process:
            1. Use DeviceManager to get optimal attention implementation
            2. Log the selected implementation for debugging
        
        Purpose:
            Configure the attention mechanism for optimal performance
            on the target device.
        """
        attention_impl = self.device_manager.get_attention_implementation(self.device)
        logger.info(f"Using attention implementation: {attention_impl}")
        return attention_impl
    
    def load_model_and_processor(self):
        """
        Load model and processor with optimal settings and fallback mechanisms.
        
        Input:
            None (uses instance configuration)
        
        Output:
            tuple: (model, processor) - Loaded model and processor objects
        
        Process:
            1. Convert model names to local paths if available
            2. Create list of models to try (primary + fallbacks)
            3. For each model, attempt loading with optimal settings
            4. If optimal settings fail, try fallback settings
            5. Add score head for reward model functionality
            6. Return successfully loaded model and processor
        
        Purpose:
            Robustly load vision-language models with device optimization
            and comprehensive fallback mechanisms to handle various failure modes.
        
        Raises:
            RuntimeError: If all model loading attempts fail
        """
        
        # Convert model names to local paths if available
        base_model = self.get_local_model_path(self.script_args.model)
        
        # Try with the primary model first, then fallback
        models_to_try = [base_model]
        if self.script_args.fallback_model and self.script_args.fallback_model != self.script_args.model:
            fallback_model = self.get_local_model_path(self.script_args.fallback_model)
            models_to_try.append(fallback_model)
        
        # Add additional local fallbacks if available
        try:
            recommended_local = self.get_recommended_local_model()
            if recommended_local and recommended_local not in models_to_try:
                models_to_try.append(recommended_local)
        except:
            pass
        
        for model_name in models_to_try:
            logger.info(f"Attempting to load model: {model_name}")
            
            try:
                # Try to determine the correct model class
                if "2.5" in model_name:
                    # For Qwen2.5-VL models
                    try:
                        from transformers import Qwen2_5VLForConditionalGeneration as ModelClass
                        logger.info("Using Qwen2.5VL model class")
                    except ImportError:
                        logger.warning("Qwen2.5VL not available, trying Qwen2VL...")
                        from transformers import Qwen2VLForConditionalGeneration as ModelClass
                        logger.info("Using Qwen2VL model class")
                else:
                    # For Qwen2VL models
                    from transformers import Qwen2VLForConditionalGeneration as ModelClass
                    logger.info("Using Qwen2VL model class")
                
                # Configure backends
                self.device_manager.configure_torch_backends(self.device)
                
                # Load processor from local path
                processor_path = self.get_local_model_path(model_name)
                processor = AutoProcessor.from_pretrained(processor_path)
                
                # Prepare model loading arguments
                model_kwargs = {
                    "torch_dtype": self.dtype,
                    "trust_remote_code": True,
                }
                
                # Add attention implementation if supported
                if self.attention_impl != "eager":
                    model_kwargs["attn_implementation"] = self.attention_impl
                
                # Device mapping
                if self.device == "cuda":
                    # Use accelerate for CUDA
                    from accelerate import Accelerator
                    accelerator = Accelerator()
                    device_map = accelerator.local_process_index
                    model_kwargs["device_map"] = device_map
                else:
                    # For MPS and CPU, load to specific device
                    model_kwargs["device_map"] = None
                
                try:
                    logger.info(f"Loading model {model_name} with {ModelClass.__name__}")
                    # Use local path for model loading
                    model_path = self.get_local_model_path(model_name)
                    model = ModelClass.from_pretrained(model_path, **model_kwargs)
                    
                    # Move to device if not using device_map
                    if model_kwargs["device_map"] is None:
                        model = model.to(self.device)
                    
                    # Add score head - check if it already exists
                    if not hasattr(model, 'score'):
                        # Get the hidden size from the model config
                        hidden_size = getattr(model.config, 'hidden_size', 1280)
                        model.score = nn.Linear(hidden_size, 1, bias=False)
                        logger.info(f"Added score head with hidden_size: {hidden_size}")
                    
                    logger.info(f"Successfully loaded model: {model_name}")
                    return model, processor
                    
                except Exception as e:
                    logger.warning(f"Failed to load {model_name} with optimal settings: {e}")
                    
                    # Try fallback settings for this model
                    try:
                        logger.info(f"Trying fallback settings for {model_name}...")
                        model_path = self.get_local_model_path(model_name)
                        model = ModelClass.from_pretrained(
                            model_path,
                            torch_dtype=torch.float32,
                            device_map=None,
                            trust_remote_code=True
                        )
                        model = model.to("cpu")
                        self.device = "cpu"
                        self.dtype = torch.float32
                        
                        # Add score head
                        if not hasattr(model, 'score'):
                            hidden_size = getattr(model.config, 'hidden_size', 1280)
                            model.score = nn.Linear(hidden_size, 1, bias=False)
                            logger.info(f"Added score head with hidden_size: {hidden_size}")
                        
                        logger.info(f"Successfully loaded {model_name} with fallback settings")
                        return model, processor
                        
                    except Exception as e2:
                        logger.warning(f"Fallback settings also failed for {model_name}: {e2}")
                        continue  # Try next model
                        
            except ImportError as e:
                logger.warning(f"Model class import failed for {model_name}: {e}")
                continue  # Try next model
        
        # If all models failed, raise an error
        raise RuntimeError(f"Failed to load any of the models: {models_to_try}. "
                          f"Please check your transformers version and model availability.")
