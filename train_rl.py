import argparse
import os
import glob
import logging
import torch
from accelerate import Accelerator
from transformers import AutoProcessor, Qwen2_5VLForConditionalGeneration
from peft import LoraConfig, get_peft_model, TaskType

from modules.fast_rl import FastRLNode
from modules.custom_vlm_ppo_trainer import PPOVLMController

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Run Fast-RL and CAA phase using Custom VLM PPO Training")
    parser.add_argument("--policy_model_name", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct", help="The base VLM policy model path")
    parser.add_argument("--extractor_model_name", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct", help="The frozen feature extractor model path (Default: 3B for symmetry)")
    parser.add_argument("--reward_heads_dir", type=str, default="./generated_heads/sb_bench-PCA-component", help="Directory containing Phase 1 .pth component files")
    parser.add_argument("--fast_rl_strategy", type=str, default="exponentiated", choices=["exponentiated", "projected", "adam"], help="Mirror Descent update strategy")
    parser.add_argument("--eta", type=float, default=0.01, help="Learning rate for Fast-RL node")
    parser.add_argument("--kl_beta", type=float, default=0.1, help="KL Divergence penalty coefficient")
    parser.add_argument("--num_heads", type=int, default=100, help="Number of PCA components to use")
    return parser.parse_args()

def load_pca_components(heads_dir: str, num_heads: int, device: torch.device):
    """Loads the 100 Phase 1 orthogonal .pth files."""
    pth_files = glob.glob(os.path.join(heads_dir, "*.pth"))
    if not pth_files:
        raise FileNotFoundError(f"No .pth files found in {heads_dir}.")
    
    import re
    def extract_index(f):
        m = re.search(r"component(\d+)\.pth$", f)
        return int(m.group(1)) if m else 0
        
    pth_files = sorted(pth_files, key=extract_index)
    pth_files = pth_files[:num_heads]
    
    weights = []
    for p in pth_files:
        state = torch.load(p, map_location="cpu", weights_only=True)
        w = state.get("weight", state)
        if isinstance(w, dict):
            w = w.get("weight")
        weights.append(w) # shape (1, hidden_dim)
        
    combined = torch.cat(weights, dim=0).to(device) # shape (num_heads, hidden_dim)
    return combined

def main():
    args = parse_args()
    accelerator = Accelerator(mixed_precision="bf16")
    
    logger.info("Loading Extractor Model (Frozen Base)...")
    extractor = Qwen2_5VLForConditionalGeneration.from_pretrained(
        args.extractor_model_name,
        torch_dtype=torch.bfloat16,
        device_map={"": accelerator.device}
    )
    extractor.eval()
    for param in extractor.parameters():
        param.requires_grad = False
        
    logger.info("Loading Policy Model and injecting LoRA for Active PPO Policy...")
    policy_base = Qwen2_5VLForConditionalGeneration.from_pretrained(
        args.policy_model_name,
        torch_dtype=torch.bfloat16,
        device_map={"": accelerator.device}
    )
    
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    """
    Another possible configuration:

    {
    "r": 64, 
    "lora_alpha": 128, # Maintaining the alpha = 2*r ratio for strong updates
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "bias": "none",
    "task_type": "CAUSAL_LM"
    }
    """
    active_policy = get_peft_model(policy_base, lora_config)
    
    logger.info("Loading Phase 1 PCA Score Heads...")
    try:
        reward_heads_weight = load_pca_components(args.reward_heads_dir, args.num_heads, accelerator.device)
        logger.info(f"Loaded reward heads matrix of shape: {reward_heads_weight.shape}")
    except Exception as e:
        logger.error(f"Could not load reward heads: {e}")
        return
        
    logger.info("Initializing Custom PPO Controller...")
    fast_rl_node = FastRLNode(num_heads=args.num_heads, strategy=args.fast_rl_strategy, eta=args.eta, device=accelerator.device)
    
    ppo_controller = PPOVLMController(
        active_policy=active_policy,
        extractor_model=extractor,
        reward_heads_weight=reward_heads_weight,
        accelerator=accelerator,
        fast_rl_node=fast_rl_node,
        kl_beta=args.kl_beta
    )
    
    # Example setup: In practice, we'd need optimizers and datasets here.
    optimizer_policy = torch.optim.AdamW(filter(lambda p: p.requires_grad, active_policy.parameters()), lr=1e-5)
    optimizer_value = torch.optim.AdamW(ppo_controller.value_head.parameters(), lr=1e-5)
    
    active_policy, ppo_controller.value_head, optimizer_policy, optimizer_value = accelerator.prepare(
        active_policy, ppo_controller.value_head, optimizer_policy, optimizer_value
    )
    
    logger.info("Initialization Complete. The generic setup is ready for dataloader ingestion.")
    
if __name__ == "__main__":
    main()
