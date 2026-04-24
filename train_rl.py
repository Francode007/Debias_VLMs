import argparse
import os
import glob
import logging
import torch
import json
from accelerate import Accelerator
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import LoraConfig, get_peft_model, PeftModel
from torch.utils.data import DataLoader
from tqdm import tqdm

from modules.model_loader import ModelLoader
from modules.device_manager import DeviceManager
from modules.fast_rl import FastRLNode
from modules.custom_vlm_ppo_trainer import PPOVLMController
from modules.rl_dataset_builder import RLDatasetBuilder
from modules.rl_data_collator import RLDataCollatorWithPadding
from modules.config import ScriptArguments
from local_model_config import get_local_model_path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Run Fast-RL and CAA phase using Custom VLM PPO Training")
    parser.add_argument("--policy_model_name", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct", help="The base VLM policy model path")
    parser.add_argument("--extractor_model_name", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct", help="Default 3B architecture symmetry")
    parser.add_argument("--reward_heads_dir", type=str, default="./generated_heads/sb_bench-PCA-component", help="Directory containing Phase 1 .pth component files")
    parser.add_argument("--fast_rl_strategy", type=str, default="exponentiated", choices=["exponentiated", "projected", "adam"], help="Mirror Descent update strategy")
    parser.add_argument("--eta", type=float, default=0.01, help="Learning rate for Fast-RL node")
    parser.add_argument("--kl_beta", type=float, default=0.1, help="KL Divergence penalty coefficient")
    parser.add_argument("--num_heads", type=int, default=100, help="Number of PCA components to use")
    
    # Training Loop Args
    parser.add_argument("--data_path", type=str, default="./sb_bench_data/data", help="Path to parquet dataset")
    parser.add_argument("--output_dir", type=str, default="./output_ppo_debiased", help="Directory to save PEFT models")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4, help="Batch size (recommended 4 for A100)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation scale")
    parser.add_argument("--use_smallset", action="store_true", help="Use a tiny subset for testing")
    parser.add_argument("--max_length", type=int, default=1024, help="Maximum sequence length constraints")
    parser.add_argument("--epochs", type=int, default=1, help="Total execution epochs")
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
        # Prevent torch weights_only warnings correctly handling linear layer dumps
        state = torch.load(p, map_location="cpu") 
        w = state.get("weight", state)
        if isinstance(w, dict):
            w = w.get("weight")
        weights.append(w) # shape (1, hidden_dim)
        
    combined = torch.cat(weights, dim=0).to(device, dtype=torch.bfloat16) 
    return combined

def load_debiased_model(base_model_path: str, peft_adapter_path: str, device: str = "cuda"):
    """
    Utility explicitly requested by User.
    Loads the frozen base model and applies the saved debiased PEFT adapter back on top.
    """
    logger.info(f"Loading Base: {base_model_path}")
    base_model = AutoModel.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map={"": device},
        trust_remote_code=True
    )
    logger.info(f"Merging LoRA Adapter: {peft_adapter_path}")
    model = PeftModel.from_pretrained(base_model, peft_adapter_path)
    return model

def main():
    args = parse_args()
    
    accelerator = Accelerator(
        mixed_precision="bf16", 
        gradient_accumulation_steps=args.gradient_accumulation_steps
    )
    
    # Setup model configuration for ModelLoader (Extractor)
    extractor_cfg = ScriptArguments(
        model=args.extractor_model_name,
        device="cuda" if torch.cuda.is_available() else "cpu",
        max_length=args.max_length,
        use_smallset=args.use_smallset
    )
    device_manager = DeviceManager()
    extractor_loader = ModelLoader(extractor_cfg, device_manager)
    
    logger.info("Loading Extractor Model (Frozen Base)...")
    extractor, _ = extractor_loader.load_model_and_processor()
    extractor.eval()
    for param in extractor.parameters():
        param.requires_grad = False
        
    logger.info("Loading Policy Model and injecting LoRA for Active PPO Policy...")
    # Setup model configuration for ModelLoader (Policy)
    script_cfg = ScriptArguments(
        model=args.policy_model_name,
        device="cuda" if torch.cuda.is_available() else "cpu",
        max_length=args.max_length,
        use_smallset=args.use_smallset
    )
    loader = ModelLoader(script_cfg, device_manager)
    
    # Instantiate custom processor
    processor_path = get_local_model_path(args.policy_model_name)
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(processor_path)
    
    policy_base, _ = loader.load_model_and_processor()
    
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    active_policy = get_peft_model(policy_base, lora_config)
    
    logger.info("Loading Phase 1 PCA Score Heads...")
    try:
        reward_heads_weight = load_pca_components(args.reward_heads_dir, args.num_heads, accelerator.device)
        # Update num_heads to actual number of loaded components
        args.num_heads = reward_heads_weight.shape[0]
        logger.info(f"Loaded reward heads matrix of shape: {reward_heads_weight.shape}")
    except Exception as e:
        logger.error(f"Could not load reward heads: {e}")
        return
        
    fast_rl_node = FastRLNode(num_heads=args.num_heads, strategy=args.fast_rl_strategy, eta=args.eta, device=accelerator.device)
    
    ppo_controller = PPOVLMController(
        active_policy=active_policy,
        extractor_model=extractor,
        reward_heads_weight=reward_heads_weight,
        accelerator=accelerator,
        fast_rl_node=fast_rl_node,
        kl_beta=args.kl_beta
    )
    
    # Optimizers
    optimizer_policy = torch.optim.AdamW(filter(lambda p: p.requires_grad, active_policy.parameters()), lr=1e-5)
    optimizer_value = torch.optim.AdamW(ppo_controller.value_head.parameters(), lr=1e-5)
    
    # Datalist Builder (RL Specific mapped for prompt sequences to trigger the dual generator pipeline)
    logger.info("Initializing Dataset Build...")
    script_cfg = ScriptArguments(
        max_length=args.max_length, 
        use_smallset=args.use_smallset
    )
    dataset_builder = RLDatasetBuilder(script_cfg)
    train_dataset = dataset_builder.build_dataset(data_path=args.data_path, processor=processor)
    collator = RLDataCollatorWithPadding(processor=processor)
    
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=args.per_device_train_batch_size, 
        collate_fn=collator, 
        shuffle=True
    )
    
    # Accelerator Preparations
    active_policy, ppo_controller.value_head, optimizer_policy, optimizer_value, train_dataloader = accelerator.prepare(
        active_policy, ppo_controller.value_head, optimizer_policy, optimizer_value, train_dataloader
    )
    
    # Start Epoch Training Loop
    logger.info("Starting PPO Training Execution...")
    
    global_step = 0
    os.makedirs(args.output_dir, exist_ok=True)
    
    for epoch in range(args.epochs):
        logger.info(f"--- Epoch {epoch+1}/{args.epochs} ---")
        
        # We process batches with gradient accumulation directly through accelerate
        with tqdm(train_dataloader, desc=f"Epoch {epoch+1}") as pbar:
            for batch in pbar:
                with accelerator.accumulate(active_policy):
                    # Execution of the explicit Dual Generation Pipeline
                    metrics = ppo_controller.step(batch, optimizer_policy, optimizer_value)
                    
                    pbar.set_postfix({
                        "loss": f"{metrics['loss']:.4f}",
                        "reward": f"{metrics['reward']:.4f}",
                        "w_hat": f"{metrics['w_hat_mean']:.4f}"
                    })
                
                global_step += 1
                
                # Dynamic Checkpointing (e.g., every 50 global steps)
                if global_step % 50 == 0:
                    checkpoint_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    logger.info(f"Saving explicitly isolated PEFT adapters to {checkpoint_path}")
                    # ONLY save LoRA parameters!
                    if accelerator.is_main_process:
                        unwrapped_model = accelerator.unwrap_model(active_policy)
                        unwrapped_model.save_pretrained(checkpoint_path)
                        
                        # Save Fast-RL States
                        state_dict = {"alpha": fast_rl_node.alpha.detach().cpu().tolist()}
                        with open(os.path.join(checkpoint_path, "fast_rl_state.json"), "w") as f:
                            json.dump(state_dict, f)
                            
                        # Save Value Head (Crucial parameter!)
                        torch.save(ppo_controller.value_head.state_dict(), os.path.join(checkpoint_path, "value_head.pth"))
                            
    # Final Save
    if accelerator.is_main_process:
        final_dir = os.path.join(args.output_dir, "final_debiased_model")
        unwrapped_model = accelerator.unwrap_model(active_policy)
        unwrapped_model.save_pretrained(final_dir)
        state_dict = {"alpha": fast_rl_node.alpha.detach().cpu().tolist()}
        with open(os.path.join(final_dir, "fast_rl_state.json"), "w") as f:
            json.dump(state_dict, f)
        torch.save(ppo_controller.value_head.state_dict(), os.path.join(final_dir, "value_head.pth"))
            
        logger.info(f"PPO Debiasing complete. Model saved to {final_dir}")

if __name__ == "__main__":
    main()
