import argparse
import os
import glob
import logging
import warnings
import torch
import json

# Suppress kernel version warning from accelerate (Modal host kernel is 4.4.0)
warnings.filterwarnings("ignore", message=".*Detected kernel version.*")

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
import time
import multiprocessing

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
    parser.add_argument("--dataset_name", type=str, default="sb_bench", help="Dataset name for formatting")
    parser.add_argument("--model_family", type=str, default="qwen", help="Model family for wrapper")
    parser.add_argument("--data_path", type=str, default="./sb_bench_data/data", help="Path to parquet dataset")
    parser.add_argument("--output_dir", type=str, default="./output_ppo_debiased", help="Directory to save PEFT models")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8, help="Batch size (recommended 8 for A100 after optimization)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation scale")
    parser.add_argument("--use_smallset", action="store_true", help="Use a tiny subset for testing")
    parser.add_argument("--max_length", type=int, default=1024, help="Maximum sequence length constraints")
    parser.add_argument("--epochs", type=int, default=1, help="Total execution epochs")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to checkpoint directory to resume training from")
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
    t_start_setup = time.time()
    args = parse_args()
    
    accelerator = Accelerator(
        mixed_precision="bf16", 
        gradient_accumulation_steps=args.gradient_accumulation_steps
    )
    
    device_manager = DeviceManager()
    
    logger.info("Loading Policy Model and injecting LoRA for Active PPO Policy...")
    script_cfg = ScriptArguments(
        model=args.policy_model_name,
        device="cuda" if torch.cuda.is_available() else "cpu",
        max_length=args.max_length,
        use_smallset=args.use_smallset,
        dataset_name=args.dataset_name,
        model_family=args.model_family
    )
    loader = ModelLoader(script_cfg, device_manager)
    
    # Instantiate custom processor
    processor_path = get_local_model_path(args.policy_model_name)
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(processor_path, use_fast=True)
    processor.tokenizer.padding_side = "left"
    
    policy_base, _ = loader.load_model_and_processor()
    
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    active_policy = get_peft_model(policy_base, lora_config)
    
    # OOM Mitigation: Enable Gradient Checkpointing
    if hasattr(active_policy, "gradient_checkpointing_enable"):
        logger.info("Enabling gradient checkpointing for policy model...")
        active_policy.gradient_checkpointing_enable()
    
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
        use_smallset=args.use_smallset,
        dataset_name=args.dataset_name,
        model_family=args.model_family
    )
    dataset_builder = RLDatasetBuilder(script_cfg)
    train_dataset = dataset_builder.build_dataset(data_path=args.data_path, processor=processor)
    collator = RLDataCollatorWithPadding(processor=processor)
    
    # Optimization: Cap num_workers to allocated CPU count (Modal reports machine cores, not allocated)
    num_cpus = multiprocessing.cpu_count()
    num_workers = min(12, max(1, int(num_cpus * 0.8)))
    logger.info(f"Using num_workers={num_workers} for DataLoader (capped at 12, machine reports {num_cpus} cores)")

    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=args.per_device_train_batch_size, 
        collate_fn=collator, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Accelerator Preparations
    active_policy, ppo_controller.value_head, optimizer_policy, optimizer_value, train_dataloader = accelerator.prepare(
        active_policy, ppo_controller.value_head, optimizer_policy, optimizer_value, train_dataloader
    )
    
    # Start Epoch Training Loop
    t_end_setup = time.time()
    setup_duration = t_end_setup - t_start_setup
    logger.info(f"Setup completed in {setup_duration:.2f}s (Model loading + Dataset build)")
    
    logger.info("Starting PPO Training Execution...")
    t_start_training = time.time()
    batch_times = []
    
    global_step = 0
    start_epoch = 0
    resume_step = 0  # Step within epoch to resume from
    os.makedirs(args.output_dir, exist_ok=True)
    
    # --- Resume from checkpoint ---
    if args.resume_from_checkpoint and os.path.exists(args.resume_from_checkpoint):
        ckpt_dir = args.resume_from_checkpoint
        logger.info(f"Resuming from checkpoint: {ckpt_dir}")
        
        # Load LoRA adapter weights
        unwrapped = accelerator.unwrap_model(active_policy)
        from peft import set_peft_model_state_dict
        adapter_bin = os.path.join(ckpt_dir, "adapter_model.bin")
        adapter_safetensors = os.path.join(ckpt_dir, "adapter_model.safetensors")
        if os.path.exists(adapter_safetensors):
            from safetensors.torch import load_file
            adapter_state = load_file(adapter_safetensors)
        elif os.path.exists(adapter_bin):
            adapter_state = torch.load(adapter_bin, map_location="cpu")
        else:
            raise FileNotFoundError(f"No adapter weights found in {ckpt_dir}")
        set_peft_model_state_dict(unwrapped, adapter_state)
        del adapter_state
        logger.info("Loaded LoRA adapter weights")
        
        # Load value head
        vh_path = os.path.join(ckpt_dir, "value_head.pth")
        if os.path.exists(vh_path):
            ppo_controller.value_head.load_state_dict(torch.load(vh_path, map_location="cpu"))
            logger.info("Loaded value head")
        
        # Load Fast-RL state
        frl_path = os.path.join(ckpt_dir, "fast_rl_state.json")
        if os.path.exists(frl_path):
            with open(frl_path) as f:
                frl_state = json.load(f)
            fast_rl_node.alpha = torch.tensor(frl_state["alpha"], device=accelerator.device)
            logger.info("Loaded Fast-RL state")
        
        # Load optimizer states
        opt_policy_path = os.path.join(ckpt_dir, "optimizer_policy.pth")
        opt_value_path = os.path.join(ckpt_dir, "optimizer_value.pth")
        if os.path.exists(opt_policy_path):
            optimizer_policy.load_state_dict(torch.load(opt_policy_path, map_location="cpu"))
            logger.info("Loaded policy optimizer state")
        if os.path.exists(opt_value_path):
            optimizer_value.load_state_dict(torch.load(opt_value_path, map_location="cpu"))
            logger.info("Loaded value optimizer state")
        
        # Load training progress
        progress_path = os.path.join(ckpt_dir, "training_progress.json")
        if os.path.exists(progress_path):
            with open(progress_path) as f:
                progress = json.load(f)
            start_epoch = progress.get("epoch", 0)
            resume_step = progress.get("step_in_epoch", 0)
            global_step = progress.get("global_step", 0)
            logger.info(f"Resuming from epoch {start_epoch}, step {resume_step}, global_step {global_step}")
        else:
            # Fallback: infer global_step from old-style checkpoint directory name (e.g. checkpoint-200)
            import re
            ckpt_match = re.search(r"checkpoint-(\d+)$", ckpt_dir.rstrip("/"))
            if ckpt_match:
                global_step = int(ckpt_match.group(1))
                # Estimate epoch and step_in_epoch (will be recalculated after dataloader is known)
                logger.info(f"No training_progress.json found. Inferred global_step={global_step} from checkpoint name.")
            else:
                logger.warning("No training_progress.json found and could not infer step from checkpoint name. Starting from step 0.")
    
    # --- Helper to save a full checkpoint ---
    def save_checkpoint(tag, epoch, step_in_epoch):
        if not accelerator.is_main_process:
            return
        checkpoint_path = os.path.join(args.output_dir, f"checkpoint-{tag}")
        logger.info(f"Saving checkpoint to {checkpoint_path}")
        os.makedirs(checkpoint_path, exist_ok=True)
        
        unwrapped_model = accelerator.unwrap_model(active_policy)
        unwrapped_model.save_pretrained(checkpoint_path)
        
        # Fast-RL state
        with open(os.path.join(checkpoint_path, "fast_rl_state.json"), "w") as f:
            json.dump({"alpha": fast_rl_node.alpha.detach().cpu().tolist()}, f)
        
        # Value head
        torch.save(ppo_controller.value_head.state_dict(), os.path.join(checkpoint_path, "value_head.pth"))
        
        # Optimizer states (essential for resuming without loss spikes)
        torch.save(optimizer_policy.state_dict(), os.path.join(checkpoint_path, "optimizer_policy.pth"))
        torch.save(optimizer_value.state_dict(), os.path.join(checkpoint_path, "optimizer_value.pth"))
        
        # Training progress
        with open(os.path.join(checkpoint_path, "training_progress.json"), "w") as f:
            json.dump({"epoch": epoch, "step_in_epoch": step_in_epoch, "global_step": global_step}, f)
        
        logger.info(f"Checkpoint saved: epoch={epoch}, step={step_in_epoch}, global_step={global_step}")
    
    total_batches_per_epoch = len(train_dataloader)
    
    # Resolve epoch/step for old-style checkpoints (no training_progress.json)
    if args.resume_from_checkpoint and resume_step == 0 and start_epoch == 0 and global_step > 0:
        start_epoch = global_step // total_batches_per_epoch
        resume_step = global_step % total_batches_per_epoch
        logger.info(f"Computed resume position: epoch={start_epoch}, step_in_epoch={resume_step} (total_batches/epoch={total_batches_per_epoch})")
    
    quarter_steps = set()
    for q in [0.25, 0.50, 0.75]:
        quarter_steps.add(int(total_batches_per_epoch * q))
    logger.info(f"Quarter-epoch checkpoints at steps: {sorted(quarter_steps)} (total batches/epoch: {total_batches_per_epoch})")
    
    for epoch in range(start_epoch, args.epochs):
        logger.info(f"--- Epoch {epoch+1}/{args.epochs} ---")
        
        step_in_epoch = 0
        with tqdm(train_dataloader, desc=f"Epoch {epoch+1}") as pbar:
            for batch in pbar:
                # Skip steps if resuming mid-epoch
                if epoch == start_epoch and step_in_epoch < resume_step:
                    step_in_epoch += 1
                    global_step += 1
                    continue
                
                t_batch_start = time.time()
                with accelerator.accumulate(active_policy):
                    metrics = ppo_controller.step(batch, optimizer_policy, optimizer_value)
                    torch.cuda.empty_cache()
                    
                    t_batch_end = time.time()
                    batch_times.append(t_batch_end - t_batch_start)
                    
                    pbar.set_postfix({
                        "loss": f"{metrics['loss']:.4f}",
                        "reward": f"{metrics['reward']:.4f}",
                        "w_hat": f"{metrics['w_hat_mean']:.4f}",
                        "gpu_mem_gb": f"{torch.cuda.max_memory_allocated() / (1024**3):.2f}"
                    })
                
                global_step += 1
                step_in_epoch += 1
                
                # Quarter-epoch checkpoint
                if step_in_epoch in quarter_steps:
                    pct = int(100 * step_in_epoch / total_batches_per_epoch)
                    save_checkpoint(f"ep{epoch+1}-{pct}pct", epoch, step_in_epoch)
        
        # End-of-epoch checkpoint
        save_checkpoint(f"ep{epoch+1}-end", epoch + 1, 0)
                            
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
        
    t_end_total = time.time()
    total_training_duration = t_end_total - t_start_training
    avg_batch_time = sum(batch_times) / len(batch_times) if batch_times else 0
    
    timing_report = {
        "setup_time": setup_duration,
        "total_training_time": total_training_duration,
        "avg_batch_time": avg_batch_time,
        "num_batches": len(batch_times),
        "total_samples": len(batch_times) * args.per_device_train_batch_size
    }
    
    print(f"\nTIMING_REPORT_JSON: {json.dumps(timing_report)}")

if __name__ == "__main__":
    main()
