#!/usr/bin/env python3
"""
Example script to run the flexible cal_emb.py
This script shows how to use the modular version with different configurations
"""

import argparse
import sys
import os

def main():
    parser = argparse.ArgumentParser(description="Run flexible cal_emb with different device configurations")
    
    # Device options
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto",
                       help="Device to use (auto will detect best available)")
    parser.add_argument("--force-fp32", action="store_true",
                       help="Force float32 precision for compatibility")
    
    # Data options
    parser.add_argument("--data-path", default="./sb_bench_data/data",
                       help="Path to data directory")
    parser.add_argument("--small-dataset", action="store_true",
                       help="Use a small subset for testing")
    parser.add_argument("--max-samples", type=int, default=None,
                       help="Maximum number of samples to process")
    
    # Output options
    parser.add_argument("--output-dir", default="./embeddings_output",
                       help="Directory to save embeddings")
    parser.add_argument("--log-dir", default="./logs",
                       help="Directory for logs")
    
    # Model options
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct",
                       help="Model to use")
    parser.add_argument("--max-length", type=int, default=512,
                       help="Maximum sequence length (reduce for memory constraints)")
    parser.add_argument("--batch-size", type=int, default=1,
                       help="Processing batch size (affects memory usage)")
    parser.add_argument("--dataloader-batch-size", type=int, default=None,
                       help="DataLoader batch size (defaults to batch-size if not set)")
    
    # Memory management options
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16,
                       help="Gradient accumulation steps")
    parser.add_argument("--reduce-memory", action="store_true",
                       help="Use memory-optimized settings (smaller batch sizes, etc.)")
    
    # Debug options
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug mode")
    parser.add_argument("--dry-run", action="store_true",
                       help="Test setup without running full processing")
    
    args = parser.parse_args()
    
    # Import the flexible module
    try:
        from cal_emb_flexible import main as run_cal_emb, ScriptArguments
    except ImportError as e:
        print(f"Error importing cal_emb_flexible: {e}")
        print("Make sure all dependencies are installed: pip install -r requirements.txt")
        sys.exit(1)
    
    # Configure arguments
    script_args = ScriptArguments(
        device=args.device,
        force_fp32=args.force_fp32,
        data_path=args.data_path,
        cls_embs_path=args.output_dir,
        log_dir=args.log_dir,
        base_model=args.model,
        max_length=args.max_length,
        batch_size=args.batch_size,
        dataloader_batch_size=args.dataloader_batch_size,
        per_device_train_batch_size=args.dataloader_batch_size or args.batch_size,
        per_device_eval_batch_size=args.dataloader_batch_size or args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        debug=args.debug,
        use_smallset=args.small_dataset,
    )
    
    # Apply memory optimization settings if requested
    if args.reduce_memory:
        script_args.batch_size = 1
        script_args.dataloader_batch_size = 1
        script_args.per_device_train_batch_size = 1
        script_args.per_device_eval_batch_size = 1
        script_args.gradient_accumulation_steps = max(script_args.gradient_accumulation_steps, 32)
        script_args.max_length = min(script_args.max_length, 256)
        print("Applied memory optimization settings")
    
    print("=" * 50)
    print("Flexible Cal_Emb Runner")
    print("=" * 50)
    print(f"Device: {args.device}")
    print(f"Force FP32: {args.force_fp32}")
    print(f"Data path: {args.data_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Model: {args.model}")
    print(f"Max length: {args.max_length}")
    print(f"Processing batch size: {args.batch_size}")
    print(f"DataLoader batch size: {args.dataloader_batch_size or args.batch_size}")
    print(f"Gradient accumulation steps: {args.gradient_accumulation_steps}")
    print(f"Memory optimization: {args.reduce_memory}")
    
    if args.dry_run:
        print("\nDry run mode - setup test only")
        print("Configuration looks good!")
        return
    
    # Check data path exists
    if not os.path.exists(args.data_path):
        print(f"Error: Data path {args.data_path} does not exist")
        print("Please check the path or download the sb_bench_data")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    print("\nStarting processing...")
    
    # Run the main function
    try:
        # Override sys.argv to pass arguments to the script
        original_argv = sys.argv
        sys.argv = ["cal_emb_flexible.py"]  # Reset argv for HfArgumentParser
        
        # Set up environment with our script args
        import cal_emb_flexible
        cal_emb_flexible.script_args = script_args
        
        # Run main
        result = run_cal_emb()
        print(f"\nProcessing completed successfully!")
        print(f"Results saved to: {args.output_dir}")
        
    except Exception as e:
        print(f"\nError during processing: {e}")
        sys.exit(1)
    finally:
        sys.argv = original_argv

if __name__ == "__main__":
    main()
