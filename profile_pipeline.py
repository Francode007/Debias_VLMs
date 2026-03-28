import os
import subprocess
import time
import threading
import glob
import pandas as pd
import json

class GPUProfiler:
    def __init__(self):
        self.running = False
        self.stats = []
        self.thread = None
        self.gpu_available = True

    def start(self):
        self.running = True
        self.stats = []
        
        # Test if nvidia-smi is available
        try:
            subprocess.check_output(["nvidia-smi"], stderr=subprocess.STDOUT)
        except (FileNotFoundError, subprocess.CalledProcessError):
            self.gpu_available = False
            return
            
        self.thread = threading.Thread(target=self._poll)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()
        
        if not self.gpu_available or not self.stats:
            return 0.0, 0.0
        
        utils, mems = zip(*self.stats)
        return sum(utils)/len(utils), max(mems)

    def _poll(self):
        while self.running:
            try:
                out = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                    text=True
                )
                
                max_util = 0.0
                max_mem = 0.0
                for line in out.strip().split('\n'):
                    parts = line.split(',')
                    if len(parts) == 2:
                        util = float(parts[0].strip())
                        mem = float(parts[1].strip())
                        max_util = max(max_util, util)
                        max_mem = max(max_mem, mem)
                
                self.stats.append((max_util, max_mem))
            except Exception:
                pass
            time.sleep(1.0)

def main():
    print("=== Pipeline Profiler ===")
    
    # 1. Determine Full Dataset Size
    data_path = "./sb_bench_data/data"
    parquet_files = glob.glob(os.path.join(data_path, "*.parquet"))
    
    total_raw_samples = 0
    if not parquet_files:
        print(f"No parquet files found in {data_path}. Attempting to run load_sb_bench.py...")
        subprocess.run([".venv/bin/python", "load_sb_bench.py"], check=True)
        parquet_files = glob.glob(os.path.join(data_path, "*.parquet"))
        
    for f in parquet_files:
        df = pd.read_parquet(f, engine="fastparquet")
        total_raw_samples += len(df)
        
    print(f"Total raw examples in dataset: {total_raw_samples}")
    total_pairs = total_raw_samples * 2
    print(f"Total preference pairs to process: {total_pairs}")
    
    # Determine python path
    python_cmd = ".venv/bin/python" if os.path.exists(".venv/bin/python") else "python"
    
    # Check if a model exists in local_model_config.py or fallback
    model_id = "Qwen/Qwen2-VL-2B-Instruct"  # Smaller model for testing/MAC
    
    results = {}
    profiler = GPUProfiler()
    
    print("\n--- Running Step 1: Embeddings (Smallset) ---")
    emb_dir = "./embeddings_output"
    if os.path.exists(emb_dir):
        for f in glob.glob(f"{emb_dir}/emb_*.npy"):
            os.remove(f)
            
    # cal_emb_modular.py uses smallset which limits to 5 samples (10 pairs)
    cmd1 = [
        python_cmd, "cal_emb_modular.py",
        "--device", "mps",  # fallback for Mac, or assume cuda if it's linux
        "--model", model_id,
        "--data_path", data_path,
        "--cls_embs_path", emb_dir,
        "--batch_size", "32",
        "--dataloader_num_workers", "8",
        "--use_smallset"
    ]
    
    # On non-Mac, we should use CUDA. Make it flexible:
    import platform
    if platform.system() != "Darwin":
        cmd1[4] = "cuda" # replace mps with cuda
        
    profiler.start()
    t0 = time.time()
    subprocess.run(cmd1, check=True)
    t1 = time.time()
    util1, mem1 = profiler.stop()
    
    step1_time = t1 - t0
    # use_smallset runs on 5 examples -> 10 pairs
    actual_pairs_processed = 10 
    time_per_pair = step1_time / actual_pairs_processed
    estimated_step1_full = time_per_pair * total_pairs
    
    results['Step 1 (Embeddings)'] = {
        'time_subset': step1_time,
        'mem_max_mb': mem1,
        'util_avg': util1,
        'est_full_time': estimated_step1_full
    }
    
    print("\n--- Running Step 2: PCA ---")
    cmd2 = [
        python_cmd, "generate_drm_heads.py",
        "--input_dir", emb_dir,
        "--output_dir", "./generated_heads",
        "--n_components", "5",  # Reduced for safety on smallset
        "--case_name", "sb_bench"
    ]
    
    profiler.start()
    t0 = time.time()
    subprocess.run(cmd2, check=True)
    t1 = time.time()
    util2, mem2 = profiler.stop()
    
    step2_time = t1 - t0
    # PCA scales heavily with N, but for extrapolation we use a linear approximation 
    estimated_step2_full = step2_time * (total_pairs / actual_pairs_processed) 
    
    results['Step 2 (PCA)'] = {
        'time_subset': step2_time,
        'mem_max_mb': mem2,
        'util_avg': util2,
        'est_full_time': estimated_step2_full
    }
    
    print("\n--- Running Step 3: Evaluate ---")
    cmd3 = [
        python_cmd, "evaluate_drm_heads.py",
        "--emb_dir", emb_dir,
        "--score_head_weight", "./generated_heads/sb_bench-PCA-component",
        "--data_path", data_path,
        "--batch_size", "1024",
        "--output_json", "./drm_head_results.json"
    ]
    if platform.system() != "Darwin":
        cmd3.extend(["--device", "cuda"])
        
    profiler.start()
    t0 = time.time()
    subprocess.run(cmd3, check=True)
    t1 = time.time()
    util3, mem3 = profiler.stop()
    
    step3_time = t1 - t0
    estimated_step3_full = step3_time * (total_pairs / actual_pairs_processed)
    
    results['Step 3 (Evaluate)'] = {
        'time_subset': step3_time,
        'mem_max_mb': mem3,
        'util_avg': util3,
        'est_full_time': estimated_step3_full
    }
    
    print("\n" + "="*60)
    print("PROFILING & EXTRAPOLATION REPORT".center(60))
    print("="*60)
    print(f"Dataset Name:  SB-Bench")
    print(f"Total Raw Examples:       {total_raw_samples}")
    print(f"Total Preference Pairs:   {total_pairs}")
    print(f"Pairs profiled (smallset): {actual_pairs_processed}")
    print("-" * 60)
    
    total_est_hours = 0
    for step, data in results.items():
        print(f"[{step}]")
        print(f"  Execution Time (subset) : {data['time_subset']:.2f} seconds")
        if profiler.gpu_available:
            print(f"  Max GPU Memory Used     : {data['mem_max_mb']:.2f} MB")
            print(f"  Avg GPU Utilization     : {data['util_avg']:.2f}%")
        else:
            print(f"  GPU Stats               : N/A (nvidia-smi not found, e.g. Mac/MPS)")
            
        est_hr = data['est_full_time'] / 3600.0
        total_est_hours += est_hr
        print(f"  -> Extrapolated Time (Full) : {est_hr:.2f} hours")
        print()
        
    print("-" * 60)
    print(f"TOTAL ESTIMATED FULL PIPELINE TIME: {total_est_hours:.2f} hours")
    print("="*60)
    
    # Optimization recommendations depending on budget
    print("\nOptimization & Budget Report:")
    print(f"1. Target Budget    : 1.00 hour of GPU Compute")
    print(f"2. Current Estimate : {total_est_hours:.2f} hours")
    
    print("\nSuggestions for Optimization:")
    if total_est_hours > 1.0:
        print("   - You exceed the 1 hour budget. You MUST optimize.")
        print("   - Embeddings: Process with larger batch sizes if VRAM allows.")
        print("   - Model Swap: Use a smaller model like Qwen2-VL-2B instead of Qwen2.5-VL-7B.")
        print("   - Precision: Ensure you run with FP16/BF16 on CUDA rather than FP32.")
    else:
        print("   - You are within the 1-hour GPU compute budget!")
        print("   - You can scale up the batch size to finish even faster, or test larger models.")
        
    print("   - PCA Step: Memory scales heavily with sample count. Ensure your host RAM is sufficient.")
    
    with open("profiling_report.json", "w") as f:
        json.dump(results, f, indent=4)
    print("\nSaved detailed metrics to profiling_report.json")

if __name__ == "__main__":
    main()
