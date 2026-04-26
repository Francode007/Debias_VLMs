import os
import subprocess
import time
import threading
import glob
import pandas as pd
import json
import platform
import argparse

class GPUProfiler:
    def __init__(self):
        self.running = False
        self.stats = []
        self.thread = None
        self.gpu_available = True

    def start(self):
        self.running = True
        self.stats = []
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
    parser = argparse.ArgumentParser(description="Profile RL Generation Pipeline over SB-Bench")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for RL Training Configuration")
    parser.add_argument("--gradient_accumulation", type=int, default=4, help="Grad Accum scale")
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs to profile")
    args = parser.parse_args()

    print("=== RL PPO Pipeline Profiler ===")
    
    data_path = "./sb_bench_data/data"
    parquet_files = glob.glob(os.path.join(data_path, "*.parquet"))
    
    total_raw_samples = 0
    if not parquet_files:
        print(f"No parquet files found in {data_path}. Attempting to run load_sb_bench.py...")
        subprocess.run([".venv/bin/python", "load_sb_bench.py"], check=True)
        parquet_files = glob.glob(os.path.join(data_path, "*.parquet"))
        
    for f in parquet_files:
        try:
            df = pd.read_parquet(f, engine="fastparquet")
            total_raw_samples += len(df)
        except Exception:
            pass
            
    print(f"Total raw examples in complete dataset: {total_raw_samples}")
    python_cmd = ".venv/bin/python" if os.path.exists(".venv/bin/python") else "python"
    
    results = {}
    profiler = GPUProfiler()
    
    print(f"\n--- Running True Dual Generation PPO Loop (10 samples, {args.epochs} epochs) ---")
    
    cmd_rl = [
        python_cmd, "train_rl.py",
        "--per_device_train_batch_size", str(args.batch_size),
        "--gradient_accumulation_steps", str(args.gradient_accumulation),
        "--data_path", data_path,
        "--use_smallset",
        "--epochs", str(args.epochs)
    ]
    
    profiler.start()
    t0 = time.time()
    
    # Run and capture output
    env = os.environ.copy()
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    process = subprocess.Popen(cmd_rl, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    
    timing_report = None
    for line in process.stdout:
        print(line, end="") # Stream output to user
        if "TIMING_REPORT_JSON:" in line:
            report_str = line.split("TIMING_REPORT_JSON:")[1].strip()
            timing_report = json.loads(report_str)
            
    process.wait()
    t1 = time.time()
    util_avg, mem_max = profiler.stop()
    
    if process.returncode != 0:
        print(f"\nError: train_rl.py exited with code {process.returncode}")
        return

    if not timing_report:
        print("\nError: Could not find TIMING_REPORT_JSON in output.")
        return

    # Extrapolate based on actual batch time
    avg_batch_time = timing_report["avg_batch_time"]
    num_batches_total = (total_raw_samples + args.batch_size - 1) // args.batch_size
    
    # Total time = Setup Time + (Avg Batch Time * Total Batches)
    est_total_time_sec = timing_report["setup_time"] + (avg_batch_time * num_batches_total)
    
    results['PPO Dual Generation (Epoch 1)'] = {
        'total_real_time': t1 - t0,
        'setup_time': timing_report["setup_time"],
        'avg_batch_time': avg_batch_time,
        'mem_max_mb': mem_max,
        'util_avg': util_avg,
        'est_full_time_sec': est_total_time_sec
    }
    
    print("\n" + "="*60)
    print("PROFILING & EXTRAPOLATION REPORT".center(60))
    print("="*60)
    print(f"Dataset Name:  SB-Bench (RL PPO Generation)")
    print(f"Total Raw Examples:       {total_raw_samples}")
    print(f"Batches per Epoch:        {num_batches_total}")
    print("-" * 60)
    
    est_hr = est_total_time_sec / 3600.0
    print(f"  Setup Time              : {timing_report['setup_time']:.2f} seconds")
    print(f"  Avg Time per Batch      : {avg_batch_time:.2f} seconds")
    print(f"  Max GPU Memory Used     : {mem_max:.2f} MB")
    print(f"  Avg GPU Utilization     : {util_avg:.2f}%")
    print(f"  -> Extrapolated Time (Full Epoch) : {est_hr:.2f} hours")
    print("-" * 60)
    
    print("\nOptimization & Budget Report:")
    print(f"1. Target Budget    : 2.00 hours per Epoch")
    print(f"2. Current Estimate : {est_hr:.2f} hours")
    
    if est_hr > 2.0:
        print("\nSuggestions for Optimization:")
        print("   - High PPO load detected. Flash Attention 2 is active.")
        print("   - Try increasing batch_size if memory permits.")
        print("   - Increase gradient accumulation to improve compute density.")
        
    with open("profiling_report_rl.json", "w") as f:
        json.dump(results, f, indent=4)
    print("\nSaved detailed metrics to profiling_report_rl.json")

if __name__ == "__main__":
    main()
