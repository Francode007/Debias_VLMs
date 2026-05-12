from datasets import load_dataset
import os
import shutil

def load_and_save_sb_bench():
    """
    Loads the SB-Bench dataset from HuggingFace and saves it to disk 
    in the format expected by cal_emb_modular.py.
    """
    dataset_name = "ucf-crcv/SB-Bench"
    output_dir = "./sb_bench_data/data"
    
    print(f"Loading dataset: {dataset_name}")
    
    try:
        # Load the dataset
        ds = load_dataset(dataset_name)
        print(f"Dataset loaded: {ds}")
        
        # Ensure output directory exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Save validation/test split (usually what we want for evaluation/analysis)
        # SB-Bench typically has a 'test' or 'validation' split.
        # If 'train' is the only one, we use that.
        
        target_split = 'train'
        if 'test' in ds:
            target_split = 'test'
        elif 'validation' in ds:
            target_split = 'validation'
        elif 'real' in ds:
            target_split = 'real'
        elif 'synthetic' in ds:
             target_split = 'synthetic'
        
        if target_split not in ds:
             # Fallback: take the first available split
             target_split = list(ds.keys())[0]

        print(f"Saving split '{target_split}' to {output_dir}")
        
        # Save as parquet files (which cal_emb_modular.py expects)
        # We can save the whole shard as one file or multiple.
        save_path = os.path.join(output_dir, "sb_bench_data.parquet")
        ds[target_split].to_parquet(save_path)
        
        print(f"Successfully saved data to {save_path}")
        
    except Exception as e:
        print(f"Error loading/saving dataset: {e}")
        import traceback
        traceback.print_exc()
        print("Please ensure you are logged in via `hf login` if needed.")

if __name__ == "__main__":
    load_and_save_sb_bench()
