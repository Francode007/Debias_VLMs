from datasets import load_dataset
import os
import shutil

def load_and_save_pope():
    """
    Loads the POPE dataset from HuggingFace and saves it to disk 
    in the format expected for evaluation.
    """
    dataset_name = "lmms-lab/POPE"
    output_dir = os.getenv("POPE_DATA_PATH", "./pope_data/data")
    
    print(f"Loading dataset: {dataset_name}")
    
    try:
        # Load the dataset
        ds = load_dataset(dataset_name)
        print(f"Dataset loaded: {ds}")
        
        # Ensure output directory exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # POPE typically has 'test' split. 
        # If not, fallback to first available.
        target_split = 'test'
        if target_split not in ds:
            if 'validation' in ds:
                target_split = 'validation'
            elif 'train' in ds:
                target_split = 'train'
            else:
                target_split = list(ds.keys())[0]

        print(f"Saving split '{target_split}' to {output_dir}")
        
        # Save as parquet files (for generate_answers.py)
        save_path_parquet = os.path.join(output_dir, "pope_data.parquet")
        ds[target_split].to_parquet(save_path_parquet)
        print(f"Successfully saved parquet data to {save_path_parquet}")

        # Map 'answer' to 'label' for eval_pope.py compatibility
        def map_for_eval(batch):
            batch["label"] = batch["answer"]
            return batch
            
        ds_mapped = ds[target_split].map(map_for_eval, batched=True)
        
        # Save as JSONL (for eval_pope.py)
        save_path_jsonl = os.path.join(output_dir, "pope_data.jsonl")
        # Remove 'image' column to keep the JSONL small and fast to load (eval_pope doesn't need images)
        ds_mapped_no_img = ds_mapped.remove_columns(["image"])
        ds_mapped_no_img.to_json(save_path_jsonl)
        print(f"Successfully saved JSONL data to {save_path_jsonl}")
        
    except Exception as e:
        print(f"Error loading/saving dataset: {e}")
        import traceback
        traceback.print_exc()
        print("Please ensure you are logged in via `hf login` if needed.")

if __name__ == "__main__":
    load_and_save_pope()
