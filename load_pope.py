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
        
        # Save as parquet files
        save_path = os.path.join(output_dir, "pope_data.parquet")
        ds[target_split].to_parquet(save_path)
        
        print(f"Successfully saved data to {save_path}")
        
    except Exception as e:
        print(f"Error loading/saving dataset: {e}")
        import traceback
        traceback.print_exc()
        print("Please ensure you are logged in via `hf login` if needed.")

if __name__ == "__main__":
    load_and_save_pope()
