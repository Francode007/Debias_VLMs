
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, Qwen2VLImageProcessor
from PIL import Image
import requests
import io
import numpy as np

def debug_qwen():
    print("Loading model/processor...")
    model_name = "Qwen/Qwen2-VL-7B-Instruct"
    try:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name, 
            torch_dtype=torch.float16, 
            device_map="auto"
        )
        processor = AutoProcessor.from_pretrained(model_name)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print("Creating dummy inputs...")
    # Create distinct images to verify stacking behavior
    img1 = Image.new('RGB', (100, 100), color='red')
    img2 = Image.new('RGB', (100, 100), color='blue')
    
    # Text
    text1 = "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe this.<|im_end|>\n<|im_start|>assistant\nIt is red.<|im_end|>"
    text2 = "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe this.<|im_end|>\n<|im_start|>assistant\nIt is blue.<|im_end|>"

    # 1. Test Processor Batching (Reference)
    print("\n--- Reference: Processor Batching ---")
    inputs_batch = processor(
        text=[text1, text2],
        images=[img1, img2],
        padding=True,
        return_tensors="pt"
    )
    
    print("Keys in batch:", inputs_batch.keys())
    for k, v in inputs_batch.items():
        print(f"{k}: shape={v.shape}")

    # Move to device
    inputs_batch = {k: v.to(model.device) for k, v in inputs_batch.items()}

    # Run Forward
    print("Running forward pass with processor batch...")
    try:
        with torch.no_grad():
            output = model(**inputs_batch)
        print("Success!")
    except Exception as e:
        print(f"Reference Forward failed: {e}")

    # 2. Test Manual Collation (Simulation of Data Collator)
    print("\n--- Simulation: Manual Collation ---")
    # Step A: Process individually (DatasetBuilder)
    in1 = processor(text=[text1], images=[img1], padding="max_length", max_length=128, return_tensors="pt")
    in2 = processor(text=[text2], images=[img2], padding="max_length", max_length=128, return_tensors="pt")

    print("Individual 1 shapes:")
    for k, v in in1.items():
        print(f"  {k}: {v.shape}")

    # Step B: Collate (DataCollator)
    # Replicate logic: stack input_ids, cat pixel_values/grids
    print("Collating...")
    collated = {}
    
    # input_ids: Stack (B, L)
    # in1['input_ids'] is (1, L)
    collated['input_ids'] = torch.cat([in1['input_ids'], in2['input_ids']], dim=0) # Result (2, L)
    collated['attention_mask'] = torch.cat([in1['attention_mask'], in2['attention_mask']], dim=0) # Result (2, L)
    
    # pixel_values: Cat (2N, D)
    # in1['pixel_values'] is (N, D)
    collated['pixel_values'] = torch.cat([in1['pixel_values'], in2['pixel_values']], dim=0)
    
    # image_grid_thw: Cat (2, 3)
    # in1['image_grid_thw'] is (1, 3)
    if 'image_grid_thw' in in1:
        collated['image_grid_thw'] = torch.cat([in1['image_grid_thw'], in2['image_grid_thw']], dim=0)

    print("Collated shapes:")
    for k, v in collated.items():
        print(f"  {k}: {v.shape}")

    # Move to device
    collated = {k: v.to(model.device) for k, v in collated.items()}

    # Run Forward
    print("Running forward pass with manually collated batch...")
    try:
        with torch.no_grad():
            output = model(**collated)
        print("Success!")
    except Exception as e:
        # Check for specific error message matching our observation
        print(f"Manual Forward failed: {e}")

if __name__ == "__main__":
    debug_qwen()
