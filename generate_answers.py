import argparse
import os
import json
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel
from PIL import Image
import io

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Path to LoRA checkpoint (omit for vanilla baseline)")
    parser.add_argument("--data_path", type=str, required=True, help="Path to parquet dataset")
    parser.add_argument("--output_jsonl", type=str, required=True, help="Path to save generated outputs")
    parser.add_argument("--batch_size", type=int, default=16, help="Inference batch size")
    return parser.parse_args()

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading Base Model: {args.base_model}")
    processor = AutoProcessor.from_pretrained(args.base_model)
    # Important: padding side left for generation
    processor.tokenizer.padding_side = "left"

    base_model = AutoModelForImageTextToText.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2" if torch.cuda.is_available() else "eager",
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True
    )

    if args.checkpoint_dir:
        print(f"Loading LoRA Checkpoint: {args.checkpoint_dir}")
        model = PeftModel.from_pretrained(base_model, args.checkpoint_dir)
    else:
        print("ℹ️ No --checkpoint_dir provided. Running vanilla Qwen2.5-VL-3B-Instruct (no LoRA adapter).")
        model = base_model
    model.eval()

    print(f"Loading Dataset: {args.data_path}")
    ds = load_dataset("parquet", data_files={"test": args.data_path})["test"]

    os.makedirs(os.path.dirname(args.output_jsonl), exist_ok=True)
    
    results = []
    
    # Process in batches
    for i in tqdm(range(0, len(ds), args.batch_size), desc="Generating Answers"):
        batch = ds[i:i+args.batch_size]
        
        batch_messages = []
        batch_images = []
        question_ids = []
        
        for j in range(len(batch["image"])):
            # Handle different common dataset column names
            question = batch.get("question", batch.get("text"))[j]
            qid = batch.get("question_id", batch.get("id"))[j]
            img = batch["image"][j]
            
            # Ensure it's a PIL image
            if isinstance(img, dict) and "bytes" in img:
                img = Image.open(io.BytesIO(img["bytes"])).convert("RGB")
            elif not isinstance(img, Image.Image):
                img = Image.open(io.BytesIO(img)).convert("RGB")
                
            batch_images.append(img)
            question_ids.append(qid)
            
            message = [
                {"role": "user", "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": question}
                ]}
            ]
            batch_messages.append(message)
            
        texts = [
            processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in batch_messages
        ]
        
        inputs = processor(
            text=texts,
            images=batch_images,
            padding=True,
            return_tensors="pt"
        ).to(device)
        
        with torch.inference_mode():
            generated_ids = model.generate(**inputs, max_new_tokens=10)
            
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        output_texts = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        for qid, text in zip(question_ids, output_texts):
            results.append({"question_id": qid, "text": text.strip()})

    print(f"Saving {len(results)} outputs to {args.output_jsonl}")
    with open(args.output_jsonl, "w") as f:
        for res in results:
            f.write(json.dumps(res) + "\n")
            
    print("Generation complete!")

if __name__ == "__main__":
    main()
