import argparse
import torch
import os
import json
from tqdm import tqdm
from PIL import Image
import io
import pandas as pd
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
import shortuuid

def eval_model(args):
    # ========================== Load Model ==========================
    print("Loading Qwen2-VL model...")
    processor = AutoProcessor.from_pretrained(args.model_path)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )

    # ========================== Load Dataset ==========================
    print(f"Loading dataset from: {args.question_file}")
    df = pd.read_parquet(args.question_file, engine="fastparquet")

    if args.max_samples is not None:
        df = df.head(args.max_samples)

    print(f"Total samples to evaluate: {len(df)}")

    # Create output directory
    os.makedirs(os.path.dirname(args.answers_file), exist_ok=True)
    ans_file = open(args.answers_file, "w")

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Running Inference"):
        # Load image from bytes
        image_data = row['file_name.bytes']
        image = Image.open(io.BytesIO(image_data)).convert("RGB")

        # Prepare question (adapt to your needs)
        prompt_text = row.get('context', '') + " " + row.get('question', '')
        
        # POPE-style Yes/No question (most suitable for hallucination eval)
        # You can change this prompt depending on what you want to evaluate
        qs = f"Is there a {row.get('object', 'person')} in the image? Answer only with YES or NO."

        # Optional: Use your original question if not doing pure POPE
        # qs = prompt_text + " Please answer this question with one word or letter."

        # Build messages for Qwen2-VL chat template
        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": qs}
            ]}
        ]

        # Apply chat template
        text_input = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Process inputs
        inputs = processor(
            text=[text_input],
            images=[image],
            padding=True,
            return_tensors="pt"
        ).to(model.device)

        # ========================== Generation ==========================
        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=32,          # Short for Yes/No answers
                do_sample=False,            # Deterministic for evaluation
                temperature=None,
                num_beams=args.num_beams,
                use_cache=True,
            )

        # Decode output
        outputs = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        outputs = outputs.strip()

        # Create question_id (use data_index if available, else generate one)
        question_id = row.get("data_index", idx)
        if not isinstance(question_id, (int, str)):
            question_id = idx

        # Write to answers file in the same format as LLaVA POPE eval expects
        ans_file.write(json.dumps({
            "question_id": question_id,
            "prompt": qs,
            "text": outputs,
            "model_id": os.path.basename(args.model_path),
            "image": f"image_{idx}.jpg",   # placeholder, since image is in bytes
            "metadata": {}
        }) + "\n")

        ans_file.flush()

    ans_file.close()
    print(f"\nInference completed! Answers saved to: {args.answers_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, 
                        default="Qwen/Qwen2.5-VL-7B-Instruct",
                        help="Path to Qwen2-VL model")
    parser.add_argument("--question-file", type=str, 
                        default="/content/drive/MyDrive/Debias_VLMs/Sb-Bench_Dataset/Real/0000.parquet",
                        help="Path to your SB-Bench parquet file")
    parser.add_argument("--answers-file", type=str, 
                        default="./output/qwen_sb_bench_pope_random.jsonl",
                        help="Output JSONL file for generated answers")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit number of samples (for debugging)")
    parser.add_argument("--num-beams", type=int, default=1,
                        help="Number of beams for generation")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    eval_model(args)