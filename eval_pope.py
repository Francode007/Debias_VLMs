import os
import json
import argparse
from tqdm import tqdm
import re

def clean_answer(text: str) -> str:
    """Clean and normalize model output for Yes/No."""
    if not text:
        return "unknown"
    
    text = text.lower().strip()
    
    # Remove common extra words
    text = re.sub(r'^[^a-z]*', '', text)   # remove leading non-letters
    text = re.sub(r'[^a-z\s]+$', '', text) # remove trailing punctuation
    
    # Strong matching
    if re.search(r'\byes\b', text):
        return "yes"
    elif re.search(r'\bno\b', text):
        return "no"
    else:
        return "unknown"


parser = argparse.ArgumentParser()
parser.add_argument("--gt_files", type=str, required=True,
                    help="Path to ground truth POPE JSON file")
parser.add_argument("--gen_files", type=str, required=True,
                    help="Path to generated answers JSONL from Qwen2.5-VL")
args = parser.parse_args()

# Load ground truth
print(f"Loading GT from: {args.gt_files}")
with open(os.path.expanduser(args.gt_files), "r") as f:
    gt_files = [json.loads(line) for line in f]

# Load generated answers
print(f"Loading generations from: {args.gen_files}")
with open(os.path.expanduser(args.gen_files), "r") as f:
    gen_files = [json.loads(line) for line in f]

# Validation
assert len(gt_files) == len(gen_files), f"GT has {len(gt_files)} samples, Gen has {len(gen_files)}"

true_pos = true_neg = false_pos = false_neg = unknown = 0
yes_answers = 0
total_questions = len(gt_files)

print("Starting evaluation...")

for idx, (gt_line, gen_line) in enumerate(tqdm(zip(gt_files, gen_files), total=total_questions)):
    gt_id = gt_line["question_id"]
    gen_id = gen_line["question_id"]
    
    assert gt_id == gen_id, f"Question ID mismatch at index {idx}: {gt_id} vs {gen_id}"

    gt_answer = gt_line["label"].lower().strip()
    gen_text = gen_line["text"]
    gen_answer = clean_answer(gen_text)

    if gt_answer == "yes":
        if gen_answer == "yes":
            true_pos += 1
            yes_answers += 1
        elif gen_answer == "no":
            false_neg += 1
        else:
            unknown += 1
            false_neg += 1   # treat unknown as wrong for recall

    elif gt_answer == "no":
        if gen_answer == "no":
            true_neg += 1
        elif gen_answer == "yes":
            false_pos += 1
            yes_answers += 1
        else:
            unknown += 1
            false_pos += 1   # treat unknown as wrong for precision

    else:
        print(f"Warning: Unknown GT label '{gt_answer}' at index {idx}")
        unknown += 1

# ====================== Compute Metrics ======================
precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0.0
recall    = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0.0
f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
accuracy  = (true_pos + true_neg) / total_questions if total_questions > 0 else 0.0
yes_prop  = yes_answers / total_questions if total_questions > 0 else 0.0
unk_prop  = unknown / total_questions if total_questions > 0 else 0.0

# ====================== Print Results ======================
print("\n" + "="*60)
print("POPE EVALUATION RESULTS - SB-Bench + Qwen2.5-VL")
print("="*60)
print(f"Accuracy     : {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Precision    : {precision:.4f} ({precision*100:.2f}%)")
print(f"Recall       : {recall:.4f} ({recall*100:.2f}%)")
print(f"F1 Score     : {f1:.4f} ({f1*100:.2f}%)")
print(f"Yes Proportion: {yes_prop:.4f} ({yes_prop*100:.2f}%)")
print(f"Unknown Rate : {unk_prop:.4f} ({unk_prop*100:.2f}%)")
print("="*60)

# Optional: Save detailed results
detailed_results = {
    "metrics": {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "yes_proportion": yes_prop,
        "unknown_rate": unk_prop,
        "total_samples": total_questions,
        "true_pos": true_pos,
        "true_neg": true_neg,
        "false_pos": false_pos,
        "false_neg": false_neg
    },
    "config": {
        "gt_file": args.gt_files,
        "gen_file": args.gen_files
    }
}

output_path = args.gen_files.replace(".jsonl", "_pope_results.json")
with open(output_path, "w") as f:
    json.dump(detailed_results, f, indent=4)

print(f"Detailed results saved to: {output_path}")