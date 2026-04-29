import torch
from transformers import AutoProcessor
from PIL import Image

processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
img1 = Image.new('RGB', (100, 100))
img2 = Image.new('RGB', (200, 200))

kwargs = {
    "padding": False,
    "truncation": False,
    "return_tensors": None,
}

msg1 = [{"role": "user", "content": [{"type": "image", "image": img1}, {"type": "text", "text": "hello"}]}]
msg2 = [{"role": "user", "content": [{"type": "image", "image": img2}, {"type": "text", "text": "world"}]}]

t1 = processor.apply_chat_template(msg1, tokenize=False, add_generation_prompt=True)
t2 = processor.apply_chat_template(msg2, tokenize=False, add_generation_prompt=True)

inputs = processor(text=[t1, t2], images=[img1, img2], **kwargs)

print("INPUT IDS:", type(inputs["input_ids"]))
if isinstance(inputs["input_ids"], list):
    print("input_ids len:", len(inputs["input_ids"]))
else:
    print("input_ids shape:", inputs["input_ids"].shape)
