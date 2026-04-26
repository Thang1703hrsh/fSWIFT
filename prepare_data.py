import os
import json
from datasets import load_dataset

# Define the target directory and file path
output_dir = "data/Ultrachat200k/SFT"
output_file = os.path.join(output_dir, "trainSFT.jsonl")

os.makedirs(output_dir, exist_ok=True)

print("Downloading HuggingFaceH4/ultrachat_200k...")
# Load the train_sft split
dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")

print("Sampling and formatting 50,000 samples...")
sampled_dataset = dataset.select(range(50000))

with open(output_file, "w", encoding="utf-8") as f:
    for item in sampled_dataset:
        # In UltraChat_200k, 'messages' is a list: [{'role': 'user', 'content': '...'}, {'role': 'assistant', 'content': '...'}]
        # We extract the user prompt and the assistant's response (ground truth)
        if len(item['messages']) >= 2:
            prompt_text = item['messages'][0]['content']
            reference_text = item['messages'][1]['content']
            
            # Construct the dictionary with the 'ground_truth' key required by your script
            formatted_item = {
                "prompt": prompt_text,
                "ground_truth": reference_text
            }
            f.write(json.dumps(formatted_item) + "\n")

print(f"Success! Data saved with 'ground_truth' keys to {output_file}")