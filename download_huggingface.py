from pathlib import Path
from huggingface_hub import snapshot_download

# Local root directory
ROOT_DIR = Path("/data/project/le-lab/fSWIFT/model_hub")

# Hugging Face repo_id -> local folder name
MODELS = {
    "Mistral-7B-v0.1": "mistralai/Mistral-7B-v0.1",
    "Qwen3-32B": "Qwen/Qwen3-32B",
    "Qwen2.5-7B-Instruct": "Qwen/Qwen2.5-7B-Instruct",
}

for model_name, repo_id in MODELS.items():
    target_dir = ROOT_DIR / model_name / "base"
    target_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"Downloading: {repo_id}")
    print(f"Saving to:   {target_dir}")
    print("=" * 80)

    local_path = snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        local_dir=str(target_dir),
        max_workers=8,
        resume_download=True,
    )

    print(f"Finished: {repo_id}")
    print(f"Local path: {local_path}\n")

print("All models downloaded successfully.")