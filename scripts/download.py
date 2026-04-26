from huggingface_hub import snapshot_download

# DOWNLOAD BASE MODEL FOR TABLE 1
repo_id_base = "Qwen/Qwen1.5-1.8B" 
local_dir_base = "model_hub/Qwen1.5-1.8B/base"  

print(f"Downloading model from {repo_id_base} to {local_dir_base}...")
snapshot_download(
    repo_id=repo_id_base,
    repo_type="model", 
    local_dir=local_dir_base, 
    local_dir_use_symlinks=False, 
)
print(f"Base Model successfully downloaded to {local_dir_base}!")

###################################################################

# DOWNLOAD TEACHER MODEL FOR TABLE 1
repo_id_teacher = "alignment-handbook/zephyr-7b-sft-full" 
local_dir_teacher = "model_hub/zephyr-7b-sft-full"  

print(f"Downloading model from {repo_id_teacher} to {local_dir_teacher}...")
snapshot_download(
    repo_id=repo_id_teacher,
    repo_type="model", 
    local_dir=local_dir_teacher, 
    local_dir_use_symlinks=False, 
)
print(f"Teacher Model successfully downloaded to {local_dir_teacher}!")

###################################################################

# DOWNLOAD DATASET
# Repository ID and the local directory to save it
repo_id_dataset = "UCLA-AGI/SPIN_iter1" 
local_dir_dataset = "data_hub/SPIN_iter1"  # You can rename this directory

print(f"Downloading dataset from {repo_id_dataset} to {local_dir_dataset}...")
snapshot_download(
    repo_id=repo_id_dataset,
    repo_type="dataset",  # Specify the repo type as a dataset
    local_dir=local_dir_dataset,  
    local_dir_use_symlinks=False, 
)
print(f"Dataset successfully downloaded to {local_dir_dataset}!")