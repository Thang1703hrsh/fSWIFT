#! /bin/bash
set -euo pipefail

MODEL="model_hub/Qwen1.5-1.8B/WSPIN/ite0"
INPUT="data/Ultrachat200k/SFT/trainSFT.jsonl"
OUTDIR="data/Ultrachat200k/WSPIN/ite1"
BATCH=8
MAX_NEW=512
FRAC_LEN=1000000

# loop over data_frac = 0..70
for frac in {0..1}; do
  python generate.py \
    --model       "$MODEL" \
    --input_dir   "$INPUT" \
    --output_dir  "$OUTDIR" \
    --batch_size  $BATCH \
    --max_new_tokens $MAX_NEW \
    --data_frac   $frac \
    --frac_len    $FRAC_LEN \
    --split       train
done

model_name_1="model_hub/zephyr-7b-sft-full"
model_name_2="model_hub/Qwen1.5-1.8B/WSPIN/ite0"
input_dir="data/Ultrachat200k/WSPIN/ite1"
output_dir="data/Ultrachat200k/WSPIN/ite1"
model1_template="normal"
model2_template="normal"
batch_size=4
num_gpus=1
max_length=1024
max_prompt_length=512
force_sequential=false  # Set to true if multiprocessing causes issues
# Create output directory if it doesn't exist
# mkdir -p $output_dir
# Run the parallel processing script
python token_weight_estimation.py   --model_name_1 $model_name_1   --model_name_2 $model_name_2   --model1_template $model1_template   --model2_template $model2_template   --input_dir $input_dir   --output_dir $output_dir  --max_length $max_length  --max_prompt_length $max_prompt_length   --batch_size $batch_size   --num_gpus $num_gpus   $(if $force_sequential; then echo "--force_sequential"; fi) 


python -u train.py \
  model=qwen \
  model.name_or_path=model_hub/Qwen1.5-1.8B/WSPIN/ite0 \
  loss=tisdpo \
  base_data_dir=data \
  datasets='["Ultrachat200k/WSPIN/ite0","Ultrachat200k/WSPIN/ite1"]'