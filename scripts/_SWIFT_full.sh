#! /bin/bash
set -euo pipefail

MODEL="model_hub/Qwen1.5-1.8B/SFT"
INPUT="data/Ultrachat200k/SFT/trainSFT.jsonl"
OUTDIR="data/Ultrachat200k/WSPIN/ite0/train"
BATCH=8
MAX_NEW=512
FRAC_LEN=1000000
FRAC=0

#### Gen sample
# loop over data_frac = 0..70
python generate_vllm.py \
  --model       "$MODEL" \
  --input_dir   "$INPUT" \
  --output_dir  "$OUTDIR" \
  --max_new_tokens $MAX_NEW \
  --data_frac   $FRAC \
  --frac_len    $FRAC_LEN \
  --split       train


#### Compute Important weight
model_name_1="model_hub/zephyr-7b-sft-full/"
model_name_2="model_hub/Qwen1.5-1.8B/SFT"
input_dir="data/Ultrachat200k/WSPIN/ite0"
output_dir="data/Ultrachat200k/WSPIN/ite0"
model1_template="normal"
model2_template="normal"
batch_size=8
num_gpus=1
max_length=1024
max_prompt_length=512
force_sequential=false  # Set to true if multiprocessing causes issues
# Create output directory if it doesn't exist
# mkdir -p $output_dir
# Run the parallel processing script
python token_weight_estimation.py   --model_name_1 $model_name_1   --model_name_2 $model_name_2   --model1_template $model1_template   --model2_template $model2_template   --input_dir $input_dir   --output_dir $output_dir  --max_length $max_length  --max_prompt_length $max_prompt_length   --batch_size $batch_size   --num_gpus $num_gpus   $(if $force_sequential; then echo "--force_sequential"; fi) 

#### Train WSPIN
python -u train.py \
  model=qwen \
  model.name_or_path=model_hub/Qwen1.5-1.8B/SFT/ \
  loss=tisdpo \
  base_data_dir=data \
  ckpt_dir=model_hub/Qwen1.5-1.8B/WSPIN/ite0/ \
  datasets='["Ultrachat200k/WSPIN/ite0"]'


MODEL="model_hub/Qwen1.5-1.8B/WSPIN/ite0"
INPUT="data/Ultrachat200k/SFT/trainSFT.jsonl"
OUTDIR="data/Ultrachat200k/WSPIN/ite1/train"
BATCH=8
MAX_NEW=512
FRAC_LEN=1000000
FRAC=0
# loop over data_frac = 0..70
python generate_vllm.py \
  --model       "$MODEL" \
  --input_dir   "$INPUT" \
  --output_dir  "$OUTDIR" \
  --max_new_tokens $MAX_NEW \
  --data_frac   $FRAC \
  --frac_len    $FRAC_LEN \
  --split       train


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
  ckpt_dir=model_hub/Qwen1.5-1.8B/WSPIN/ite1/ \
  datasets='["Ultrachat200k/WSPIN/ite0","Ultrachat200k/WSPIN/ite1"]'



MODEL="model_hub/Qwen1.5-1.8B/WSPIN/ite1"
INPUT="data/Ultrachat200k/SFT/trainSFT.jsonl"
OUTDIR="data/Ultrachat200k/WSPIN/ite2/train"
BATCH=8
MAX_NEW=512
FRAC_LEN=1000000
FRAC=0
# loop over data_frac = 0..70

python generate_vllm.py \
  --model       "$MODEL" \
  --input_dir   "$INPUT" \
  --output_dir  "$OUTDIR" \
  --max_new_tokens $MAX_NEW \
  --data_frac   $FRAC \
  --frac_len    $FRAC_LEN \
  --split       train



model_name_1="model_hub/zephyr-7b-sft-full"
model_name_2="model_hub/Qwen1.5-1.8B/WSPIN/ite1"
input_dir="data/Ultrachat200k/WSPIN/ite2"
output_dir="data/Ultrachat200k/WSPIN/ite2"
model1_template="normal"
model2_template="normal"
batch_size=8
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
  model.name_or_path=model_hub/Qwen1.5-1.8B/WSPIN/ite1 \
  loss=tisdpo \
  base_data_dir=data \
  ckpt_dir=model_hub/Qwen1.5-1.8B/WSPIN/ite2/ \
  datasets='["Ultrachat200k/WSPIN/ite1","Ultrachat200k/WSPIN/ite2"]'



MODEL="model_hub/Qwen1.5-1.8B/WSPIN/ite2"
INPUT="data/Ultrachat200k/SFT/trainSFT.jsonl"
OUTDIR="data/Ultrachat200k/WSPIN/ite3/train"
BATCH=8
MAX_NEW=512
FRAC_LEN=1000000
FRAC=0
# loop over data_frac = 0..70
python generate_vllm.py \
  --model       "$MODEL" \
  --input_dir   "$INPUT" \
  --output_dir  "$OUTDIR" \
  --max_new_tokens $MAX_NEW \
  --data_frac   $FRAC \
  --frac_len    $FRAC_LEN \
  --split       train



model_name_1="model_hub/zephyr-7b-sft-full"
model_name_2="model_hub/Qwen1.5-1.8B/WSPIN/ite2"
input_dir="data/Ultrachat200k/WSPIN/ite3"
output_dir="data/Ultrachat200k/WSPIN/ite3"
model1_template="normal"
model2_template="normal"
batch_size=8
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
  model.name_or_path=model_hub/Qwen1.5-1.8B/WSPIN/ite2 \
  loss=tisdpo \
  base_data_dir=data \
  ckpt_dir=model_hub/Qwen1.5-1.8B/WSPIN/ite3/ \
  datasets='["Ultrachat200k/WSPIN/ite2","Ultrachat200k/WSPIN/ite3"]'
