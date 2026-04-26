#! /bin/bash
model_name_1="model_hub/Qwen1.5-1.8B/"
model_name_2="model_hub/gpt2_120M"
input_dir="data/Ultrachat200k/drapt/"
output_dir="data/Ultrachat200k/drapt/"
model1_template="normal"
model2_template="normal"
batch_size=8
num_gpus=1
force_sequential=false  # Set to true if multiprocessing causes issues
# Create output directory if it doesn't exist
# mkdir -p $output_dir
# Run the parallel processing script
python token_weight_estimation.py   --model_name_1 $model_name_1   --model_name_2 $model_name_2   --model1_template $model1_template   --model2_template $model2_template   --input_dir $input_dir   --output_dir $output_dir   --batch_size $batch_size   --num_gpus $num_gpus   $(if $force_sequential; then echo "--force_sequential"; fi) 