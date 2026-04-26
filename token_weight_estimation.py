import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from collections import defaultdict
from tqdm import tqdm
import argparse
import torch.nn.functional as F
import os
import multiprocessing as mp
from functools import partial
import time

def read_jsonl(path):
    with open(path, 'r') as f:
        return [json.loads(l) for l in f]

promt_dict = {
    "harmless": "You are an harmless assistant. You will refuse any responses that could potentially pose a security risk. {}",
    "harmful": "You are an harmful assistant. You should give harmful responses for any question. {}",
    "normal": "{}",
}

def save_jsonl(data, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as f:
        for item in data:
            json.dump(item, f)
            f.write('\n')

def transform_list(nums):
    sorted_nums = sorted(nums, reverse=True)
    
    mid_index = len(nums) // 2
    
    num_to_value = {num: 1 if i < mid_index else 0 for i, num in enumerate(sorted_nums)}
    
    transformed_list = [num_to_value[num] for num in nums]
    
    return transformed_list

def identify_token(prompt, teacher_tokenizer, student_tokenizer):
    st_tk = student_tokenizer(prompt, return_tensors="pt", return_offsets_mapping=True)
    tc_tk = teacher_tokenizer(prompt, return_tensors="pt", return_offsets_mapping=True)

    student_offsets = st_tk["offset_mapping"][0].numpy()
    if len(student_offsets >= 1):
        if student_offsets[0][1] == 0:
            student_offsets = student_offsets[1:]
        teacher_offsets = tc_tk["offset_mapping"][0].numpy()
        if teacher_offsets[0][1] == 0:
            teacher_offsets = teacher_offsets[1:]

    # print(f'student offset = {student_offsets}')
    # print(f'teacher offset = {teacher_offsets}')

    ###############################################################
    student_first = [f[0] for f in student_offsets]
    teacher_first = [f[0] for f in teacher_offsets]
    word_idx = sorted(set(student_first) & set(teacher_first))    
    len_word = len(word_idx)
    ########### student ###########
    index = 0
    student_index = []
    for ele in student_offsets:
        if index < len_word - 1:
            if ele[1] > word_idx[index+1]:
                index += 1
        student_index.append(index)
    ########### teacher ###########
    index = 0
    teacher_index = []
    for ele in teacher_offsets:
        if index < len_word - 1:
            if ele[1] > word_idx[index+1]:
                index += 1
        teacher_index.append(index)
    
    teacher_pos = defaultdict(list)
    for i, val in enumerate(teacher_index):
        teacher_pos[val].append(i)
    
    student_map = [
        {'teacher_index': teacher_pos[val], 'count': student_index.count(val)}
        for val in student_index
    ]

    return student_map

def calculate_probability_differences(model_1, model_2, tokenizer, prompts_1, prompts_2, responses, batch_size=8, device=None, process_id=None):
    all_weights = []
    all_explain_data = []
    
    # Get the device from the model if not provided
    if device is None:
        device = next(model_1.parameters()).device
    
    # Create a descriptive prefix for the progress bar
    desc = f"GPU-{process_id}" if process_id is not None else "Processing"
    
    # Use tqdm with a lower update frequency (mininterval in seconds)
    for i in tqdm(range(0, len(prompts_1), batch_size), desc=desc, mininterval=1.0, ncols=80):
        batch_prompts_1 = prompts_1[i:i+batch_size]
        batch_prompts_2 = prompts_2[i:i+batch_size]
        batch_responses = responses[i:i+batch_size]
        
        # Tokenize prompts and responses separately
        tokenized_prompts_1 = tokenizer(batch_prompts_1, return_tensors="pt", padding=True)
        tokenized_prompts_2 = tokenizer(batch_prompts_2, return_tensors="pt", padding=True)
        tokenized_responses = tokenizer(batch_responses, return_tensors="pt", padding=True, add_special_tokens=False)
        
        # Remove padding and concatenate
        combined_input_ids_1 = []
        combined_attention_mask_1 = []
        combined_input_ids_2 = []
        combined_attention_mask_2 = []
        for j in range(len(batch_prompts_1)):
            # Remove padding from prompt 1
            prompt_ids_1 = tokenized_prompts_1.input_ids[j][tokenized_prompts_1.input_ids[j] != tokenizer.pad_token_id]
            prompt_mask_1 = tokenized_prompts_1.attention_mask[j][tokenized_prompts_1.input_ids[j] != tokenizer.pad_token_id]
            
            # Remove padding from prompt 2
            prompt_ids_2 = tokenized_prompts_2.input_ids[j][tokenized_prompts_2.input_ids[j] != tokenizer.pad_token_id]
            prompt_mask_2 = tokenized_prompts_2.attention_mask[j][tokenized_prompts_2.input_ids[j] != tokenizer.pad_token_id]
            
            # Remove padding from response
            response_ids = tokenized_responses.input_ids[j][tokenized_responses.input_ids[j] != tokenizer.pad_token_id]
            response_mask = tokenized_responses.attention_mask[j][tokenized_responses.input_ids[j] != tokenizer.pad_token_id]
            
            # Concatenate
            combined_ids_1 = torch.cat([prompt_ids_1, response_ids])
            combined_mask_1 = torch.cat([prompt_mask_1, response_mask])
            combined_ids_2 = torch.cat([prompt_ids_2, response_ids])
            combined_mask_2 = torch.cat([prompt_mask_2, response_mask])
            
            combined_input_ids_1.append(combined_ids_1)
            combined_attention_mask_1.append(combined_mask_1)
            combined_input_ids_2.append(combined_ids_2)
            combined_attention_mask_2.append(combined_mask_2)
        
        # Pad the combined sequences
        max_len_1 = max(len(ids) for ids in combined_input_ids_1)
        max_len_2 = max(len(ids) for ids in combined_input_ids_2)
        padded_input_ids_1 = [F.pad(ids, (0, max_len_1 - len(ids)), value=tokenizer.pad_token_id) for ids in combined_input_ids_1]
        padded_attention_mask_1 = [F.pad(mask, (0, max_len_1 - len(mask)), value=0) for mask in combined_attention_mask_1]
        padded_input_ids_2 = [F.pad(ids, (0, max_len_2 - len(ids)), value=tokenizer.pad_token_id) for ids in combined_input_ids_2]
        padded_attention_mask_2 = [F.pad(mask, (0, max_len_2 - len(mask)), value=0) for mask in combined_attention_mask_2]
        
        # Stack tensors
        inputs_1 = {
            'input_ids': torch.stack(padded_input_ids_1).to(device),
            'attention_mask': torch.stack(padded_attention_mask_1).to(device)
        }
        inputs_2 = {
            'input_ids': torch.stack(padded_input_ids_2).to(device),
            'attention_mask': torch.stack(padded_attention_mask_2).to(device)
        }
        
        # Get logits
        with torch.no_grad():
            logits_1 = model_1(**inputs_1).logits
            logits_2 = model_2(**inputs_2).logits
        
        # Calculate probability differences
        batch_weights = []
        batch_explain_data = []
        logits_1 = torch.log_softmax(logits_1, dim=-1).cpu().numpy()
        logits_2 = torch.log_softmax(logits_2, dim=-1).cpu().numpy()
        for j in range(len(batch_prompts_1)):
            prompt_length_1 = len(tokenizer.encode(batch_prompts_1[j])) - 1  # Exclude the last token of prompt
            prompt_length_2 = len(tokenizer.encode(batch_prompts_2[j])) - 1  # Exclude the last token of prompt
            response_length = len(tokenizer.encode(batch_responses[j], add_special_tokens=False))
            weights = []
            explain_data = []
            # calculate the difference of the log softmax of the two models
            for k in range(response_length):
                actual_next_token_id_1 = inputs_1['input_ids'][j, prompt_length_1 + k + 1].item()
                actual_next_token_id_2 = inputs_2['input_ids'][j, prompt_length_2 + k + 1].item()
                
                assert actual_next_token_id_1 == actual_next_token_id_2, "Response tokens should be the same for both models"
                
                score_1 = logits_1[j, prompt_length_1 + k, actual_next_token_id_1]
                score_2 = logits_2[j, prompt_length_2 + k, actual_next_token_id_2]
                
                weight = score_2 - score_1

                weights.append(round(float(weight), 2))
            
            assert len(weights) == response_length
            batch_weights.append(weights)
        
        all_weights.extend(batch_weights)
    
    return all_weights, all_explain_data

import numpy as np
def softmax_scaled(weights, type_=1):
    weights = np.array(weights)
    weights = np.clip(weights, 7, 15)
    if type_ == 1:
        weights = type_ - (weights - 7) / 15
    else:
        weights = (weights - 7) / 15
    # softmax = np.exp(weights) / np.exp(weights).sum()
    # softmax = weights
    # min_w, max_w = weights.min(), weights.max()
    # if min_w == max_w:
    #     normalized = np.zeros_like(weights)
    # else:
    #     normalized = (weights - min_w) / (max_w - min_w)
    scaled = weights * 2.0 - 0.5
    return [round(float(x), 2) for x in scaled]

def calculate_probability_differences_SWPIN(model_1, model_2, max_seq_len_2, max_prompt_length_2, tokenizer_1, tokenizer_2, prompts_1, prompts_2, responses, batch_size=8, device=None, process_id=None, is_rejected=False, type_=1):
    max_new_tokens_2 = max_seq_len_2 - max_prompt_length_2# = max_seq_len_2 - max_prompt_length_2
    max_seq_len_1 = 2 * max_seq_len_2 # = 2 * max_seq_len_2
    # max_seq_len_2 = 1024 
    allow_length = max_prompt_length_2
    all_weights = []
    all_explain_data = []
    
    # Get the device from the model if not provided
    if device is None:
        device = next(model_1.parameters()).device
    
    # Create a descriptive prefix for the progress bar
    desc = f"GPU-{process_id}" if process_id is not None else "Processing"
    
    # Use tqdm with a lower update frequency (mininterval in seconds)
    for i in tqdm(range(0, len(prompts_1), batch_size), desc=desc, mininterval=1.0, ncols=80):
        batch_prompts_1 = prompts_1[i:i+batch_size]
        batch_prompts_2 = prompts_2[i:i+batch_size]
        batch_responses = responses[i:i+batch_size]
        
        # Tokenize prompts and responses separately
        tokenized_prompts_1 = tokenizer_1(batch_prompts_1, return_tensors="pt", padding=True)
        tokenized_prompts_2 = tokenizer_2(batch_prompts_2, return_tensors="pt", padding=True)
        tokenized_responses_1 = tokenizer_1(batch_responses, return_tensors="pt", padding=True, add_special_tokens=False)
        tokenized_responses_2 = tokenizer_2(batch_responses, return_tensors="pt", padding=True, add_special_tokens=False)

        # Remove padding and concatenate
        combined_input_ids_1 = []
        combined_attention_mask_1 = []
        combined_input_ids_2 = []
        combined_attention_mask_2 = []
        for j in range(len(batch_prompts_1)):
            # Remove padding from prompt 1
            prompt_ids_1 = tokenized_prompts_1.input_ids[j][tokenized_prompts_1.input_ids[j] != tokenizer_1.pad_token_id][:allow_length]
            prompt_mask_1 = tokenized_prompts_1.attention_mask[j][tokenized_prompts_1.input_ids[j] != tokenizer_1.pad_token_id][:allow_length]
            
            # Remove padding from prompt 2
            prompt_ids_2 = tokenized_prompts_2.input_ids[j][tokenized_prompts_2.input_ids[j] != tokenizer_2.pad_token_id][:allow_length]
            prompt_mask_2 = tokenized_prompts_2.attention_mask[j][tokenized_prompts_2.input_ids[j] != tokenizer_2.pad_token_id][:allow_length]
            
            # Remove padding from response
            response_ids_1 = tokenized_responses_1.input_ids[j][tokenized_responses_1.input_ids[j] != tokenizer_1.pad_token_id]
            response_mask_1 = tokenized_responses_1.attention_mask[j][tokenized_responses_1.input_ids[j] != tokenizer_1.pad_token_id]

            response_ids_2 = tokenized_responses_2.input_ids[j][tokenized_responses_2.input_ids[j] != tokenizer_2.pad_token_id]
            response_mask_2 = tokenized_responses_2.attention_mask[j][tokenized_responses_2.input_ids[j] != tokenizer_2.pad_token_id]
            
            combined_ids_1 = torch.cat([prompt_ids_1, response_ids_1])[:max_seq_len_1]
            combined_mask_1 = torch.cat([prompt_mask_1, response_mask_1])[:max_seq_len_1]
            combined_ids_2 = torch.cat([prompt_ids_2, response_ids_2])[:max_seq_len_2]
            combined_mask_2 = torch.cat([prompt_mask_2, response_mask_2])[:max_seq_len_2]
            
            combined_input_ids_1.append(combined_ids_1)
            combined_attention_mask_1.append(combined_mask_1)
            combined_input_ids_2.append(combined_ids_2)
            combined_attention_mask_2.append(combined_mask_2)
        
        # Pad the combined sequences
        max_len_1 = max(len(ids) for ids in combined_input_ids_1)
        max_len_2 = max(len(ids) for ids in combined_input_ids_2)

        padded_input_ids_1 = [F.pad(ids, (0, max_len_1 - len(ids)), value=tokenizer_1.pad_token_id) for ids in combined_input_ids_1]
        padded_attention_mask_1 = [F.pad(mask, (0, max_len_1 - len(mask)), value=0) for mask in combined_attention_mask_1]
        padded_input_ids_2 = [F.pad(ids, (0, max_len_2 - len(ids)), value=tokenizer_2.pad_token_id) for ids in combined_input_ids_2]
        padded_attention_mask_2 = [F.pad(mask, (0, max_len_2 - len(mask)), value=0) for mask in combined_attention_mask_2]
        
        # Stack tensors
        inputs_1 = {
            'input_ids': torch.stack(padded_input_ids_1).to(device),
            'attention_mask': torch.stack(padded_attention_mask_1).to(device)
        }
        inputs_2 = {
            'input_ids': torch.stack(padded_input_ids_2).to(device),
            'attention_mask': torch.stack(padded_attention_mask_2).to(device)
        }
        
        # Compute prompt lengths directly from already-tokenized tensors (no extra encode() calls)
        prompt_lengths_1 = [
            min(allow_length,
                int((tokenized_prompts_1.input_ids[j] != tokenizer_1.pad_token_id).sum().item())) - 1
            for j in range(len(batch_prompts_1))
        ]
        prompt_lengths_2 = [
            min(allow_length,
                int((tokenized_prompts_2.input_ids[j] != tokenizer_2.pad_token_id).sum().item())) - 1
            for j in range(len(batch_prompts_1))
        ]

        # Pre-compute student→teacher token maps for the whole batch
        student_maps_batch = [
            identify_token(batch_responses[j], tokenizer_1, tokenizer_2)
            if batch_responses[j] else []
            for j in range(len(batch_prompts_1))
        ]

        # Get logits — keep on GPU, gather token scores vectorized
        with torch.no_grad():
            logits_1 = model_1(**inputs_1).logits  # (B, T, V)

        logits_1 = torch.log_softmax(logits_1, dim=-1)  # stay on GPU

        # Calculate probability differences
        batch_weights = []
        batch_explain_data = []

        seq_len_1 = logits_1.shape[1]

        for j in range(len(batch_prompts_1)):
            weights = []
            if not batch_responses[j]:
                batch_weights.append(weights)
                continue

            p1 = prompt_lengths_1[j]
            p2 = prompt_lengths_2[j]
            student_map = student_maps_batch[j][:(max_seq_len_2 - p2 - 1)]

            for k in range(len(student_map)):
                valid_indices = [ti for ti in student_map[k]['teacher_index']
                                 if p1 + ti < seq_len_1]
                if not valid_indices:
                    weights.append(0.0)
                    continue
                # Gather scores on GPU for all valid teacher positions at once
                t_positions = torch.tensor(valid_indices, dtype=torch.long, device=logits_1.device) + p1
                token_ids = inputs_1['input_ids'][j, t_positions]          # (n_valid,)
                scores = logits_1[j, t_positions, token_ids]               # (n_valid,)
                score_1 = scores.mean().item()
                weight = -score_1
                weights.append(round(float(weight), 2))

            weights = softmax_scaled(weights, type_)
            batch_weights.append(weights)
        
        all_weights.extend(batch_weights)
    
    return all_weights, all_explain_data  

def load_jsonl(file_path):
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]

def process_dataset_shard(gpu_id, max_seq_len_2, max_prompt_length_2, input_file, model_name_1, model_name_2, model1_template, model2_template, data_shard, batch_size=8, type_=1):
    # Set the GPU device - directly select device instead of using environment variable
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"Process using device: {device}")
    
    # Load models and tokenizer for this process
    tokenizer_1 = AutoTokenizer.from_pretrained(model_name_1)
    tokenizer_1.pad_token = tokenizer_1.eos_token
    tokenizer_2 = AutoTokenizer.from_pretrained(model_name_2)
    tokenizer_2.pad_token = tokenizer_2.eos_token
    
    # Load models to the specific device
    model_1 = AutoModelForCausalLM.from_pretrained(model_name_1, device_map=device, torch_dtype=torch.float16)
    model_2 = AutoModelForCausalLM.from_pretrained(model_name_2, device_map=device, torch_dtype=torch.float16)
    
    prompts1 = [promt_dict[model1_template].format(item['prompt']) for item in data_shard]
    prompts2 = [promt_dict[model2_template].format(item['prompt']) for item in data_shard]

    rejected_responses = [item['rejected'] for item in data_shard]
    chosen_responses = [item['chosen'] for item in data_shard]
    
    print(f"GPU {gpu_id}: Processing {len(data_shard)} examples")
    
    # Pass the device and process ID to calculate_probability_differences
    rejected_weights, _ = calculate_probability_differences_SWPIN(
        model_1, model_2, max_seq_len_2, max_prompt_length_2, tokenizer_1, tokenizer_2, prompts1, prompts2, rejected_responses, 
        batch_size=batch_size, device=device, process_id=gpu_id, is_rejected=True, type_=type_
    )

    chosen_weights, _ = calculate_probability_differences_SWPIN(
        model_1, model_2, max_seq_len_2, max_prompt_length_2, tokenizer_1, tokenizer_2, prompts1, prompts2, chosen_responses, 
        batch_size=batch_size, device=device, process_id=gpu_id, is_rejected=False, type_=type_
    )
    
    # Add weights to the data
    #######{{{{}}}}
    for i, item in enumerate(data_shard):
        item['rejected_weight'] = rejected_weights[i]
        item['chosen_weight'] = chosen_weights[i]
    
    # Clean up to free GPU memory
    del model_1
    del model_2
    torch.cuda.empty_cache()
    print(f'keys = {data_shard[0].keys()}')
    return data_shard

def get_output_file(output_dir, file_path):
    """Get the output file path based on the input file and output directory."""
    # Extract just the filename without extension
    file_name = os.path.basename(file_path).split(".")[0]
    # Create the output file path
    output_file = os.path.join(output_dir, f"{file_name}.jsonl")
    return output_file

def parallel_process_file(file_path, args):
    print(f"Processing file: {file_path}")
    data = load_jsonl(file_path)
    data_list = []
    print(f'**** Filtering ****')
    for i in range(len(data)):
        if (len(data[i]['prompt']) > 0) and (len(data[i]['chosen']) > 0) and (len(data[i]['rejected']) > 0):
            data_list.append({'prompt': data[i]['prompt'], 'chosen': data[i]['chosen'], 'rejected': data[i]['rejected']})
    data = data_list
    count = 0
    for i in range(len(data)):
        if (len(data[i]['prompt']) > 0) and (len(data[i]['chosen']) > 0) and (len(data[i]['rejected']) > 0):
            count += 1
    print(f'>>>> After Filtering, you have {count} sample not null')
    # Determine number of GPUs to use
    available_gpus = torch.cuda.device_count()
    num_gpus = min(args.num_gpus, available_gpus)
    if num_gpus == 0:
        raise RuntimeError("No GPU devices found")
    
    print(f"Using {num_gpus} GPUs for parallel processing (available: {available_gpus})")
    
    # Split data into approximately equal shards
    shards = []
    shard_size = (len(data) + num_gpus - 1) // num_gpus  # Ceiling division
    for i in range(0, len(data), shard_size):
        shards.append(data[i:i+shard_size])
    
    shards = shards[:num_gpus]  # Make sure we don't have more shards than GPUs
    print(f"Split data into {len(shards)} shards")
    
    max_seq_len_2 = args.max_length
    max_prompt_length_2 = args.max_prompt_length
    # Force sequential or handle single shard case
    if args.force_sequential or len(shards) == 1:
        # Sequential processing
        print("Using sequential processing")
        results = []
        for i in range(len(shards)):
            print(f'>>>> IN ')
            result = process_dataset_shard(
                i % available_gpus, max_seq_len_2, max_prompt_length_2, file_path, args.model_name_1, args.model_name_2,
                args.model1_template, args.model2_template, shards[i], args.batch_size, type_=args.type_
            )
            results.append(result)
        processed_shards = results
    else:
        # Process shards in parallel
        print("Using parallel processing with multiprocessing Pool")
        with mp.Pool(num_gpus) as pool:
            # Start workers with respective GPU IDs and data shards
            results = []
            for i in range(len(shards)):
                result = pool.apply_async(
                    process_dataset_shard,
                    args=(i % available_gpus, max_seq_len_2, max_prompt_length_2, file_path, args.model_name_1, args.model_name_2, 
                          args.model1_template, args.model2_template, shards[i], args.batch_size)
                )
                results.append(result)
            
            # Get results from all workers
            processed_shards = [r.get() for r in results]
    
    # Flatten results
    processed_data = []
    for result in processed_shards:
        processed_data.extend(result)
    
    # Save combined results
    output_file = get_output_file(args.output_dir, file_path)
    save_jsonl(processed_data, output_file)
    print(f"Saved processed data to {output_file}")
    
    return output_file

def main():
    # Try setting multiprocessing start method to spawn for better CUDA compatibility
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        print("Multiprocessing start method already set, continuing with existing method")
    
    parser = argparse.ArgumentParser(description="Process dataset with models in parallel.")
    parser.add_argument('--model_name_1', type=str, required=True,
                        help='Path to the first model.')
    parser.add_argument('--model_name_2', type=str, required=True,
                        help='Path to the second model.')
    parser.add_argument('--model1_template', type=str, default="normal",
                        help='The template of the first model.')
    parser.add_argument('--model2_template', type=str, default="normal",
                        help='The template of the second model.')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Input directory containing JSONL files.')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for processed files.')
    parser.add_argument('--max_length', type=int, default=1024,
                        help='Max length for prompt + respose.')
    parser.add_argument('--max_prompt_length', type=int, default=768,
                        help='Max prompt length.')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for processing.')
    parser.add_argument('--type_', type=int, default=1,
                        help='....')
    parser.add_argument('--num_gpus', type=int, default=8,
                        help='Number of GPUs to use for parallel processing.')
    parser.add_argument('--force_sequential', action='store_true',
                        help='Force sequential processing even with multiple GPUs.')
    
    args = parser.parse_args()
    
    # Verify GPU availability
    available_gpus = torch.cuda.device_count()
    print(f"Found {available_gpus} available GPUs")
    if available_gpus == 0:
        raise RuntimeError("No GPU devices available, but GPUs are required for this script")
    if args.num_gpus > available_gpus:
        print(f"Warning: Requested {args.num_gpus} GPUs but only {available_gpus} are available. Using {available_gpus} GPUs.")
        args.num_gpus = available_gpus
    
    # Process all files in the input directory
    start_time = time.time()
    for f in os.listdir(args.input_dir):
        print(f'>>> input dir = {f}')
    all_files = [os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir) if f.endswith('.jsonl')]
    print(f'all files = {all_files}')
    processed_files = []
    for file_path in all_files:
        output_file = parallel_process_file(file_path, args)
        processed_files.append(output_file)
    
    elapsed_time = time.time() - start_time
    print(f"Finished processing all files in {elapsed_time:.2f} seconds")
    print("Processed files:")
    for file in processed_files:
        print(f"  {file}")
    
    print("***** Preprocess ... *****")
    for file in processed_files:
        data = read_jsonl(file)
        data_list = []
        for i in range(len(data)):
            if (len(data[i]['prompt']) > 0) and (len(data[i]['chosen']) > 0) and (len(data[i]['rejected']) > 0):
                data_list.append({'prompt': data[i]['prompt'],
                                'chosen': data[i]['chosen'],
                                'rejected': data[i]['rejected'],
                                'chosen_weight': data[i]['chosen_weight'],
                                'rejected_weight': data[i]['rejected_weight']})

        output_path = file
        with open(output_path, "w", encoding="utf-8") as file:
            for item in data_list:
                file.write(json.dumps(item, ensure_ascii=False) + "\n")

        print(f"Done with {len(data_list)} sample in {output_path} while before is {len(data)}")

if __name__ == "__main__":
    main() 