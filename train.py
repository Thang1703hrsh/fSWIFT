import torch
torch.backends.cuda.matmul.allow_tf32 = True
import torch.nn as nn
import transformers
from utils import get_local_dir, get_local_run_dir, disable_dropout, init_distributed, get_open_port, build_exp_name
import os
import hydra
import torch.multiprocessing as mp
from omegaconf import OmegaConf, DictConfig
import trainers
import wandb
import json
import socket
from typing import Optional, Set, List, Union
import resource
from transform_config import TransformConfig, get_transform_config

OmegaConf.register_new_resolver("get_local_run_dir", lambda exp_name, local_dir: get_local_run_dir(exp_name, local_dir))
OmegaConf.register_new_resolver("build_exp_name", lambda loss_name, model_name, datasets, reverse_dataset, transform: build_exp_name(loss_name, model_name, datasets, reverse_dataset, transform))


def worker_main(rank: int, world_size: int, config: DictConfig, policy_tokenizer, policy: nn.Module, reference_model: Optional[nn.Module] = None):
    """Main function for each worker process (may be only 1 for BasicTrainer/TensorParallelTrainer)."""
    if 'FSDP' in config.trainer:
        init_distributed(rank, world_size, port=config.fsdp_port)
    
    if config.debug:
        wandb.init = lambda *args, **kwargs: None
        wandb.log = lambda *args, **kwargs: None

    if rank == 0 and config.wandb.enabled:
        os.environ['WANDB_CACHE_DIR'] = get_local_dir(config.output_dir)
        wandb.init(
            entity=config.wandb.entity,
            project=config.wandb.project,
            config=OmegaConf.to_container(config),
            dir=get_local_dir(config.output_dir),
            name=config.exp_name,
        )

    # Convert transform configuration to a proper object if needed
    # if 'transform' in config and isinstance(config.transform, (dict, str)):
    transform_config = get_transform_config(config.transform)
    
    TrainerClass = getattr(trainers, config.trainer)
    print(f'Creating trainer on process {rank} with world size {world_size}')
    trainer = TrainerClass(policy_tokenizer, policy, config, config.seed, config.local_run_dir, config.ckpt_dir, reference_model=reference_model, 
                         rank=rank, world_size=world_size, transform_config=transform_config)

    trainer.train()
    trainer.save()


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(config: DictConfig):
    """Main entry point for training. Validates config, creates/initializes model(s), and kicks off worker process(es)."""
    print(f">>> batch size = {config.batch_size}")
    # Load transform configuration before resolving experiment name
    if isinstance(config.transform, str):
        # Check if it's a path to a configuration file
        if os.path.exists(config.transform) and config.transform.endswith('.yaml'):
            transform_config = TransformConfig.from_file(config.transform)
            print(f"Loaded transform configuration from {config.transform}")
        # Check if it's a preset name
        elif os.path.exists(f"config/transform/{config.transform}.yaml"):
            transform_config = TransformConfig.from_preset(config.transform)
            print(f"Loaded transform configuration preset: {config.transform}")
        # Otherwise it's just a method name
        else:
            transform_config = TransformConfig(method=config.transform)
            print(f"Using transform method: {config.transform}")
    else:
        # Using the default configuration from OmegaConf
        transform_config = config.transform
        print("Using transform configuration from config file")
    
    # Update config.transform with the full config object for experiment naming
    config.transform = transform_config.to_dict() if hasattr(transform_config, 'to_dict') else transform_config

    # Now resolve hydra references with the updated transform config
    OmegaConf.resolve(config)
    # config.max_length = 1024
    # config.max_prompt_length = 1024 - 256
    # config.model="gpt2"
    # config.model.name_or_path="model_hub/gpt2_120M/"
    if config.loss.name != 'sft':
        config.loss.beta = 0.1
    missing_keys: Set[str] = OmegaConf.missing_keys(config)
    if missing_keys:
        raise ValueError(f"Got missing keys in config:\n{missing_keys}")

    if config.eval_every % config.batch_size != 0:
        print('WARNING: eval_every must be divisible by batch_size')
        print('Setting eval_every to', config.eval_every - config.eval_every % config.batch_size)
        config.eval_every = config.eval_every - config.eval_every % config.batch_size

    if 'FSDP' in config.trainer and config.fsdp_port is None:
        free_port = get_open_port()
        print('no FSDP port specified; using open port for FSDP:', free_port)
        config.fsdp_port = free_port

    # Print transform configuration details
    method = transform_config.get('method', 'origin')
    print(f">>> Transform method: {method}")
    if method in transform_config:
        print(f"Transform parameters: {transform_config[method]}")
    print(OmegaConf.to_yaml(config))

    config_path = os.path.join(config.local_run_dir, 'config.yaml')
    with open(config_path, 'w') as f:
        OmegaConf.save(config, f)

    print('=' * 80)
    print(f'Writing to {socket.gethostname()}:{config.local_run_dir}')
    print('=' * 80)
 
    os.environ['XDG_CACHE_HOME'] = get_local_dir(config.output_dir)
    print('building policy')
    model_kwargs = {'device_map': 'balanced'} if config.trainer == 'BasicTrainer' else {}
    policy_dtype = getattr(torch, config.model.policy_dtype)
    # policy_dtype = torch.float16
    policy = transformers.AutoModelForCausalLM.from_pretrained(
        config.model.name_or_path, low_cpu_mem_usage=True, torch_dtype=policy_dtype, **model_kwargs)
    policy_tokenizer = transformers.AutoTokenizer.from_pretrained(config.model.name_or_path)
    disable_dropout(policy)
    if config.get('activation_checkpointing', False):
        policy.gradient_checkpointing_enable()
        print('Gradient checkpointing enabled for policy')

    if config.loss.name in {'dpo', 'ipo', 'tdpo', 'tisdpo', 'fswift'}:
        print('building reference model')
        reference_model_dtype = getattr(torch, config.model.reference_dtype)
        # Load reference model on CPU to save GPU VRAM; BasicTrainer moves batches to GPU when needed
        ref_kwargs = {'device_map': 'cpu'} if config.trainer == 'BasicTrainer' else model_kwargs
        reference_model = transformers.AutoModelForCausalLM.from_pretrained(
            config.model.name_or_path, low_cpu_mem_usage=True, torch_dtype=reference_model_dtype, **ref_kwargs)
        disable_dropout(reference_model)
    else:
        reference_model = None
    
    if 'FSDP' in config.trainer:
        world_size = torch.cuda.device_count()
        print('starting', world_size, 'processes for FSDP training')
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
        print(f'setting RLIMIT_NOFILE soft limit to {hard} from {soft}')
        mp.spawn(worker_main, nprocs=world_size, args=(world_size, config, policy_tokenizer, policy, reference_model), join=True)
    else:
        print('starting single-process worker')
        worker_main(0, 1, config, policy_tokenizer, policy, reference_model)

if __name__ == '__main__':
    main()