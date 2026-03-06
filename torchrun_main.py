import os
import time
import json
import random
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.utils.data
import torch.distributed as dist
from torch.nn.utils import get_total_norm, clip_grads_with_norm_

import transformers
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from transformers import LlamaForCausalLM as HF_LlamaForCausalLM

import datasets
import datasets.distributed
import wandb

from tqdm import tqdm
from loguru import logger

from peft_pretraining import training_utils, args_utils
from peft_pretraining.dataloader import PreprocessedIterableDataset
from peft_pretraining.modeling_llama import LlamaForCausalLM

import bitsandbytes as bnb

import matplotlib.pyplot as plt
transformers.logging.set_verbosity_error()

def parse_args(args):
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_config", type=str, required=True)
    parser.add_argument("--use_hf_model", default=False, action="store_true")
    parser.add_argument("--continue_from", type=str, default=None)
    parser.add_argument("--new_wandb_on_resume", action="store_true")
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--gradient_accumulation", type=int, default=None)
    parser.add_argument("--total_batch_size", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--optimizer", default="Adam")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--scheduler", type=str, default="wsd", choices=["linear", "cosine", "cosine_restarts", "wsd"])
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)
    parser.add_argument("--activation_checkpointing", action="store_true")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_steps", type=int, default=1_000)
    parser.add_argument("--eval_every", type=int, default=2_000)
    parser.add_argument("--num_training_steps", type=int, default=10_000,
                        help="Number of **update steps** to train for. "
                             "Notice that gradient accumulation is taken into account.")
    parser.add_argument("--max_train_tokens", type=training_utils.max_train_tokens_to_number, default=None,
                        help="Number of tokens to train on. Overwrites num_training_steps. "
                             "You can use M and B suffixes, e.g. 100M or 1B.")
    parser.add_argument("--save_every", type=int, default=10_000)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--tags", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="bfloat16" if torch.cuda.is_bf16_supported() else "float32")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--grad_clipping", type=float, default=1.0)   
    parser.add_argument("--run_name", type=str, default="default")

    # prores
    parser.add_argument("--prores_warmup_steps", type=int, default=0)
    
    # disable ddp, single_gpu
    parser.add_argument("--single_gpu", default=False, action="store_true")
    
    args = parser.parse_args(args)

    args = args_utils.check_args_torchrun_main(args)
    return args


@torch.no_grad()
def evaluate_model(model, preprocess_batched, pad_idx, global_rank, world_size, device, batch_size):
    _time = time.time()
    val_data = datasets.load_dataset("allenai/c4", "en", split="validation") #DGX
    val_data = val_data.shuffle(seed=42)
    logger.info(f"Loaded validation dataset in {time.time() - _time:.2f} seconds")

    if not args.single_gpu:
        val_data = datasets.distributed.split_dataset_by_node(val_data, rank=global_rank, world_size=world_size)

    val_data_mapped = val_data.map(
        preprocess_batched,
        batched=True,
        remove_columns=["text", "timestamp", "url"],
    )
    val_data_mapped.batch = lambda batch_size: training_utils.batch_fn(val_data_mapped, batch_size)

    target_eval_tokens = 10_000_000
    evaluated_on_tokens = 0
    total_loss = torch.tensor(0.0).to(device)
    total_batches = 1
    logger.info(f"Eval set prepared in {time.time() - _time:.2f} seconds")

    for batch in val_data_mapped.batch(batch_size=batch_size):
        if evaluated_on_tokens > target_eval_tokens:
            break
        total_batches += 1

        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch["input_ids"].clone()
        labels[labels == pad_idx] = -100
        loss = model(**batch, labels=labels).loss
        total_loss += loss.detach()

        evaluated_on_tokens += (batch["input_ids"] != pad_idx).sum().item() * world_size

    total_loss = total_loss / total_batches

    # Gather losses across all GPUs
    gathered_losses = [torch.zeros_like(total_loss) for _ in range(world_size)]
    dist.all_gather(gathered_losses, total_loss)
    total_loss = sum([t.item() for t in gathered_losses]) / world_size

    return total_loss, evaluated_on_tokens


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    assert "LOCAL_RANK" in os.environ, "torchrun should set LOCAL_RANK"
    global_rank = int(os.environ['RANK'])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)

    logger.info(f"Global rank {global_rank}, local rank {local_rank}, device: {torch.cuda.current_device()}")

    dist.init_process_group(backend="nccl", rank=global_rank, world_size=world_size)

    logger.info("Process group initialized")
    device = f"cuda:{local_rank}"

    if args.total_batch_size is not None:
        if args.gradient_accumulation is None:
            assert args.total_batch_size % world_size == 0, "total_batch_size must be divisible by world_size"
            args.gradient_accumulation = args.total_batch_size // (args.batch_size * world_size)
            assert args.gradient_accumulation > 0, "gradient_accumulation must be greater than 0"

    assert args.gradient_accumulation * args.batch_size * world_size == args.total_batch_size, \
        "gradient_accumulation * batch_size * world_size must be equal to total_batch_size"

    # turn off logger
    if global_rank != 0: logger.remove()
            
    # initialize wandb without config (it is passed later)
    if global_rank == 0:
        run_id = None
        if args.continue_from and not args.new_wandb_on_resume:
            # load existing run id
            wandb_json = os.path.join(args.continue_from, 'wandb.json')
            with open(wandb_json, 'r') as f:
                run_id = json.load(f)['wandb_id']
            logger.info(f"resuming wandb run id: {run_id}")
        wandb.init(project="prores", name=args.run_name, id=run_id)
        
    logger.info(f"Using dist with rank {global_rank} (only rank 0 will log)")
    logger.info("*" * 40)
    logger.info(f"Starting training with the arguments")
    for k, v in vars(args).items():
        logger.info(f"{k:30} {v}")
    logger.info("*" * 40)

    # for longer runs, better cache c4 locally
    data = datasets.load_dataset("allenai/c4", "en", split="train", streaming=True)

    seed_for_shuffle = 0 
    
    logger.info(f"Shuffling data with seed {seed_for_shuffle}")
    data: datasets.Dataset = data.shuffle(seed=seed_for_shuffle)
    if not args.single_gpu:
        data = datasets.distributed.split_dataset_by_node(
            data, rank=global_rank, world_size=world_size,
        )

    # it doesn't matter which tokenizer we use, because we train from scratch
    # T5 tokenizer was trained on C4 and we are also training on C4, so it's a good choice
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')

    def preprocess_batched(batch):
        batch = tokenizer(
            batch["text"],
            max_length=args.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return batch

    model_config = AutoConfig.from_pretrained(args.model_config)
    if args.use_hf_model:
        model: HF_LlamaForCausalLM = AutoModelForCausalLM.from_config(model_config)
    else:
        model = LlamaForCausalLM(model_config)

    if args.activation_checkpointing:
        model.gradient_checkpointing_enable()

    global_step = 0
    update_step = 0
    tokens_seen = 0
    tokens_seen_before = 0

    if args.continue_from is not None:
        logger.info("*" * 40)
        logger.info(f"Loading model from {args.continue_from}")
        
        checkpoint_path = os.path.join(args.continue_from, "pytorch_model.bin")
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"), strict=True)
        logger.info(f"Model successfully loaded (strict=True policy)")

        if os.path.exists(os.path.join(args.continue_from, "training_state.json")):
            logger.info(f"Loading training state like global_step, update_step, and tokens_seen from {args.continue_from}")
            with open(os.path.join(args.continue_from, "training_state.json")) as f:
                _old_state = json.load(f)
            global_step = _old_state["global_step"]
            update_step = _old_state["update_step"]
            tokens_seen = _old_state["tokens_seen"]
            tokens_seen_before = _old_state["tokens_seen_before"]
            logger.info(f"global_step       : {global_step}")
            logger.info(f"update_step       : {update_step}")
            logger.info(f"tokens_seen       : {tokens_seen}")
            logger.info(f"tokens_seen_before: {tokens_seen_before}")
            logger.info(f"Will train for {args.num_training_steps - update_step} update steps")
        else:
            logger.warning(f"Did not find training state in {args.continue_from}, global step will start from zero")
        logger.info("*" * 40)

    skip_batches = global_step
    dataset = PreprocessedIterableDataset(data, tokenizer, batch_size=args.batch_size, max_length=args.max_length, skip_batches=skip_batches)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=None, num_workers=args.workers)

    if args.dtype in ["bf16", "bfloat16"]:
        model = model.to(device=device, dtype=torch.bfloat16)
    else:
        model = model.to(device=device)

    n_total_params = sum(p.numel() for p in model.parameters())
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    # Initialize wandb
    run_config = dict(vars(args))
    run_config.update({
        "max_lr": run_config.pop("lr"),  # rename lr to max_lr to avoid conflicts with scheduler
        "total_params_M": n_total_params / 1_000_000,
        "dataset": 'c4',
        "model": model_config.to_dict(),
        "world_size": world_size,
        "device": str(device),
    })

    if global_rank == 0:
        wandb.config.update(run_config, allow_val_change=True)
        wandb.save(os.path.abspath(__file__), policy="now") # save current script
        # fix tqdm visual length to 80 so that the progress bar
        # doesn't jump around when changing from external display to laptop
        pbar = tqdm(total=args.num_training_steps, desc="Update steps", ncols=80)
        pbar.update(update_step)
    
    # print params and trainable params
    logger.info(f"\n{model}\n")
    logger.info(f"Total params: {sum(p.numel() for p in model.parameters()) / 1_000_000:.2f}M")
    logger.info(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1_000_000:.2f}M")
    
    layer_wise_flag = False
    if args.optimizer.lower() == "adamw":
        no_decay = ["bias", "layernorm.weight", "embed_tokens"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=args.lr,
            betas=(0.9, 0.95),
            eps=1e-8,
            fused=True
        )
    elif args.optimizer.lower() == "adamw_embed":
        # using a large learning rate on embeddings improves performance at smaller scales (e.g. 350M)
        # although we find it insignificant at 7B parameters
        no_decay = ["bias", "layernorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if (not any(nd in n for nd in no_decay)) and "embed_tokens" not in n],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
            {
                "params": [p for n, p in model.named_parameters() if "embed_tokens" in n],
                "weight_decay": 0.0,
                "lr": 0.1 # extremely large learning rate for embedding params
            }
        ]
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=args.lr,
            betas=(0.9, 0.95),
            eps=1e-8,
            fused=True
        )
    else:
        raise ValueError(f"Optimizer {args.optimizer} not supported")

    if not layer_wise_flag:
        scheduler = training_utils.get_scheculer(
            optimizer=optimizer,
            scheduler_type=args.scheduler,
            num_training_steps=args.num_training_steps,
            warmup_steps=args.warmup_steps,
            min_lr_ratio=args.min_lr_ratio,
        )

    if args.continue_from:
        # laod optimizer and scheduler states
        optimizer_ckpt_path = os.path.join(args.continue_from, "optimizer.pt")
        if os.path.exists(optimizer_ckpt_path):
            logger.info(f"Loading optimizer and scheduler states from {optimizer_ckpt_path}")
            checkpoint = torch.load(optimizer_ckpt_path, map_location="cpu")
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
        else:
            logger.warning(f"No optimizer checkpoint found at {optimizer_ckpt_path}")
            
    if not args.single_gpu:
        model: LlamaForCausalLM = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
        )

    # global steps and others are defined above
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.pad_token_id = tokenizer.unk_token_id # this is redundant
    pad_idx = tokenizer.pad_token_id
    update_time = time.time()
    local_step = 0  # when continue_from is used, local_step != global_step

    # norm type of model
    norm_type = os.getenv("NORM_TYPE", 'pre').lower()

    # configure prores alpha before first forward
    def get_layerwise_warmup_steps(prores_warmup_steps, num_layers):
        # min_warmup_steps = max(1000, prores_warmup_steps // num_layers)
        min_warmup_steps = prores_warmup_steps // num_layers
        return np.linspace(min_warmup_steps, prores_warmup_steps, num_layers)
            
    def set_prores_alpha(model, alpha_list):
        for layer_idx in range(len(alpha_list)):
            model.module.model.layers[layer_idx].alpha = alpha_list[layer_idx]
        
    if norm_type in ['pre_prores', 'lns_prores', 'deeppost_prores', 'post_prores', 'sandwich_post_prores', 'sandwich_prores']:
        # get layerwise warmup steps
        prores_schedule = os.getenv('PRORES_SCHEDULE', 'linear')
        if prores_schedule == 'linear_equal':
            # identical warmup for all layers
            layerwise_warmup_steps = np.array([args.prores_warmup_steps]*model_config.num_hidden_layers)
        else:
            layerwise_warmup_steps = get_layerwise_warmup_steps(
                args.prores_warmup_steps, model_config.num_hidden_layers
            )
            if prores_schedule == 'linear_reverse':
                layerwise_warmup_steps = np.flip(layerwise_warmup_steps)
        
        if prores_schedule in ['linear', 'linear_equal', 'linear_reverse']:
            alpha_list = np.clip(update_step / layerwise_warmup_steps, 0.0, 1.0)
        elif prores_schedule == 'sqrt':
            alpha_list = np.clip(np.sqrt(update_step / layerwise_warmup_steps), 0.0, 1.0)
        elif prores_schedule == 'square':
            alpha_list = np.clip((update_step / layerwise_warmup_steps)**2, 0.0, 1.0)
        elif prores_schedule in ['stagewise_0', 'stagewise_L', 'stagewise_sqrt_L', 'stagewise_sqrt_l']:
            if prores_schedule == 'stagewise_0':
                stepwise_min = 0
            elif prores_schedule == 'stagewise_L':
                stepwise_min = 1 / model_config.num_hidden_layers
            elif prores_schedule == 'stagewise_sqrt_L':
                stepwise_min = np.sqrt(1 / model_config.num_hidden_layers)
            elif prores_schedule == 'stagewise_sqrt_l':
                stepwise_min = 1 / np.sqrt(list(range(1, model_config.num_hidden_layers+1)))
            alpha_list = np.clip(
                (
                    stepwise_min+(1-stepwise_min)*(update_step - np.concat(([0],layerwise_warmup_steps[:-1])))/np.diff(layerwise_warmup_steps, prepend=0)
                ),
                stepwise_min,
                1
            )
        elif prores_schedule in ['fix_L', 'fix_sqrt_L', 'fix_sqrt_l']:
            if prores_schedule == 'fix_L':
                alpha_list = [1/model_config.num_hidden_layers]*model_config.num_hidden_layers
            elif prores_schedule == 'fix_sqrt_L':
                alpha_list = [1/np.sqrt(model_config.num_hidden_layers)]*model_config.num_hidden_layers
            elif prores_schedule == 'fix_sqrt_l':
                alpha_list = 1 / np.sqrt(list(range(1, model_config.num_hidden_layers+1)))
            
        set_prores_alpha(model, alpha_list)
        # simply switch to no alpha case, if warmup finished
        if update_step > args.prores_warmup_steps and os.getenv('RESET_NORM_TYPE', 'true')=='true' and not prores_schedule.startswith('fix_'):
            os.environ['NORM_TYPE'] = norm_type.removesuffix('_prores')

    # ##############################
    # TRAINING LOOP
    # we'll never go through all the data, so no need for epochs
    # ##############################

    for batch_idx, batch in enumerate(dataloader):

        global_step += 1
        local_step += 1

        if update_step > args.num_training_steps:
            logger.info(f"Reached max number of update steps (f{args.num_training_steps}). Stopping training.")
            print(f"Rank {global_rank} stopping training.")
            break

        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch["input_ids"].clone()
        labels[labels == pad_idx] = -100
        tokens_seen += (batch["input_ids"] != pad_idx).sum().item() * world_size

        loss = model(**batch, labels=labels).loss
        scaled_loss = loss / args.gradient_accumulation
        scaled_loss.backward()

        if global_step % args.gradient_accumulation != 0:
            continue

        # The below code is only executed during the update step
        
        # calculate and log grad norm
        grad_norm = get_total_norm([p.grad for p in trainable_params if p.grad is not None], norm_type=2)
        if global_rank == 0:
            wandb.log({'grad_norm': grad_norm.item()}, step=update_step)
        # add grad clipping
        if args.grad_clipping != 0.0: 
            clip_grads_with_norm_(
                [p for p in trainable_params if p.grad is not None],
                args.grad_clipping,
                grad_norm
            )

        if global_rank == 0: pbar.update(1)
        
        if not layer_wise_flag:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        update_step += 1
        update_time = time.time() - update_time

        # update alpha for prores models
        if norm_type in ['pre_prores', 'lns_prores', 'deeppost_prores', 'post_prores', 'sandwich_post_prores', 'sandwich_prores']:
            # get layerwise warmup steps
            if update_step <= args.prores_warmup_steps:
                if prores_schedule in ['linear', 'linear_equal', 'linear_reverse']:
                    alpha_list = np.clip(update_step / layerwise_warmup_steps, 0.0, 1.0)
                elif prores_schedule == 'sqrt':
                    alpha_list = np.clip(np.sqrt(update_step / layerwise_warmup_steps), 0.0, 1.0)
                elif prores_schedule == 'square':
                    alpha_list = np.clip((update_step / layerwise_warmup_steps)**2, 0.0, 1.0)
                elif prores_schedule in ['stagewise_0', 'stagewise_L', 'stagewise_sqrt_L', 'stagewise_sqrt_l']:
                    alpha_list = np.clip(
                        (
                            stepwise_min+(1-stepwise_min)*(update_step - np.concat(([0],layerwise_warmup_steps[:-1])))/np.diff(layerwise_warmup_steps, prepend=0)
                        ),
                        stepwise_min,
                        1
                    )
                elif prores_schedule in ['fix_L', 'fix_sqrt_L', 'fix_sqrt_l']:
                    pass
                set_prores_alpha(model, alpha_list)
            elif os.getenv('RESET_NORM_TYPE', 'true')=='true':
                # we can skip alpha once all alpha's warmup to 1
                os.environ['NORM_TYPE'] = norm_type.removesuffix('_prores')
            
            # log alpha values
            if global_rank == 0:
                wandb.log(
                    {
                        f'alpha/layer_{i}': alpha_list[i] for i in range(len(alpha_list))
                    },
                    step=update_step
                )

        # save checkpoint by save_every
        if local_step > args.gradient_accumulation and update_step % args.save_every == 0 and global_rank == 0:
            current_model_directory = f"{args.save_dir}/model_{update_step}"
            logger.info(f"Saving model and optimizer to {current_model_directory}, update step {update_step}")
            os.makedirs(current_model_directory, exist_ok=True)

            # Save model weights
            model.module.save_pretrained(current_model_directory, max_shard_size='100GB')

            # Save optimizer & scheduler
            optimizer_checkpoint = {
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "update_step": update_step,
                "global_step": global_step,
                "config": run_config,
                "wandb": wandb.run.dir,
                "dtype": args.dtype,
            }
            torch.save(optimizer_checkpoint, f"{current_model_directory}/optimizer.pt")

            # Save training state
            training_state_checkpoint = {
                "global_step": global_step,
                "update_step": update_step,
                "tokens_seen": tokens_seen,
                "tokens_seen_before": tokens_seen_before,
                "update_time": update_time
            }
            with open(f"{current_model_directory}/training_state.json", "w") as f:
                json.dump(training_state_checkpoint, f, indent=4)

            # Save dataloader sampler state if supported
            if hasattr(dataloader.sampler, "state_dict"):
                torch.save(dataloader.sampler.state_dict(), f"{current_model_directory}/sampler.pt")

            # Save wandb related info
            wandb_info = {
                "wandb_id": wandb.run.id,
            }
            # with open(f"{args.save_dir}/wandb.json", "w") as f:
            with open(f"{current_model_directory}/wandb.json", "w") as f:
                json.dump(wandb_info, f, indent=2)

        # evaluation
        if update_step % args.eval_every == 0:
            logger.info(f"Performing evaluation at step {update_step}")
            total_loss, evaluated_on_tokens = evaluate_model(
                model, preprocess_batched, pad_idx, global_rank, world_size, device, args.batch_size
            )
            if global_rank == 0:
                wandb.log({
                    "final_eval_loss": total_loss,
                    "Misc/final_eval_tokens": evaluated_on_tokens,
                    },
                    step=update_step, # should not be global_step
                )
            logger.info(f"Eval loss at step {update_step}: {total_loss}")

        if not layer_wise_flag:
            lr = optimizer.param_groups[0]["lr"]
        else:
            pass
        tokens_in_update = tokens_seen - tokens_seen_before
        tokens_seen_before = tokens_seen
        batches_in_update = args.gradient_accumulation * world_size

        if global_rank == 0:
            wandb.log({
                "loss": loss.item(),
                "lr": lr,
                "Misc/update_step": update_step,
                "tokens_seen": tokens_seen,
                "Misc/throughput_tokens": tokens_in_update / update_time,
                "Misc/throughput_examples": args.total_batch_size / update_time,
                "Misc/throughput_batches": batches_in_update / update_time,
                },
                step=update_step, # should not be global_step
            )
        update_time = time.time()

    # ##############################
    # END of training loop
    # ##############################
    logger.info("Training finished")
    if global_rank == 0: pbar.close()

    current_model_directory = f"{args.save_dir}/model_{update_step}"
    if global_rank == 0 and not os.path.exists(current_model_directory):
        logger.info(f"Saving model and optimizer to {current_model_directory}, update step {update_step}")
        os.makedirs(args.save_dir, exist_ok=True)
        model.module.save_pretrained(current_model_directory)

        optimizer_checkpoint = {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "update_step": update_step,
            "global_step": global_step,
            "config": run_config,
            "wandb": wandb.run.dir,
            "dtype": args.dtype,
        }
        torch.save(optimizer_checkpoint, f"{current_model_directory}/optimizer.pt")

        training_state_checkpoint = {
            "global_step": global_step,
            "update_step": update_step,
            "tokens_seen": tokens_seen,
            "tokens_seen_before": tokens_seen_before,
            "update_time": update_time,
        }
        with open(f"{current_model_directory}/training_state.json", "w") as f:
            json.dump(training_state_checkpoint, f, indent=4)

    # Final evaluation
    logger.info("Running final evaluation")
    model.eval()
    del loss, optimizer, scheduler
    import gc; gc.collect()
    torch.cuda.empty_cache()

    total_loss, evaluated_on_tokens = evaluate_model(
        model, preprocess_batched, pad_idx, global_rank, world_size, device, args.batch_size
    )

    if global_rank == 0:
        wandb.log({
            "final_eval_loss": total_loss,
            "Misc/final_eval_tokens": evaluated_on_tokens,
            },
            step=update_step, # should not be global_step
        )
        logger.info(f"Final eval loss: {total_loss}")

    logger.info("Script finished successfully")
    print(f"Rank {global_rank} finished successfully")


if __name__ == "__main__":
    print("Starting script")
    args = parse_args(None)
    main(args)
