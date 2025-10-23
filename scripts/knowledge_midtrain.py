"""Midtrain on FineWiki to inject factual knowledge before conversation training."""

import os
import time
from contextlib import nullcontext

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import wandb

from nanochat.checkpoint_manager import load_model, save_checkpoint
from nanochat.common import (
    DummyWandb,
    autodetect_device_type,
    compute_cleanup,
    compute_init,
    get_base_dir,
    print0,
)
from nanochat.dataloader import tokenizing_distributed_data_loader
from nanochat.dataset import parquets_iter_batched_finewiki
from nanochat.loss_eval import evaluate_bpb
from nanochat.tokenizer import get_token_bytes

# -----------------------------------------------------------------------------
# User settings
run = "dummy"
device_type = ""
num_iterations = 2800
max_seq_len = 2048
device_batch_size = 16
unembedding_lr = 0.004
embedding_lr = 0.2
matrix_lr = 0.02
init_lr_frac = 1.0
weight_decay = 0.0
eval_every = 200
eval_tokens = 10 * 524288
total_batch_size = 524288
dry_run = 0

config_keys = [
    k
    for k, v in globals().items()
    if not k.startswith("_") and isinstance(v, (int, float, bool, str))
]
exec(open(os.path.join("nanochat", "configurator.py")).read())
user_config = {k: globals()[k] for k in config_keys}

# -----------------------------------------------------------------------------
# Compute init
device_type = autodetect_device_type() if device_type == "" else device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0
autocast_ctx = (
    torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16)
    if device_type == "cuda"
    else nullcontext()
)
get_max_memory = torch.cuda.max_memory_allocated if device_type == "cuda" else lambda: 0

# wandb logging init
use_dummy_wandb = run == "dummy" or not master_process
wandb_run = (
    DummyWandb()
    if use_dummy_wandb
    else wandb.init(project="nanochat-knowledge", name=run, config=user_config)
)

# Load base checkpoint
model, tokenizer, meta = load_model("base", device, phase="train")
orig_model = model
model = torch.compile(model, dynamic=False)
num_params = sum(p.numel() for p in model.parameters())
print0(f"Loaded base checkpoint with {num_params:,} parameters")

num_flops_per_token = model.estimate_flops()
tokens_per_fwdbwd = device_batch_size * max_seq_len
world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size
assert total_batch_size % world_tokens_per_fwdbwd == 0
grad_accum_steps = total_batch_size // world_tokens_per_fwdbwd

# Optimizers (re-init fresh for midtraining)
optimizers = model.setup_optimizers(
    unembedding_lr=unembedding_lr,
    embedding_lr=embedding_lr,
    matrix_lr=matrix_lr,
    weight_decay=weight_decay,
)
adamw_optimizer, muon_optimizer = optimizers

for opt in optimizers:
    for group in opt.param_groups:
        group["lr"] = group["lr"] * init_lr_frac
        group["initial_lr"] = group["lr"]

# Data loaders (FineWiki)
train_loader = tokenizing_distributed_data_loader(
    device_batch_size,
    max_seq_len,
    split="train",
    parquets_iter_fn=parquets_iter_batched_finewiki,
    device=device,
)
build_val_loader = lambda: tokenizing_distributed_data_loader(
    device_batch_size,
    max_seq_len,
    split="train",
    parquets_iter_fn=parquets_iter_batched_finewiki,
    device=device,
)
x, y = next(train_loader)

token_bytes = get_token_bytes(device=device)

# Training loop
min_val_bpb = float("inf")
smooth_train_loss = 0
ema_beta = 0.9
total_training_time = 0

for step in range(num_iterations + 1):
    last_step = step == num_iterations
    flops_so_far = num_flops_per_token * total_batch_size * step

    if eval_every > 0 and (last_step or step % eval_every == 0):
        model.eval()
        val_loader = build_val_loader()
        eval_steps = eval_tokens // (device_batch_size * max_seq_len * ddp_world_size)
        with autocast_ctx:
            val_bpb = evaluate_bpb(model, val_loader, eval_steps, token_bytes)
        print0(f"Step {step:05d} | Validation bpb: {val_bpb:.4f}")
        if val_bpb < min_val_bpb:
            min_val_bpb = val_bpb
        wandb_run.log(
            {
                "step": step,
                "val/bpb": val_bpb,
                "total_training_flops": flops_so_far,
                "total_training_time": total_training_time,
            }
        )
        model.train()

    if master_process and last_step and not dry_run:
        base_dir = get_base_dir()
        model_tag = meta.get("user_config", {}).get("model_tag")
        if not model_tag:
            model_tag = f"d{model.config.n_layer}"
        checkpoint_dir = os.path.join(base_dir, "knowledge_checkpoints", model_tag)
        os.makedirs(checkpoint_dir, exist_ok=True)
        save_checkpoint(
            checkpoint_dir,
            step,
            orig_model.state_dict(),
            [opt.state_dict() for opt in optimizers],
            {
                "step": step,
                "val_bpb": val_bpb,
                "model_config": meta["model_config"],
                "user_config": user_config,
                "device_batch_size": device_batch_size,
                "max_seq_len": max_seq_len,
            },
        )

    if last_step:
        break

    if device_type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for micro_step in range(grad_accum_steps):
        with autocast_ctx:
            loss = model(x, y)
        train_loss = loss.detach()
        loss = loss / grad_accum_steps
        loss.backward()
        x, y = next(train_loader)

    # restore learning rates
    for opt in optimizers:
        for group in opt.param_groups:
            group["lr"] = group["initial_lr"]

    for opt in optimizers:
        opt.step()
    model.zero_grad(set_to_none=True)
    if device_type == "cuda":
        torch.cuda.synchronize()
    dt = time.time() - t0

    smooth_train_loss = (
        ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss.item()
    )
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta ** (step + 1))
    if step > 10:
        total_training_time += dt

    wandb_run.log(
        {
            "step": step,
            "train/loss": debiased_smooth_loss,
            "train/dt": dt,
        }
    )

# cleanup
wandb_run.finish()
compute_cleanup()
