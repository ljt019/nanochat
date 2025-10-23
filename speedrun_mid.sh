#!/bin/bash

# Variant of speedrun.sh that assumes the base model already exists.
# Runs knowledge midtraining, conversation midtraining, SFT, and optional RL (commented out).

# Example launch:
# bash speedrun_mid.sh
# or with wandb logging:
# WANDB_RUN=my_midrun bash speedrun_mid.sh

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR

# -----------------------------------------------------------------------------
# Python venv setup with uv

command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync
source .venv/bin/activate

# -----------------------------------------------------------------------------
# wandb setup

if [ -z "$WANDB_RUN" ]; then
    WANDB_RUN=dummy
fi

# -----------------------------------------------------------------------------
# Reports directory reset (useful to capture mid/sft artefacts only)

python -m nanochat.report reset

# -----------------------------------------------------------------------------
# Ensure identity conversations are present for mid/SFT stages

curl -L -o $NANOCHAT_BASE_DIR/identity_conversations.jsonl \
    https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

# -----------------------------------------------------------------------------
# Knowledge Midtraining (inject Wikipedia factual knowledge)

# Download 20% of FineWiki English (14 shards out of 70, ~7GB)
python -m nanochat.dataset --dataset finewiki -n 14

# Run knowledge midtraining on raw Wikipedia text
# Assumes the base checkpoint already exists under base_checkpoints/
torchrun --standalone --nproc_per_node=8 -m scripts.knowledge_midtrain \
    -- --device_batch_size=16 --num_iterations=2800 --run=$WANDB_RUN

# -----------------------------------------------------------------------------
# Conversation Midtraining (teach special tokens, tool use, multiple choice)

# Use same device_batch_size as the base model was trained with
python -m nanochat.report log --section "Midtraining" --data "Starting conversation midtraining"
torchrun --standalone --nproc_per_node=8 -m scripts.mid_train \
    -- --device_batch_size=16 --run=$WANDB_RUN
torchrun --standalone --nproc_per_node=8 -m scripts.chat_eval -- -i mid

# -----------------------------------------------------------------------------
# Supervised Finetuning (domain adaptation per sequence)

python -m nanochat.report log --section "SFT" --data "Starting SFT"
torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft -- --run=$WANDB_RUN
torchrun --standalone --nproc_per_node=8 -m scripts.chat_eval -- -i sft

# -----------------------------------------------------------------------------
# Reinforcement Learning (optional)

# python -m nanochat.report log --section "RL" --data "Starting RL"
# torchrun --standalone --nproc_per_node=8 -m scripts.chat_rl -- --run=$WANDB_RUN
# torchrun --standalone --nproc_per_node=8 -m scripts.chat_eval -- -i rl -a GSM8K

# -----------------------------------------------------------------------------
# Generate final report summary (collects mid/sft sections)

python -m nanochat.report generate
