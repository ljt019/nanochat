#!/bin/bash

# Variant of speedrun.sh that assumes the base model already exists.
# Runs knowledge midtraining, conversation midtraining, SFT, and optional RL (commented out).

# Example launch (automatically logs to speedrun_mid.log):
# bash speedrun_mid.sh
# 
# Example launch with custom log file:
# LOGFILE=my_custom.log bash speedrun_mid.sh
#
# Example launch with wandb logging:
# WANDB_RUN=snout_midrun bash speedrun_mid.sh

# Setup logging (redirect all output to log file)
if [ -z "$LOGFILE" ]; then
    LOGFILE="speedrun_mid_$(date +%Y%m%d_%H%M%S).log"
fi
exec > >(tee -a "$LOGFILE") 2>&1
echo "=== Starting speedrun_mid at $(date) ==="
echo "=== Logging to: $LOGFILE ==="

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR

# -----------------------------------------------------------------------------
# Python venv setup with uv

command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
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
# Use custom identity conversations (should already be in cache directory)
# To regenerate: python dev/gen_synthetic_data.py && cp identity_conversations.jsonl $NANOCHAT_BASE_DIR/

# Verify identity file exists
if [ ! -f "$NANOCHAT_BASE_DIR/identity_conversations.jsonl" ]; then
    echo "Error: identity_conversations.jsonl not found in $NANOCHAT_BASE_DIR"
    echo "Please run: python dev/gen_synthetic_data.py && cp identity_conversations.jsonl $NANOCHAT_BASE_DIR/"
    exit 1
fi

# -----------------------------------------------------------------------------
# Knowledge Midtraining (inject Wikipedia factual knowledge)

# Download 20% of FineWiki English (3 files out of 15, ~7.5GB)
python -m nanochat.dataset --dataset finewiki -n 3

# Run knowledge midtraining on raw Wikipedia text
# Assumes the base checkpoint already exists under base_checkpoints/
torchrun --standalone --nproc_per_node=8 -m scripts.knowledge_midtrain \
    -- --device_batch_size=16 --num_iterations=600 --run=$WANDB_RUN

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
