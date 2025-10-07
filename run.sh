#!/bin/zsh
# ============================================================
# Project: GPT-2 Post-Training Quantization on AG News
# Author:  Amol Vijayachandran
# Purpose: Unified runner for all three parts of the assignment
# ============================================================

set -e  # Exit immediately on error

# Default directories
SRC_DIR="src"
OUT_DIR="outputs"
FIG_DIR="reports/figs"

# Create output dirs if missing
mkdir -p "$OUT_DIR" "$FIG_DIR"

# Parse command
MODE=$1
shift  # shift arguments so $@ contains only remaining args

case "$MODE" in
  "baseline")
    echo "🚀 Running Part 1 — Full Fine-tuning Baseline (GPT-2 Small)"
    python $SRC_DIR/train_baseline.py \
      --device auto \
      --epochs 3 \
      --max_length 256 \
      --train_batch_size 8 \
      --eval_batch_size 32 \
      --output_dir $OUT_DIR/baseline-gpt2-full \
      "$@"
    ;;

  "dev")
    echo "🧪 Running quick dev sanity check (subset data)"
    python $SRC_DIR/train_baseline.py \
      --device auto \
      --epochs 1 \
      --max_length 128 \
      --train_batch_size 4 \
      --eval_batch_size 8 \
      --subset_train 2000 \
      --subset_val 500 \
      --subset_test 1000 \
      --output_dir $OUT_DIR/dev-baseline \
      "$@"
    ;;

  "ptq")
    echo "⚙️ Running Part 2 — Post-Training Quantization from Scratch (INT8)"
    python $SRC_DIR/ptq_scratch.py \
      --baseline_dir $OUT_DIR/baseline-gpt2-full \
      --model_name gpt2 \
      --device auto \
      --max_length 256 \
      --eval_batch_size 32 \
      --quant_mode asymmetric \
      --output_dir $OUT_DIR/ptq-scratch-int8 \
      "$@"
    ;;

  "bnb")
    echo "📦 Running Part 3 — Library-Based Quantization (bitsandbytes 8-bit / 4-bit)"
    python $SRC_DIR/quant_bnb.py \
      --baseline_dir $OUT_DIR/baseline-gpt2-full \
      --model_name gpt2 \
      --device auto \
      --max_length 256 \
      --eval_batch_size 32 \
      --output_dir $OUT_DIR/bnb-quantized \
      "$@"
    ;;

  *)
    echo "Usage: ./run.sh [baseline|dev|ptq|bnb]"
    echo
    echo "Examples:"
    echo "  ./run.sh baseline"
    echo "  ./run.sh dev"
    echo "  ./run.sh ptq"
    echo "  ./run.sh bnb"
    ;;
esac