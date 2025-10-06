t1_full: python src/train_baseline.py \
  --device auto \
  --epochs 3 \
  --max_length 256 \
  --train_batch_size 8 \
  --eval_batch_size 32 \
  --output_dir outputs/baseline-gpt2-full
