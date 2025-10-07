### Links

- Report 1:
- Report 2:

## Setup

### MPS

```bash
conda create --name ANLP_A2 python=3.12
conda activate ANLP_A2

pip install "torch>=2.3" "transformers>=4.44" "datasets>=2.19" \
            "accelerate>=0.33" "evaluate>=0.4" scikit-learn \
            matplotlib tqdm

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("MPS available:", torch.backends.mps.is_available())
print("CUDA available:", torch.cuda.is_available())
```

### CUDA

```bash
conda create --name ANLP_A2 python=3.12
conda activate ANLP_A2
```
install PyTorch matching your CUDA (pick one of the two lines)

```bash
# CUDA 12.1:
pip install --index-url https://download.pytorch.org/whl/cu121 "torch>=2.3" torchvision torchaudio
# CUDA 11.8:
pip install --index-url https://download.pytorch.org/whl/cu118 "torch>=2.3" torchvision torchaudio
```

```bash
pip install "transformers>=4.44" "datasets>=2.19" "accelerate>=0.33" "evaluate>=0.4" scikit-learn matplotlib tqdm torch torchvision "bitsandbytes>=0.43"

python - <<'PY'
import torch, bitsandbytes as bnb
print("torch:", torch.__version__)
print("CUDA:", torch.version.cuda, "| available:", torch.cuda.is_available())
print("GPUs:", torch.cuda.device_count())
print("bnb:", bnb.__version__)
```

## Run

```bash
python src/train_baseline.py \
  --device auto \
  --epochs 3 \
  --max_length 256 \
  --train_batch_size 8 \
  --eval_batch_size 32 \
  --output_dir outputs/baseline-gpt2-full
```