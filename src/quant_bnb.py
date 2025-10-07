"""
Notes
  • Requires CUDA + bitsandbytes installed (pip install bitsandbytes>=0.43)
  • Uses HF Transformers + Accelerate device_map='auto' to place layers on GPU
"""

import os, json, math, argparse, warnings
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
from transformers import (
    AutoConfig,
    AutoTokenizer,
    GPT2ForSequenceClassification,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    set_seed,
)
import bitsandbytes as bnb 

from utils.data_utils import build_tokenizer, build_datasets, LABEL_NAMES
from utils.metrics_utils import compute_metrics, save_confusion_matrix, classification_text_report
from utils.efficiency_utils import get_device, measure_latency, pretty_size


# -------------------------
# Helpers
# -------------------------

def approx_param_bytes(model: torch.nn.Module) -> int:
    """
    Approximate parameter storage size by summing tensor element_size*numel.
    Works with bnb layers since their .weight exposes a tensor dtype (int8 or fp16/bf16).
    """
    total = 0
    for p in model.parameters():
        if p is None:  # shouldn't happen
            continue
        try:
            total += p.numel() * p.element_size()
        except Exception:
            pass
    return int(total)


# -------------------------
# Loading functions
# -------------------------

def load_bnb_model(baseline_dir: str, mode: str, compute_dtype: torch.dtype | None = None):
    """
    Load a GPT-2 classifier with bitsandbytes quantization.
    mode: 'int8' or 'nf4'
    compute_dtype controls matmul compute (bf16 preferred on recent GPUs).
    """
    if compute_dtype is None:
        compute_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

    if mode == 'int8':
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
    elif mode == 'nf4':
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
    else:
        raise ValueError("mode must be 'int8' or 'nf4'")

    tok = AutoTokenizer.from_pretrained(baseline_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.padding_side = 'right'

    model = GPT2ForSequenceClassification.from_pretrained(
        baseline_dir,
        quantization_config=bnb_config,
        device_map='auto',  # accelerate will place weights on GPU
    )
    model.config.pad_token_id = tok.pad_token_id
    model.eval()
    return model, tok


# -------------------------
# Main
# -------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline_dir', type=str, required=True)
    parser.add_argument('--model_name', type=str, default='gpt2')
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--max_length', type=int, default=256)
    parser.add_argument('--eval_batch_size', type=int, default=32)
    parser.add_argument('--subset_val', type=int, default=0)
    parser.add_argument('--subset_test', type=int, default=0)
    parser.add_argument('--output_dir', type=str, default='outputs/bnb-quantized')
    args = parser.parse_args()

    set_seed(42)
    os.makedirs(args.output_dir, exist_ok=True)

    # === Prepare datasets ===
    tokenizer_ref = build_tokenizer(args.model_name)
    subsets = {'train': 0, 'val': args.subset_val, 'test': args.subset_test}
    _, val_ds, test_ds, collator = build_datasets(
        tokenizer_ref, args.max_length, val_size=2000, subsets=subsets
    )

    results: Dict[str, Any] = {}

    for mode in ['int8', 'nf4']:
        print(f"\n=== Loading bitsandbytes model: {mode} ===")
        model, tok = load_bnb_model(args.baseline_dir, mode)
        model.eval()
        for p in model.parameters():
            p.requires_grad = False

        # bitsandbytes models are already on their proper device (device_map='auto')
        try:
            # Most quantized models have their first parameter already placed correctly
            device = next(model.parameters()).device
        except Exception:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # --- helper for batched inference ---
        def run_inference(dataset):
            all_logits, all_labels = [], []
            for i in range(0, len(dataset), args.eval_batch_size):
                batch = dataset[i:i + args.eval_batch_size]
                batch = collator(batch)
                batch = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
                with torch.no_grad():
                    outputs = model(**batch)
                    logits = outputs.logits.detach().cpu()
                all_logits.append(logits)
                all_labels.append(batch["labels"].cpu())
            return torch.cat(all_logits), torch.cat(all_labels)

        # === Evaluate on val + test ===
        print("Running evaluation ...")
        val_logits, val_labels = run_inference(val_ds)
        test_logits, test_labels = run_inference(test_ds)
        val_preds = val_logits.argmax(dim=-1)
        test_preds = test_logits.argmax(dim=-1)

        val_metrics = compute_metrics((val_logits.numpy(), val_labels.numpy()))
        test_metrics = compute_metrics((test_logits.numpy(), test_labels.numpy()))

        # === Confusion matrix & report ===
        cm_path = os.path.join('reports/figs', f'confusion_matrix_bnb_{mode}.png')
        save_confusion_matrix(
            test_labels.numpy(), test_preds.numpy(), LABEL_NAMES,
            cm_path, title=f'bitsandbytes {mode.upper()} — GPT-2 on AG News'
        )
        classification_text_report(
            test_labels.numpy(), test_preds.numpy(), LABEL_NAMES,
            os.path.join(args.output_dir, f'classification_report_{mode}.txt')
        )

        # === Efficiency ===
        param_bytes = approx_param_bytes(model)
        peak_mem = None
        try:
            torch.cuda.reset_peak_memory_stats()
            _ = model(**{k: v.to(device) for k, v in collator([test_ds[0]]).items()})
            peak_mem = torch.cuda.max_memory_allocated()
        except Exception:
            pass

        avg_ms = measure_latency(
            model, test_ds, collator,
            device='cuda', batch_size=1,
            warmup_steps=20, measure_steps=100
        )

        # === Save quantized model ===
        save_dir = os.path.join(args.output_dir, f"gpt2-{mode}")
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        try:
            model.save_pretrained(save_dir, safe_serialization=True)
            tok.save_pretrained(save_dir)
        except Exception as e:
            warnings.warn(f"Could not save {mode} model: {e}")

        # === Store results ===
        results[mode] = {
            "val_metrics": {k: float(v) for k, v in val_metrics.items()
                            if isinstance(v, (int, float, np.floating))},
            "test_metrics": {k: float(v) for k, v in test_metrics.items()},
            "params_approx_bytes": int(param_bytes),
            "params_approx_pretty": pretty_size(int(param_bytes)),
            "peak_cuda_allocated_bytes": int(peak_mem) if peak_mem else None,
            "peak_cuda_allocated_pretty": pretty_size(int(peak_mem)) if peak_mem else None,
            "latency_ms_per_sample_b1": float(avg_ms) if not math.isnan(avg_ms) else None,
            "confusion_matrix_png": cm_path,
            "save_dir": save_dir,
        }

        print(
            f"Finished {mode}: "
            f"accuracy={results[mode]['test_metrics']['accuracy']:.4f}, "
            f"size≈{results[mode]['params_approx_pretty']}"
        )

    # === Write combined results ===
    out_json = os.path.join(args.output_dir, 'metrics_quant_bnb.json')
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2)

    print('\n==== bitsandbytes Quantization Summary ====')
    print(json.dumps(results, indent=2))
    print('\nSaved results to:', out_json)


if __name__ == '__main__':
    main()
