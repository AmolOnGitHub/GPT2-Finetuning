"""
Notes:
- We quantize modules that expose a .weight parameter (Linear/Conv1D in GPT-2).
- Bias stays in FP32. We keep original weights alongside quant params to allow restore.
- Memory footprint is reported as the *simulated* INT8 size of quantized weight tensors
plus FP32 bias, saved as a JSON summary; we also save a light checkpoint containing
the quant params so you can inspect them.
"""

import os, json, math, time, copy, argparse
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import torch
import torch.nn as nn

from transformers import GPT2ForSequenceClassification, set_seed, Trainer, TrainingArguments
from safetensors.torch import save_file
from utils.data_utils import build_tokenizer, build_datasets, LABEL_NAMES
from utils.metrics_utils import compute_metrics, save_confusion_matrix, classification_text_report
from utils.efficiency_utils import (
    get_device, measure_latency, folder_size_bytes, pretty_size
)


# =============================
# INT8 linear quant/dequant
# =============================

@torch.no_grad()
def linear_quantize_tensor(
    w: torch.Tensor,
    num_bits: int = 8,
    symmetric: bool = False,
) -> Tuple[torch.Tensor, float, int]:
    """
    Quantize a FP32 tensor to INT{num_bits} using per-tensor affine mapping.

    Returns: (q_w, scale, zero_point)
      q_w: integer tensor (torch.int8 for 8-bit)
      scale: float scale
      zero_point: int zero-point in integer domain
    """
    assert w.dtype in (torch.float32, torch.float16, torch.bfloat16, torch.float64)
    w = w.detach()

    qmin = -(2 ** (num_bits - 1))
    qmax = (2 ** (num_bits - 1)) - 1

    if symmetric:
        max_abs = torch.max(w.abs())
        scale = (max_abs / qmax).clamp(min=1e-12).item()
        zero_point = 0
        q = torch.clamp(torch.round(w / scale), qmin, qmax).to(torch.int8)

    else:
        w_min, w_max = w.min(), w.max()
        # Avoid degenerate range
        if (w_max - w_min) < 1e-12:
            scale = 1.0
            zero_point = 0
            q = torch.zeros_like(w, dtype=torch.int8)
        else:
            scale = ((w_max - w_min) / float(qmax - qmin)).item()
            zero_point = int(round(qmin - (w_min.item() / scale)))
            q = torch.round(w / scale + zero_point).clamp(qmin, qmax).to(torch.int8)

    return q, float(scale), int(zero_point)


@torch.no_grad()
def linear_dequantize_tensor(q: torch.Tensor, scale: float, zero_point: int) -> torch.Tensor:
    """
    Dequantize INT8 tensor back to FP32.
    """
    return (q.to(torch.float32) - zero_point) * scale


def simulated_int8_size_bytes(w: torch.Tensor, has_bias: bool, bias: torch.Tensor | None) -> int:
    """
    Return simulated on-disk size if weights were INT8 (bias kept FP32).
    Includes scale + zero_point per tensor (8 bytes each in JSON summary approx).
    """
    weight_int8 = w.numel()  # one byte per weight
    bias_bytes = bias.numel() * 4 if (has_bias and bias is not None) else 0
    overhead = 16  # scale+zp rough metadata
    return int(weight_int8 + bias_bytes + overhead)


# =============================
# Model traversal / quantization
# =============================

class QuantizedModuleState:
    def __init__(self, name: str, scale: float, zero_point: int, orig_shape: torch.Size, int8_bytes: int):
        self.name = name
        self.scale = float(scale)
        self.zero_point = int(zero_point)
        self.orig_shape = tuple(orig_shape)
        self.int8_bytes = int(int8_bytes)


@torch.no_grad()
def apply_weight_only_ptq(model: nn.Module, num_bits: int = 8, symmetric: bool = False) -> Dict[str, QuantizedModuleState]:
    """
    Quantize all parameters named 'weight' and replace them with dequantized FP32 tensors.
    Returns a dict of QuantizedModuleState for reporting size and inspection.
    """
    states: Dict[str, QuantizedModuleState] = {}

    for name, module in model.named_modules():
        if hasattr(module, 'weight') and isinstance(getattr(module, 'weight'), torch.nn.Parameter):
            w_param: nn.Parameter = module.weight
            if w_param is None or w_param.data is None:
                continue

            w = w_param.data
            
            # Quantize weight per-tensor
            q, s, zp = linear_quantize_tensor(w, num_bits=num_bits, symmetric=symmetric)
            
            # Replace stored FP32 weights with *dequantized* version (evaluation path)
            w_deq = linear_dequantize_tensor(q, s, zp).to(w.dtype)
            module.weight.data = w_deq
            
            # Record simulated INT8 size
            b = getattr(module, 'bias', None)
            states[f"{name}.weight"] = QuantizedModuleState(
                name=f"{name}.weight",
                scale=s,
                zero_point=zp,
                orig_shape=w.shape,
                int8_bytes=simulated_int8_size_bytes(w, has_bias=hasattr(module, 'bias'), bias=b.data if b is not None else None),
            )

    return states



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline_dir', type=str, required=True, help='Path to Part 1 fine-tuned checkpoint (Trainer output dir).')
    parser.add_argument('--model_name', type=str, default='gpt2', help='Backbone config/tokenizer name (e.g., gpt2).')
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--max_length', type=int, default=256)
    parser.add_argument('--eval_batch_size', type=int, default=32)
    parser.add_argument('--subset_val', type=int, default=0)
    parser.add_argument('--subset_test', type=int, default=0)

    parser.add_argument('--num_bits', type=int, default=8, choices=[8])
    parser.add_argument('--quant_mode', type=str, default='asymmetric', choices=['asymmetric', 'symmetric'])

    parser.add_argument('--output_dir', type=str, default='outputs/ptq-scratch-int8')
    args = parser.parse_args()

    set_seed(42)
    device = get_device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    # Load tokenizer and datasets (reuse utils)
    tokenizer = build_tokenizer(args.model_name)
    subsets = {'train': 0, 'val': args.subset_val, 'test': args.subset_test}
    _, val_ds, test_ds, collator = build_datasets(tokenizer, args.max_length, val_size=2000, subsets=subsets)

    # Load fine-tuned baseline model from Part 1
    model = GPT2ForSequenceClassification.from_pretrained(args.baseline_dir)
    model.to(device)
    model.eval()

    # Baseline quick eval for reference
    targs = TrainingArguments(output_dir=args.output_dir, per_device_eval_batch_size=args.eval_batch_size, report_to='none')
    trainer = Trainer(model=model, args=targs, eval_dataset=val_ds, tokenizer=tokenizer, data_collator=collator, compute_metrics=compute_metrics)

    # Apply weight-only PTQ (per-tensor)
    base_val = trainer.evaluate()
    symmetric = (args.quant_mode == 'symmetric')
    qstates = apply_weight_only_ptq(model, num_bits=args.num_bits, symmetric=symmetric)

    # Save quantized model (FP32 dequantized)
    quant_model_dir = os.path.join(args.output_dir, 'quantized_model')
    os.makedirs(quant_model_dir, exist_ok=True)
    model.save_pretrained(quant_model_dir)
    tokenizer.save_pretrained(quant_model_dir)

    # Save true INT8 tensors for file size comparison
    int8_state_dict = {}
    for name, module in model.named_modules():
        if hasattr(module, 'weight') and isinstance(module.weight, nn.Parameter):
            w = module.weight.data
            q, s, zp = linear_quantize_tensor(w, num_bits=args.num_bits, symmetric=symmetric)
            int8_state_dict[f"{name}.weight"] = q.cpu()
            int8_state_dict[f"{name}.scale"] = torch.tensor([s], dtype=torch.float32)
            int8_state_dict[f"{name}.zero_point"] = torch.tensor([zp], dtype=torch.int32)

    int8_model_path = os.path.join(quant_model_dir, 'model_int8.safetensors')
    save_file(int8_state_dict, int8_model_path)
    print(f"Saved true INT8 tensors to: {int8_model_path}")
    
    # Evaluate quantized model
    q_val = trainer.evaluate()
    test_out = trainer.predict(test_ds)
    logits, labels = test_out.predictions, test_out.label_ids
    preds = logits.argmax(axis=-1)
    test_metrics = compute_metrics((logits, labels))


    # Results, reporting, saving
    cm_path = os.path.join('reports/figs', f'confusion_matrix_ptq_scratch_int8_{args.quant_mode}.png')
    save_confusion_matrix(labels, preds, LABEL_NAMES, cm_path, title=f'PTQ-Scratch INT8 ({args.quant_mode}) — GPT-2 on AG News')
    classification_text_report(labels, preds, LABEL_NAMES, os.path.join(args.output_dir, 'classification_report_ptq.txt'))

    int8_total = int(sum(s.int8_bytes for s in qstates.values()))

    qsummary = {
        'mode': args.quant_mode,
        'num_bits': args.num_bits,
        'modules': {
            name: {
                'scale': s.scale,
                'zero_point': s.zero_point,
                'shape': list(s.orig_shape),
                'int8_bytes': s.int8_bytes,
            } for name, s in qstates.items()
        },
        'total_int8_bytes': int8_total,
        'total_int8_pretty': pretty_size(int8_total)
    }
    with open(os.path.join(args.output_dir, 'ptq_scratch_summary.json'), 'w') as f:
        json.dump(qsummary, f, indent=2)

    avg_ms = measure_latency(model, test_ds, collator, device, batch_size=1, warmup_steps=20, measure_steps=100)

    results: Dict[str, Any] = {
        'device': device,
        'quantization': {
            'mode': args.quant_mode,
            'num_bits': args.num_bits,
            'total_int8_bytes': int8_total,
            'total_int8_pretty': pretty_size(int8_total),
        },
        'baseline_val_metrics': {k: float(v) for k, v in base_val.items() if isinstance(v, (int, float, np.floating))},
        'ptq_val_metrics': {k: float(v) for k, v in q_val.items() if isinstance(v, (int, float, np.floating))},
        'ptq_test_metrics': {k: float(v) for k, v in test_metrics.items()},
        'latency_ms_per_sample_b1': float(avg_ms) if not math.isnan(avg_ms) else None,
        'confusion_matrix_png': cm_path,
    }

    with open(os.path.join(args.output_dir, 'metrics_ptq_scratch.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print('\n==== PTQ-Scratch INT8 Summary ====')
    print(json.dumps(results, indent=2))
    print('\nQuantization summary saved to:', os.path.join(args.output_dir, 'ptq_scratch_summary.json'))
    print('Confusion matrix saved to:', cm_path)


if __name__ == '__main__':
    main()
