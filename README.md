# Finetuning GPT2 and Quantization Analysis

Fine-tuned GPT2 on AG News and applied post-training quantization to analyze accuracy-efficiency trade-offs.

## Setup & Running

```bash
pip install -r requirements.txt
```

### Running Fine-Tuning

```bash
./run.sh baseline
```

This creates the following in `./outputs`:

```
baseline-gpt2-full
├── checkpoint-14750/
├── checkpoint-29500/
├── checkpoint-44250/
├── classification_report.txt
├── config.json
├── merges.txt
├── metrics_baseline.json
├── model.safetensors
├── special_tokens_map.json
├── tokenizer_config.json
├── tokenizer.json
├── training_args.bin
└── vocab.json
```

### Running Quantization from Scratch

```bash
./run.sh ptq
```

This creates the following in `./outputs`:

```
ptq-scratch-int8
├── classification_report_ptq.txt
├── metrics_ptq_scratch.json
├── ptq_scratch_summary.json
└── quantized_model
    ├── config.json
    ├── merges.txt
    ├── model_int8.safetensors
    ├── model.safetensors
    ├── special_tokens_map.json
    ├── tokenizer_config.json
    ├── tokenizer.json
    └── vocab.json
```

### Running BNB Quantization

```bash
./run.sh bnb
```

This creates the following in `./outputs`:

```
bnb-quantized
├── classification_report_int8.txt
├── classification_report_nf4.txt
├── gpt2-int8
│   ├── config.json
│   ├── merges.txt
│   ├── model.safetensors
│   ├── special_tokens_map.json
│   ├── tokenizer_config.json
│   ├── tokenizer.json
│   └── vocab.json
├── gpt2-nf4
│   ├── config.json
│   ├── merges.txt
│   ├── model.safetensors
│   ├── special_tokens_map.json
│   ├── tokenizer_config.json
│   ├── tokenizer.json
│   └── vocab.json
└── metrics_quant_bnb.json
```

## Pre-trained Models

Here are links to the pretrained models:

- [Baseline](https://iiithydstudents-my.sharepoint.com/:f:/g/personal/amol_vijayachandran_students_iiit_ac_in/EoUaS1efyb1BlXSUaIKi8vsBWm1v7bl_rtFnKvdjR7KauQ?e=fdmPBt)
- [INT8-Scratch](https://iiithydstudents-my.sharepoint.com/:f:/g/personal/amol_vijayachandran_students_iiit_ac_in/ErXRWdnOVzJApXPzRq8AvKEBKhNtlhgvLfCwNK-1Ae3Ueg?e=1JJTen)
- [INT8-bitsandbytes](https://iiithydstudents-my.sharepoint.com/:f:/g/personal/amol_vijayachandran_students_iiit_ac_in/Ev4vk2t5F6FPjf-tjsBV6U0BEwQsP7ryn1QEA65R2zBi8Q?e=552HMt)
- [NF4-bitsandbytes](https://iiithydstudents-my.sharepoint.com/:f:/g/personal/amol_vijayachandran_students_iiit_ac_in/EgApx_9_Yo5DoSR1LDDLjIUBmP4uAdcs3Hq1poCWWVAswg?e=ah2RiF)