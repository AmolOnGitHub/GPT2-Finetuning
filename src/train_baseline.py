"""
Notes:
- GPT-2 has no pad token; we set pad_token = eos_token.
"""

import os, json, argparse, math

from transformers import GPT2ForSequenceClassification, Trainer, TrainingArguments, set_seed

from utils.data_utils import build_tokenizer, build_datasets, LABEL_NAMES
from utils.metrics_utils import compute_metrics, save_confusion_matrix, classification_text_report
from utils.efficiency_utils import (
    get_device, folder_size_bytes, pretty_size, measure_latency
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="gpt2")   # gpt2 is default GPT-2 Small
    parser.add_argument("--output_dir", type=str, default="outputs/baseline-gpt2")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--val_size", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--fp16", type=lambda x: str(x).lower()=="true", default=False)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--subset_train", type=int, default=0)
    parser.add_argument("--subset_val", type=int, default=0)
    parser.add_argument("--subset_test", type=int, default=0)
    parser.add_argument("--latency_batch_size", type=int, default=1)
    parser.add_argument("--latency_warmup", type=int, default=20)
    parser.add_argument("--latency_steps", type=int, default=100)
    args = parser.parse_args()

    device = get_device(args.device)
    set_seed(42)
    os.makedirs(args.output_dir, exist_ok=True)


    # Load GPT2, add padding tokens, add new head for classification
    tokenizer = build_tokenizer(args.model_name)
    model = GPT2ForSequenceClassification.from_pretrained(args.model_name, num_labels=len(LABEL_NAMES))
    model.config.pad_token_id = tokenizer.pad_token_id
    model.to(device)


    # Load and preprocess AG News dataset
    subsets = {"train": args.subset_train, "val": args.subset_val, "test": args.subset_test}
    train_ds, val_ds, test_ds, collator = build_datasets(tokenizer, args.max_length, args.val_size, subsets)


    # All hyperparameters for Trainer
    ## Model will evaluae and save at end of each epoch
    ## Keep best model by accuracy
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        learning_rate=args.lr,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        report_to="none",
        fp16=bool(args.fp16 and device=="cuda"),    # mixed precision on CUDA
    )


    # Training loop via Transformers, compute metrics using our function
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(args.output_dir)

    # Evaluate on validation and test sets, save confusion matrix and classification report
    
    val_metrics = trainer.evaluate()
    test_preds = trainer.predict(test_ds)
    test_logits, test_labels = test_preds.predictions, test_preds.label_ids
    preds = test_logits.argmax(axis=-1)
    test_metrics = compute_metrics((test_logits, test_labels))

    save_confusion_matrix(test_labels, preds, LABEL_NAMES, os.path.join("reports/figs", "confusion_matrix_baseline.png"), title="Baseline — GPT‑2 on AG News")
    classification_text_report(test_labels, preds, LABEL_NAMES, os.path.join(args.output_dir, "classification_report.txt"))

    size_bytes = folder_size_bytes(args.output_dir)
    latency = measure_latency(model, test_ds, collator, device, args.latency_batch_size, args.latency_warmup, args.latency_steps)

    results = {
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "model_size": pretty_size(size_bytes),
        "latency_ms_per_sample": latency,
        "device": device,
    }
    with open(os.path.join(args.output_dir, "metrics_baseline.json"), "w") as f: json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()