"""Fine-tune a pretrained model to classify text as human (0) or AI-generated (1).

Needs ml/data/text/ from prepare_data.py. Uses only train + validation — never test.

Run from the ml folder:
    Quick laptop check (a few minutes, catches bugs, scores mean nothing):
        python text/train_text.py --limit 200 --epochs 1 --max-length 128
    Real run (on a GPU, e.g. Google Colab):
        python text/train_text.py

Saves the model, tokenizer, and validation metrics to ml/models/text/<run name>/.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

ML_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ML_DIR / "data" / "text"
MODELS_DIR = ML_DIR / "models" / "text"
LABELS = {0: "human", 1: "ai"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="distilbert/distilroberta-base",
                        help="Base model from DATASETS.md (or FacebookAI/roberta-base to compare)")
    parser.add_argument("--epochs", type=float, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=512,
                        help="Tokens per text; longer texts are cut off for the model")
    parser.add_argument("--limit", type=int, default=None,
                        help="Use only this many train and validation rows (for quick tests)")
    parser.add_argument("--label-smoothing", type=float, default=0.1,
                        help="Train toward 95/5 instead of 100/0 so scores don't pile up at 0.999+ (0 = off)")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def compute_metrics(eval_pred):
    """Validation metrics at the default 0.5 cutoff. The real threshold is chosen in a later step."""
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "accuracy": (tp + tn) / len(labels),
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "precision": precision,
        "recall": recall,
        "false_positive_rate": fp / (fp + tn) if fp + tn else 0.0,  # humans wrongly flagged as AI
    }


def main():
    args = parse_args()
    run_name = f"{args.model.split('/')[-1]}-{datetime.now():%Y%m%d-%H%M}"
    if args.limit:
        run_name += f"-limit{args.limit}"
    out_dir = MODELS_DIR / run_name

    data = load_dataset("json", data_files={
        "train": str(DATA_DIR / "train.jsonl"),
        "validation": str(DATA_DIR / "validation.jsonl"),
    })
    if args.limit:
        for split in data:
            data[split] = data[split].shuffle(seed=args.seed).select(range(min(args.limit, len(data[split]))))

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    data = data.map(
        lambda batch: tokenizer(batch["text"], truncation=True, max_length=args.max_length),
        batched=True,
        remove_columns=["text", "src"],
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=2,
        id2label=LABELS,
        label2id={name: i for i, name in LABELS.items()},
    )

    device = "cuda (GPU)" if torch.cuda.is_available() else \
        "mps (Apple GPU)" if torch.backends.mps.is_available() else "cpu"
    print(f"Training {args.model} on {len(data['train']):,} texts using {device}")

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(out_dir / "checkpoints"),
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size * 2,
            learning_rate=args.learning_rate,
            weight_decay=0.01,
            warmup_ratio=0.1,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=1,
            load_best_model_at_end=True,
            # Keep the epoch whose confidence best matched reality. F1 is already above target;
            # overconfidence is the problem, and validation loss is what measures it.
            metric_for_best_model="eval_loss",
            label_smoothing_factor=args.label_smoothing,
            fp16=torch.cuda.is_available(),
            logging_steps=50,
            report_to="none",
            seed=args.seed,
        ),
        train_dataset=data["train"],
        eval_dataset=data["validation"],
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )
    trainer.train()
    history = list(trainer.state.log_history)  # before evaluate() adds a repeat of the last epoch

    metrics = trainer.evaluate()
    trainer.save_model(str(out_dir))  # model + tokenizer, loadable with from_pretrained(out_dir)
    record = {
        "run": run_name,
        "base_model": args.model,
        "dataset": "yaful/MAGE (see data/text/manifest.json)",
        "trained": datetime.now().isoformat(timespec="seconds"),
        "device": device,
        "settings": vars(args),
        "train_rows": len(data["train"]),
        "validation_rows": len(data["validation"]),
        "validation_metrics_at_0.5": {k.removeprefix("eval_"): v for k, v in metrics.items()},
        # Loss and metrics after every epoch (the checkpoints folder holding this gets deleted).
        "log_history": history,
    }
    (out_dir / "metrics.json").write_text(json.dumps(record, indent=2))

    m = record["validation_metrics_at_0.5"]
    print(f"\nValidation (cutoff 0.5): F1 {m['f1']:.3f} | accuracy {m['accuracy']:.3f} | "
          f"false-positive rate {m['false_positive_rate']:.3f}")
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
