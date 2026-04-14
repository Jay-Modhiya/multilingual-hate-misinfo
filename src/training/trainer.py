"""
src/training/trainer.py
────────────────────────
Full training loop for MuRIL classifier.

Features:
  - AdamW + linear warmup scheduler  (same as CA-3 DistilBERT setup)
  - MLflow experiment tracking       (logs every epoch)
  - Early stopping on val F1
  - Checkpoint saving (best model only)
  - FP16 mixed precision             (enabled on Kaggle GPU)
  - Works on CPU too                 (for local testing)

Run this file on Kaggle GPU notebook:
    !python src/training/trainer.py --task hate
    !python src/training/trainer.py --task misinfo
"""

import os
import sys
import time
import argparse
import logging
import yaml
import numpy as np
import mlflow
import mlflow.pytorch
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from transformers import get_linear_schedule_with_warmup

# ── project imports ──
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.data.loader     import load_config, load_davidson, load_hasoc, load_fakenewsnet
from src.data.preprocessor import CleanerPipeline, MuRILTokenizerWrapper, preprocess_splits
from src.models.muril_classifier import build_model, save_model
from src.models.evaluator import compute_metrics, full_report, plot_confusion_matrix

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
#  Device setup
# ─────────────────────────────────────────────

def get_device():
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    logger.warning("No GPU found — running on CPU (slow).")
    return torch.device("cpu")


# ─────────────────────────────────────────────
#  DataLoaders
# ─────────────────────────────────────────────

def build_dataloaders(train_ds, val_ds, test_ds, batch_size: int):
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_dl  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return train_dl, val_dl, test_dl


# ─────────────────────────────────────────────
#  One epoch
# ─────────────────────────────────────────────

def train_one_epoch(model, dataloader, optimizer, scheduler, scaler, device, use_fp16):
    model.train()
    total_loss, steps = 0.0, 0

    for batch in dataloader:
        labels = batch.pop("labels").to(device)
        batch  = {k: v.to(device) for k, v in batch.items()}

        optimizer.zero_grad()

        if use_fp16:
            with autocast():
                loss, _ = model(**batch, labels=labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss, _ = model(**batch, labels=labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        scheduler.step()
        total_loss += loss.item()
        steps += 1

    return total_loss / steps


# ─────────────────────────────────────────────
#  Evaluation pass
# ─────────────────────────────────────────────

def evaluate(model, dataloader, device, label_map):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in dataloader:
            labels = batch.pop("labels").to(device)
            batch  = {k: v.to(device) for k, v in batch.items()}
            _, logits = model(**batch)
            preds = logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    metrics = compute_metrics(all_labels, all_preds, label_map)
    return metrics, all_preds, all_labels


# ─────────────────────────────────────────────
#  Main trainer
# ─────────────────────────────────────────────

def train(cfg: dict, task: str = "hate"):
    """
    Full training pipeline for one task.

    task = "hate"   → trains on Davidson + HASOC combined
    task = "misinfo" → trains on FakeNewsNet
    """
    device  = get_device()
    use_fp16 = cfg["training"]["fp16"] and torch.cuda.is_available()
    scaler   = GradScaler() if use_fp16 else None

    # ── Load data ──
    logger.info(f"\n{'='*50}\nTask: {task.upper()}\n{'='*50}")

    tokenizer = MuRILTokenizerWrapper(cfg)

    if task == "hate":
        davidson = load_davidson(cfg)
        hasoc    = load_hasoc(cfg)

        # Merge both datasets for hate task
        import pandas as pd
        train_df = pd.concat([davidson["train"], hasoc["train"]], ignore_index=True).sample(frac=1, random_state=42)
        val_df   = pd.concat([davidson["val"],   hasoc["val"]],   ignore_index=True)
        test_df  = pd.concat([davidson["test"],  hasoc["test"]],  ignore_index=True)
        label_map = {0: "not_hate", 1: "hate"}

        from src.data.preprocessor import CleanerPipeline, HateMisinfoDataset
        cleaner  = CleanerPipeline(cfg)

        splits_mock = {"train": train_df, "val": val_df, "test": test_df, "label_map": label_map}
        train_ds, val_ds, test_ds = preprocess_splits(splits_mock, cfg, lang="hi", tokenizer=tokenizer)

    else:  # misinfo
        fakenews  = load_fakenewsnet(cfg)
        label_map = fakenews["label_map"]
        train_ds, val_ds, test_ds = preprocess_splits(fakenews, cfg, lang="en", tokenizer=tokenizer)

    train_dl, val_dl, test_dl = build_dataloaders(
        train_ds, val_ds, test_ds, cfg["training"]["batch_size"]
    )

    # ── Model ──
    model = build_model(cfg, task=task).to(device)

    # ── Optimizer & scheduler ──
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["training"]["learning_rate"]),
        weight_decay=cfg["training"]["weight_decay"],
    )
    total_steps  = len(train_dl) * cfg["training"]["epochs"]
    warmup_steps = int(total_steps * cfg["training"]["warmup_ratio"])
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # ── MLflow ──
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(f"{cfg['mlflow']['experiment_name']}-{task}")

    os.makedirs(cfg["paths"]["model_dir"], exist_ok=True)
    best_f1, patience_counter = 0.0, 0
    best_model_path = os.path.join(cfg["paths"]["model_dir"], f"muril_{task}_best.pt")

    with mlflow.start_run(run_name=f"muril-{task}-{time.strftime('%Y%m%d-%H%M')}"):

        # Log hyperparams
        mlflow.log_params({
            "model":          cfg["model"]["hf_checkpoint"],
            "task":           task,
            "epochs":         cfg["training"]["epochs"],
            "batch_size":     cfg["training"]["batch_size"],
            "learning_rate":  cfg["training"]["learning_rate"],
            "max_length":     cfg["preprocessing"]["max_length"],
            "fp16":           use_fp16,
            "train_samples":  len(train_ds),
            "val_samples":    len(val_ds),
        })
        mlflow.set_tags(cfg["mlflow"]["tags"])

        # ── Training loop ──
        for epoch in range(1, cfg["training"]["epochs"] + 1):
            t0 = time.time()

            train_loss = train_one_epoch(model, train_dl, optimizer, scheduler, scaler, device, use_fp16)
            val_metrics, _, _ = evaluate(model, val_dl, device, label_map)

            elapsed = time.time() - t0
            logger.info(
                f"Epoch {epoch}/{cfg['training']['epochs']}  "
                f"loss={train_loss:.4f}  "
                f"val_f1={val_metrics['f1']:.2f}%  "
                f"val_acc={val_metrics['accuracy']:.2f}%  "
                f"time={elapsed:.1f}s"
            )

            # Log to MLflow
            mlflow.log_metrics({
                "train_loss":  train_loss,
                "val_accuracy": val_metrics["accuracy"],
                "val_precision": val_metrics["precision"],
                "val_recall":   val_metrics["recall"],
                "val_f1":       val_metrics["f1"],
            }, step=epoch)

            # Early stopping + checkpointing
            if val_metrics["f1"] > best_f1:
                best_f1 = val_metrics["f1"]
                patience_counter = 0
                save_model(model, best_model_path)
                logger.info(f"  ✓ New best model saved (val F1={best_f1:.2f}%)")
            else:
                patience_counter += 1
                logger.info(f"  No improvement ({patience_counter}/{cfg['training']['early_stopping_patience']})")
                if patience_counter >= cfg["training"]["early_stopping_patience"]:
                    logger.info("Early stopping triggered.")
                    break

        # ── Final test evaluation ──
        logger.info("\nLoading best model for test evaluation …")
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        test_metrics, test_preds, test_labels = evaluate(model, test_dl, device, label_map)

        logger.info(f"\n{'='*50}")
        logger.info(f"FINAL TEST RESULTS — Task: {task.upper()}")
        logger.info(f"{'='*50}")
        logger.info(f"Accuracy : {test_metrics['accuracy']:.2f}%")
        logger.info(f"Precision: {test_metrics['precision']:.2f}%")
        logger.info(f"Recall   : {test_metrics['recall']:.2f}%")
        logger.info(f"F1-Score : {test_metrics['f1']:.2f}%")
        logger.info(f"\n{full_report(test_labels, test_preds, label_map)}")

        # Log final test metrics
        mlflow.log_metrics({
            "test_accuracy":  test_metrics["accuracy"],
            "test_precision": test_metrics["precision"],
            "test_recall":    test_metrics["recall"],
            "test_f1":        test_metrics["f1"],
        })

        # Save confusion matrix as artifact
        cm_path = f"outputs/confusion_matrix_{task}.png"
        os.makedirs("outputs", exist_ok=True)
        plot_confusion_matrix(
            test_labels, test_preds, label_map,
            title=f"MuRIL — {task.upper()} Test Set",
            save_path=cm_path
        )
        mlflow.log_artifact(cm_path)
        mlflow.log_artifact(best_model_path)

        logger.info(f"\nMLflow run complete. Model saved at: {best_model_path}")

    return test_metrics


# ─────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task",   default="hate",   choices=["hate", "misinfo"])
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    train(cfg, task=args.task)
