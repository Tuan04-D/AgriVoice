"""Train the Vision Diagnosis Model: linear probe, then LoRA fine-tuning.

Prepare the datasets first (``download_datasets.py`` and ``prepare_datasets.py``),
then run ``python train.py``. Dataset and output locations come from
``config.py``; checkpoints, embeddings, split tables and histories are written
under ``config.OUTPUT_ROOT`` and consumed by ``test.py``.
"""

import argparse
import copy
import math
import time

import pandas as pd
import torch
import torch.nn as nn

import config
import data
import model as model_lib
import plots
import utils


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train the vision diagnosis model.")
    parser.add_argument("--stage", choices=("all", "linear_probe", "lora"), default="all")
    return parser.parse_args()


def train_linear_probe(features, labels, class_weights, dirs, device):
    """Train the linear probe on frozen embeddings with early stopping."""
    cfg = config.LINEAR_PROBE
    logger = utils.get_logger()
    train_x = features["train"].to(device)
    train_y = labels["train"].to(device)
    head = model_lib.build_head(train_x.shape[1], config.NUM_CLASSES, device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    generator = torch.Generator().manual_seed(config.SEED)
    best = {"macro_f1": -1.0, "nll": math.inf, "epoch": 0}
    best_state = None
    history = []
    stale = 0
    for epoch in range(1, cfg["max_epochs"] + 1):
        head.train()
        permutation = torch.randperm(len(train_x), generator=generator).to(device)
        total_loss = 0.0
        for offset in range(0, len(train_x), cfg["batch_size"]):
            index = permutation[offset:offset + cfg["batch_size"]]
            loss = criterion(head(train_x[index]), train_y[index])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(index)
        train_metrics = utils.classification_metrics(
            model_lib.head_logits(head, features["train"], device), labels["train"],
            config.NUM_CLASSES,
        )
        val_metrics = utils.classification_metrics(
            model_lib.head_logits(head, features["val"], device), labels["val"], config.NUM_CLASSES,
        )
        history.append({
            "epoch": epoch,
            "train_loss": total_loss / len(train_x),
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "val_nll": val_metrics["nll"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "lr": cfg["lr"],
        })
        if utils.is_better(val_metrics, best, cfg["min_delta"]):
            best = {
                "macro_f1": val_metrics["macro_f1"], "nll": val_metrics["nll"], "epoch": epoch,
                "accuracy": val_metrics["accuracy"],
            }
            best_state = copy.deepcopy(head.state_dict())
            stale = 0
        else:
            stale += 1
        logger.info(
            "[linear_probe] epoch %d | loss %.4f | train acc %.4f | val nll %.4f | val acc %.4f | "
            "val macro-F1 %.4f | stale %d", epoch, history[-1]["train_loss"],
            train_metrics["accuracy"], val_metrics["nll"], val_metrics["accuracy"],
            val_metrics["macro_f1"], stale,
        )
        if stale >= cfg["patience"]:
            logger.info(
                "[linear_probe] early stopping at epoch %d (best epoch %d)", epoch, best["epoch"]
            )
            break
    head.load_state_dict(best_state)
    pd.DataFrame(history).to_csv(dirs["metrics"] / "linear_probe_history.csv", index=False)
    plots.plot_history(history, dirs["plots"] / "linear_probe_history.png", "linear_probe")
    torch.save(
        {
            "state_dict": {f"head.{key}": value.cpu() for key, value in head.state_dict().items()},
            "embed_dim": train_x.shape[1],
            "class_names": config.CLASS_NAMES,
            "best": best,
        },
        dirs["checkpoints"] / "linear_probe_best.pt",
    )
    return head, history, best


def train_lora(visual, lp_head, frames, transforms, labels, class_weights, dirs, device, use_amp,
               start_time):
    """Fine-tune the backbone with LoRA adapters and a warm-started head."""
    cfg = config.LORA
    logger = utils.get_logger()
    train_transform, eval_transform = transforms
    modules = model_lib.inject_lora(visual, cfg)
    diagnosis_model = model_lib.DiagnosisModel(visual, copy.deepcopy(lp_head)).to(device)
    lora_params = [
        parameter for name, parameter in diagnosis_model.named_parameters()
        if parameter.requires_grad and "lora_" in name
    ]
    head_params = list(diagnosis_model.head.parameters())
    trainable_count = sum(parameter.numel() for parameter in lora_params + head_params)
    total_count = sum(parameter.numel() for parameter in diagnosis_model.parameters())
    logger.info(
        "[lora] modules: %d, trainable params: %d / %d (%.3f%%)", len(modules), trainable_count,
        total_count, 100 * trainable_count / total_count,
    )

    batch_size = model_lib.probe_batch_size(
        diagnosis_model, device, use_amp, cfg, config.MODEL["image_size"], config.NUM_CLASSES
    )
    lr_scale = math.sqrt(batch_size / cfg["reference_batch_size"])
    optimizer = torch.optim.AdamW([
        {"params": lora_params, "lr": cfg["lr_lora"] * lr_scale,
         "weight_decay": cfg["weight_decay"]},
        {"params": head_params, "lr": cfg["lr_head"] * lr_scale, "weight_decay": 1e-4},
    ])
    train_loader = data.make_loader(
        frames["train"], train_transform, batch_size, shuffle=True, device=device, persistent=True
    )
    val_loader = data.make_loader(
        frames["val"], eval_transform, config.DATA["eval_batch_size"], shuffle=False, device=device,
        persistent=True,
    )
    steps_per_epoch = len(train_loader)
    scheduler = model_lib.cosine_with_warmup(
        optimizer, cfg["warmup_epochs"] * steps_per_epoch, cfg["max_epochs"] * steps_per_epoch,
        cfg["min_lr_ratio"],
    )
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    logger.info(
        "[lora] batch size %d, lr scale %.3f, steps/epoch %d", batch_size, lr_scale, steps_per_epoch
    )

    best = {"macro_f1": -1.0, "nll": math.inf, "epoch": 0}
    best_state = None
    history = []
    stale = 0
    for epoch in range(1, cfg["max_epochs"] + 1):
        diagnosis_model.train()
        epoch_start = time.time()
        if use_amp:
            torch.cuda.reset_peak_memory_stats()
        loss_sum = torch.zeros((), device=device)
        correct = torch.zeros((), device=device)
        seen = 0
        for images, batch_labels, _ in train_loader:
            images = images.to(device, non_blocking=True)
            batch_labels = batch_labels.to(device, non_blocking=True)
            with utils.autocast_context(device, use_amp):
                logits = diagnosis_model(images)
            loss = criterion(logits.float(), batch_labels)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(lora_params + head_params, cfg["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            loss_sum += loss.detach() * len(batch_labels)
            correct += (logits.argmax(1) == batch_labels).sum()
            seen += len(batch_labels)
        train_seconds = time.time() - epoch_start
        val_features = data.extract_features(diagnosis_model.visual, val_loader, device, use_amp)
        val_metrics = utils.classification_metrics(
            model_lib.head_logits(diagnosis_model.head, val_features, device), labels["val"],
            config.NUM_CLASSES,
        )
        history.append({
            "epoch": epoch,
            "train_loss": float(loss_sum.item() / seen),
            "train_accuracy": float(correct.item() / seen),
            "val_nll": val_metrics["nll"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_balanced_accuracy": val_metrics["balanced_accuracy"],
            "lr_lora": optimizer.param_groups[0]["lr"],
            "lr_head": optimizer.param_groups[1]["lr"],
            "grad_scale": float(scaler.get_scale()) if use_amp else 1.0,
            "train_seconds": round(train_seconds, 1),
            "epoch_seconds": round(time.time() - epoch_start, 1),
            "train_images_per_second": round(seen / max(train_seconds, 1e-6), 1),
            "peak_gpu_memory_gb": (
                round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2) if use_amp else 0.0
            ),
        })
        if utils.is_better(val_metrics, best, cfg["min_delta"]):
            best = {
                "macro_f1": val_metrics["macro_f1"], "nll": val_metrics["nll"], "epoch": epoch,
                "accuracy": val_metrics["accuracy"],
            }
            best_state = model_lib.trainable_state(diagnosis_model)
            stale = 0
            torch.save(
                {
                    "state_dict": best_state, "epoch": epoch, "val_metrics": val_metrics,
                    "lora_config": cfg, "lora_modules": modules, "batch_size": batch_size,
                    "model": config.MODEL, "class_names": config.CLASS_NAMES,
                },
                dirs["checkpoints"] / "lora_best.pt",
            )
        else:
            stale += 1
        torch.save(
            {
                "state_dict": model_lib.trainable_state(diagnosis_model),
                "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(), "epoch": epoch, "best": best, "history": history,
            },
            dirs["checkpoints"] / "lora_last.pt",
        )
        pd.DataFrame(history).to_csv(dirs["metrics"] / "lora_history.csv", index=False)
        row = history[-1]
        logger.info(
            "[lora] epoch %d | loss %.4f | train acc %.4f | val nll %.4f | val acc %.4f | "
            "val macro-F1 %.4f | lr %.2e | %.0fs | %.0f img/s | %.2f GB | stale %d", epoch,
            row["train_loss"], row["train_accuracy"], row["val_nll"], row["val_accuracy"],
            row["val_macro_f1"], row["lr_lora"], row["epoch_seconds"],
            row["train_images_per_second"], row["peak_gpu_memory_gb"], stale,
        )
        if stale >= cfg["patience"]:
            logger.info("[lora] early stopping at epoch %d (best epoch %d)", epoch, best["epoch"])
            break
        if utils.elapsed_hours(start_time) > config.TIME_BUDGET_HOURS:
            logger.info("[lora] time budget reached at epoch %d", epoch)
            break
    diagnosis_model.load_state_dict(best_state, strict=False)
    plots.plot_history(history, dirs["plots"] / "lora_history.png", "lora")
    return {
        "model": diagnosis_model, "history": history, "best": best, "batch_size": batch_size,
        "modules": modules,
    }


def main():
    """Run the configured training stages and save all training outputs."""
    args = parse_args()
    dirs = utils.ensure_output_dirs(config.OUTPUT_ROOT)
    logger = utils.setup_logging(dirs["logs"] / "train.log")
    utils.seed_everything(config.SEED)
    device, use_amp = utils.device_and_amp()
    start_time = time.time()
    utils.save_json(config.as_dict(), dirs["logs"] / "config.json")
    utils.save_json(utils.environment_info(device), dirs["logs"] / "environment.json")
    logger.info("Output directory: %s", dirs["root"])

    frames, in_dist, near_ood, far_ood, dedup_stats = data.prepare_all(device)
    data.write_split_files(dirs, in_dist, near_ood, far_ood, frames, dedup_stats)
    labels = data.split_labels(frames)
    class_weights = utils.compute_class_weights(
        frames["train"]["class_index"].to_numpy(), config.NUM_CLASSES
    )
    logger.info(
        "Class weights: %s",
        dict(zip(config.CLASS_NAMES, class_weights.numpy().round(3).tolist())),
    )

    visual, mean, std = model_lib.load_backbone(config.MODEL, device)
    transforms = data.build_transforms(config.MODEL["image_size"], mean, std)
    embed_dim = visual.output_dim
    summary = {
        "model": config.MODEL,
        "embed_dim": embed_dim,
        "normalization": {"mean": mean, "std": std},
        "dedup": dedup_stats,
    }

    if args.stage in ("all", "linear_probe"):
        features = data.extract_all_features(
            visual, frames, transforms[1], device, use_amp, "linear_probe"
        )
        data.save_features(dirs["features"] / "linear_probe_features.pt", features, frames)
        lp_head, lp_history, lp_best = train_linear_probe(
            features, labels, class_weights, dirs, device
        )
        summary["linear_probe"] = {"best": lp_best, "epochs": len(lp_history)}
        logger.info("[linear_probe] best: %s", lp_best)
    else:
        lp_head, _ = model_lib.load_head_state(
            dirs["checkpoints"] / "linear_probe_best.pt", embed_dim, device
        )

    if args.stage in ("all", "lora") and config.LORA["enabled"]:
        result = train_lora(
            visual, lp_head, frames, transforms, labels, class_weights, dirs, device, use_amp,
            start_time,
        )
        features = data.extract_all_features(
            result["model"].visual, frames, transforms[1], device, use_amp, "lora"
        )
        data.save_features(dirs["features"] / "lora_features.pt", features, frames)
        summary["lora"] = {
            "best": result["best"], "epochs": len(result["history"]),
            "batch_size": result["batch_size"], "modules": result["modules"], "config": config.LORA,
        }
        logger.info("[lora] best: %s", result["best"])

    summary["runtime_hours"] = utils.elapsed_hours(start_time)
    utils.save_json(summary, dirs["metrics"] / "training_summary.json")
    logger.info("Training finished in %.2f hours", summary["runtime_hours"])


if __name__ == "__main__":
    main()
