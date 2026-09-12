"""Evaluate the trained models and export the serving artifacts.

Run ``python test.py`` after ``train.py``. Evaluation reads the embeddings saved
during training, so it needs neither a GPU nor the backbone weights. It writes
metrics, per-image predictions, plots and the artifacts consumed by the backend.
"""

import argparse
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix

import calibration
import config
import data
import model as model_lib
import plots
import utils

CHECKPOINTS = {"linear_probe": "linear_probe_best.pt", "lora": "lora_best.pt"}
SCORE_SPLITS = ("val", "test", "near_ood_val", "near_ood_test", "far_ood_val", "far_ood_test")
REPORT_SPLITS = ("test", "near_ood_test", "far_ood_test")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate the vision diagnosis model.")
    parser.add_argument("--models", nargs="*", default=["linear_probe", "lora"],
                        choices=list(CHECKPOINTS))
    return parser.parse_args()


def prediction_table(frame, probs, sets, scores, is_ood, levels, in_distribution):
    """Build the per-image prediction table for one split."""
    top3 = np.argsort(-probs, axis=1)[:, :3]
    table = pd.DataFrame({
        "image_id": frame["image_id"],
        "source": frame["source"],
        "label": frame["label"],
        "relpath": frame["relpath"],
        "true_class": frame["class_name"],
        "pred_class": [config.CLASS_NAMES[index] for index in top3[:, 0]],
        "pred_prob": probs.max(1),
        "top3": [
            "|".join(f"{config.CLASS_NAMES[column]}:{probs[row, column]:.4f}" for column in top3[row])
            for row in range(len(frame))
        ],
        "conformal_set": [
            "|".join(config.CLASS_NAMES[column] for column in np.flatnonzero(sets[row]))
            for row in range(len(frame))
        ],
        "set_size": sets.sum(1),
        **{f"score_{method}": values for method, values in scores.items()},
        "is_ood": is_ood,
        "gate_level": levels,
    })
    if in_distribution:
        table["correct"] = table["pred_class"] == table["true_class"]
    return table


def evaluate_model(name, head, features, frames, labels, dirs, device):
    """Evaluate one model and return its report and calibration state."""
    logger = utils.get_logger()
    logger.info("[%s] evaluation started", name)
    n_bins = config.CALIBRATION["n_bins"]
    logits = {
        split: model_lib.head_logits(head, values, device)
        for split, values in features.items() if split != "train"
    }
    numpy_labels = {split: labels[split].numpy() for split in ("val", "calib", "test")}
    report = {"model": name}

    report["classification"] = {
        split: utils.classification_metrics(logits[split], labels[split], config.NUM_CLASSES)
        for split in ("val", "calib", "test")
    }
    test_pred = logits["test"].argmax(1).numpy()
    report["per_class_test"] = classification_report(
        numpy_labels["test"], test_pred, labels=range(config.NUM_CLASSES),
        target_names=config.CLASS_NAMES, output_dict=True, zero_division=0,
    )
    matrix = confusion_matrix(numpy_labels["test"], test_pred, labels=range(config.NUM_CLASSES))
    pd.DataFrame(matrix, index=config.CLASS_NAMES, columns=config.CLASS_NAMES).to_csv(
        dirs["metrics"] / f"{name}_confusion_test.csv"
    )
    plots.plot_confusion(
        matrix, dirs["plots"] / f"{name}_confusion_test.png", f"{name} - test confusion matrix"
    )

    temperature = calibration.fit_temperature(logits["calib"], labels["calib"])
    probs = {split: calibration.softmax_np(values, temperature) for split, values in logits.items()}
    raw_test = calibration.softmax_np(logits["test"])
    before = utils.calibration_metrics(raw_test, numpy_labels["test"], n_bins)
    after = utils.calibration_metrics(probs["test"], numpy_labels["test"], n_bins)
    report["calibration"] = {
        "temperature": temperature,
        "calib_nll_before": utils.calibration_metrics(
            calibration.softmax_np(logits["calib"]), numpy_labels["calib"], n_bins
        )["nll"],
        "calib_nll_after": utils.calibration_metrics(
            probs["calib"], numpy_labels["calib"], n_bins
        )["nll"],
        "test_before": before,
        "test_after": after,
    }
    plots.plot_reliability(
        before, after, dirs["plots"] / f"{name}_reliability_test.png",
        f"{name} - reliability (T={temperature:.3f})",
    )
    logger.info(
        "[%s] temperature %.4f | test ECE %.4f -> %.4f | NLL %.4f -> %.4f", name, temperature,
        before["ece"], after["ece"], before["nll"], after["nll"],
    )

    correct = (probs["test"].argmax(1) == numpy_labels["test"]).astype(np.float64)
    raw_summary, raw_coverage, raw_risks = utils.risk_coverage(raw_test.max(1), correct)
    cal_summary, cal_coverage, cal_risks = utils.risk_coverage(probs["test"].max(1), correct)
    report["selective_prediction"] = {
        "uncalibrated_msp": raw_summary, "calibrated_msp": cal_summary
    }
    plots.plot_risk_coverage(
        {
            "uncalibrated": (raw_coverage, raw_risks, raw_summary["aurc"]),
            "calibrated": (cal_coverage, cal_risks, cal_summary["aurc"]),
        },
        dirs["plots"] / f"{name}_risk_coverage_test.png", f"{name} - risk-coverage (test)",
    )

    scores = {
        split: calibration.ood_scores(
            logits[split], features[split], temperature, features["train"], device
        )
        for split in SCORE_SPLITS
    }
    thresholds = calibration.ood_thresholds(scores["val"])
    validation = {
        method: {
            kind: calibration.ood_metrics(
                scores["val"][method], scores[f"{kind}_ood_val"][method], thresholds[method]
            )
            for kind in ("near", "far")
        }
        for method in config.OOD["methods"]
    }
    selected = calibration.select_ood_method(validation)
    ood_test = {
        method: {
            kind: calibration.ood_metrics(
                scores["test"][method], scores[f"{kind}_ood_test"][method], thresholds[method]
            )
            for kind in ("near", "far")
        }
        for method in config.OOD["methods"]
    }
    report["ood"] = {
        "tpr_target": config.OOD["tpr"], "thresholds": thresholds, "selected_method": selected,
        "validation": validation, "test": ood_test,
    }
    plots.plot_ood_histogram(
        scores, selected, thresholds[selected], dirs["plots"] / f"{name}_ood_{selected}.png",
        f"{name} - OOD score ({selected})",
    )
    logger.info(
        "[%s] OOD method %s | near AUROC %.4f | far AUROC %.4f", name, selected,
        ood_test[selected]["near"]["auroc"], ood_test[selected]["far"]["auroc"],
    )

    alpha = config.CONFORMAL["alpha"]
    lam = config.CONFORMAL["raps_lambda"]
    k_reg = calibration.raps_k_reg(probs["calib"], numpy_labels["calib"], alpha)
    conformal = {}
    sets = {}
    for method in config.CONFORMAL["methods"]:
        qhat = calibration.conformal_calibrate(
            probs["calib"], numpy_labels["calib"], method, alpha, lam, k_reg
        )
        sets[method] = {
            split: calibration.conformal_sets(probs[split], qhat, method, lam, k_reg)
            for split in REPORT_SPLITS
        }
        conformal[method] = {
            "qhat": qhat,
            "test": calibration.conformal_summary(sets[method]["test"], numpy_labels["test"]),
            "near_ood_test_avg_set_size": float(sets[method]["near_ood_test"].sum(1).mean()),
            "far_ood_test_avg_set_size": float(sets[method]["far_ood_test"].sum(1).mean()),
        }
        logger.info(
            "[%s] conformal %s | qhat %.4f | coverage %.4f | avg size %.3f", name, method, qhat,
            conformal[method]["test"]["coverage"], conformal[method]["test"]["avg_set_size"],
        )
    report["conformal"] = {"alpha": alpha, "lambda": lam, "k_reg": k_reg, "methods": conformal}
    plots.plot_set_sizes(
        {split: sets["raps"][split].sum(1) for split in REPORT_SPLITS},
        dirs["plots"] / f"{name}_raps_set_sizes.png", f"{name} - RAPS set sizes",
    )

    gate = {}
    for split in REPORT_SPLITS:
        is_ood = scores[split][selected] < thresholds[selected]
        levels = calibration.gate_levels(probs[split], sets["raps"][split], is_ood)
        gate[split] = calibration.gate_summary(levels, correct if split == "test" else None)
        table = prediction_table(
            frames[split], probs[split], sets["raps"][split], scores[split], is_ood, levels,
            split == "test",
        )
        table.to_csv(dirs["predictions"] / f"{name}_{split}.csv", index=False)
    report["gate_preview"] = {"rules": config.GATE_PREVIEW, "splits": gate}

    utils.save_json(report, dirs["metrics"] / f"{name}_report.json")
    state = {
        "temperature": temperature,
        "ood_thresholds": thresholds,
        "ood_method": selected,
        "conformal": {
            "method": "raps", "alpha": alpha, "lambda": lam, "k_reg": k_reg,
            "qhat": conformal["raps"]["qhat"],
            "qhat_by_method": {method: conformal[method]["qhat"] for method in conformal},
        },
    }
    return report, state


def export_artifacts(name, dirs, state, report, features, labels, training_summary):
    """Write the model, calibration and KNN-bank artifacts for serving."""
    payload = torch.load(
        dirs["checkpoints"] / CHECKPOINTS[name], map_location="cpu", weights_only=False
    )
    lora = None
    if name == "lora":
        lora = {"config": payload.get("lora_config"), "modules": payload.get("lora_modules")}
    torch.save(
        {
            "format_version": 1,
            "model": training_summary["model"],
            "embed_dim": training_summary["embed_dim"],
            "classes": config.CLASSES,
            "class_names": config.CLASS_NAMES,
            "lora": lora,
            "state_dict": payload["state_dict"],
            "feature_normalization": "l2",
        },
        dirs["artifacts"] / "vision_model_v1.pt",
    )
    np.savez_compressed(
        dirs["artifacts"] / "knn_bank_v1.npz",
        features=features["train"].half().numpy(),
        labels=labels["train"].numpy(),
    )
    selected = state["ood_method"]
    raps = report["conformal"]["methods"]["raps"]["test"]
    utils.save_json(
        {
            "format_version": 1,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "selected_model": name,
            "model_file": "vision_model_v1.pt",
            "knn_bank_file": "knn_bank_v1.npz",
            "preprocess": {
                "image_size": config.MODEL["image_size"],
                "mean": training_summary["normalization"]["mean"],
                "std": training_summary["normalization"]["std"],
                "resize": "shortest_side_bicubic",
                "crop": "center",
            },
            "classes": config.CLASSES,
            "temperature": state["temperature"],
            "ood": {
                "method": selected,
                "thresholds": state["ood_thresholds"],
                "tpr_target": config.OOD["tpr"],
                "knn_k": config.OOD["knn_k"],
                "score_direction": "higher_is_in_distribution",
            },
            "conformal": {**state["conformal"], "force_top1": True, "randomized": False},
            "gate_preview": config.GATE_PREVIEW,
            "test_metrics": {
                "accuracy": report["classification"]["test"]["accuracy"],
                "macro_f1": report["classification"]["test"]["macro_f1"],
                "ece_after_ts": report["calibration"]["test_after"]["ece"],
                "near_ood_auroc": report["ood"]["test"][selected]["near"]["auroc"],
                "far_ood_auroc": report["ood"]["test"][selected]["far"]["auroc"],
                "raps_coverage": raps["coverage"],
                "raps_avg_set_size": raps["avg_set_size"],
            },
        },
        dirs["artifacts"] / "vision_calib_v1.json",
    )


def summary_row(name, report):
    """Return the headline metrics of one model as a flat dictionary."""
    test = report["classification"]["test"]
    calibration_report = report["calibration"]
    method = report["ood"]["selected_method"]
    raps = report["conformal"]["methods"]["raps"]["test"]
    gate = report["gate_preview"]["splits"]["test"]
    return {
        "model": name,
        "val_macro_f1": report["classification"]["val"]["macro_f1"],
        "test_accuracy": test["accuracy"],
        "test_macro_f1": test["macro_f1"],
        "test_balanced_accuracy": test["balanced_accuracy"],
        "temperature": calibration_report["temperature"],
        "ece_before_ts": calibration_report["test_before"]["ece"],
        "ece_after_ts": calibration_report["test_after"]["ece"],
        "nll_before_ts": calibration_report["test_before"]["nll"],
        "nll_after_ts": calibration_report["test_after"]["nll"],
        "brier_after_ts": calibration_report["test_after"]["brier"],
        "aurc": report["selective_prediction"]["calibrated_msp"]["aurc"],
        "ood_method": method,
        "near_ood_auroc": report["ood"]["test"][method]["near"]["auroc"],
        "far_ood_auroc": report["ood"]["test"][method]["far"]["auroc"],
        "near_ood_fpr_at_95tpr": report["ood"]["test"][method]["near"]["fpr_at_tpr"],
        "far_ood_fpr_at_95tpr": report["ood"]["test"][method]["far"]["fpr_at_tpr"],
        "raps_coverage": raps["coverage"],
        "raps_avg_set_size": raps["avg_set_size"],
        "gate_high_rate_test": gate["distribution"]["high"],
        "gate_high_accuracy_test": gate["accuracy_by_level"]["high"],
    }


def write_summary(reports, final_name, dirs):
    """Write the comparison table as CSV, JSON and markdown."""
    frame = pd.DataFrame([summary_row(name, report) for name, report in reports.items()])
    frame.to_csv(dirs["metrics"] / "summary.csv", index=False)
    utils.save_json(
        {"final_model": final_name, "rows": frame.to_dict(orient="records")},
        dirs["metrics"] / "summary.json",
    )
    transposed = frame.set_index("model").T.reset_index().rename(columns={"index": "metric"})
    (dirs["metrics"] / "summary.md").write_text(
        f"# Vision Diagnosis Model - summary\n\nFinal model: **{final_name}**\n\n"
        f"{utils.markdown_table(transposed)}\n",
        encoding="utf-8",
    )
    utils.get_logger().info("Summary:\n%s", transposed.to_string(index=False))
    return frame


def main():
    """Evaluate every requested model and export the artifacts of the best one."""
    args = parse_args()
    dirs = utils.ensure_output_dirs(config.OUTPUT_ROOT)
    logger = utils.setup_logging(dirs["logs"] / "test.log")
    utils.seed_everything(config.SEED)
    device, _ = utils.device_and_amp()
    training_summary = utils.load_json(dirs["metrics"] / "training_summary.json")
    base_frames = data.frames_from_splits(dirs)

    reports = {}
    states = {}
    evaluated = {}
    for name in args.models:
        features, frames = data.load_features(
            dirs["features"] / f"{name}_features.pt", base_frames
        )
        labels = data.split_labels(frames)
        head, _ = model_lib.load_head_state(
            dirs["checkpoints"] / CHECKPOINTS[name], training_summary["embed_dim"], device
        )
        reports[name], states[name] = evaluate_model(
            name, head, features, frames, labels, dirs, device
        )
        evaluated[name] = {"features": features, "labels": labels}

    final_name = max(
        reports, key=lambda name: reports[name]["classification"]["val"]["macro_f1"]
    )
    logger.info("Final model (by val macro-F1): %s", final_name)
    export_artifacts(
        final_name, dirs, states[final_name], reports[final_name],
        evaluated[final_name]["features"], evaluated[final_name]["labels"], training_summary,
    )
    write_summary(reports, final_name, dirs)
    manifest = [
        {"path": str(path.relative_to(dirs["root"])), "bytes": path.stat().st_size}
        for path in sorted(dirs["root"].rglob("*")) if path.is_file()
    ]
    pd.DataFrame(manifest).to_csv(dirs["logs"] / "output_manifest.csv", index=False)
    logger.info("Evaluation finished: %d files under %s", len(manifest), dirs["root"])


if __name__ == "__main__":
    main()
