"""Matplotlib figures for training, calibration, OOD and conformal results."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config


def plot_history(history, path, title):
    """Plot loss and accuracy curves for a training run."""
    frame = pd.DataFrame(history)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(frame["epoch"], frame["train_loss"], label="train loss")
    axes[0].plot(frame["epoch"], frame["val_nll"], label="val NLL")
    axes[0].set_xlabel("epoch")
    axes[0].legend()
    axes[1].plot(frame["epoch"], frame["train_accuracy"], label="train accuracy")
    axes[1].plot(frame["epoch"], frame["val_accuracy"], label="val accuracy")
    axes[1].plot(frame["epoch"], frame["val_macro_f1"], label="val macro-F1")
    axes[1].set_xlabel("epoch")
    axes[1].legend()
    figure.suptitle(f"{title} training history")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def plot_confusion(matrix, path, title):
    """Plot a row-normalized confusion matrix with absolute counts."""
    normalized = matrix / np.maximum(matrix.sum(1, keepdims=True), 1)
    figure, axis = plt.subplots(figsize=(9, 8))
    image = axis.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    axis.set_xticks(range(config.NUM_CLASSES), config.CLASS_NAMES, rotation=60, ha="right",
                    fontsize=8)
    axis.set_yticks(range(config.NUM_CLASSES), config.CLASS_NAMES, fontsize=8)
    for row in range(config.NUM_CLASSES):
        for column in range(config.NUM_CLASSES):
            axis.text(column, row, int(matrix[row, column]), ha="center", va="center", fontsize=7,
                      color="white" if normalized[row, column] > 0.5 else "black")
    axis.set_xlabel("predicted")
    axis.set_ylabel("true")
    axis.set_title(title)
    figure.colorbar(image, ax=axis, fraction=0.046)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def plot_reliability(before, after, path, title):
    """Plot reliability diagrams before and after temperature scaling."""
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for axis, metrics, label in zip(axes, (before, after), ("Before TS", "After TS")):
        bins = metrics["bins"]
        centers = [(item["lower"] + item["upper"]) / 2 for item in bins]
        accuracies = [item["accuracy"] or 0.0 for item in bins]
        confidences = [item["confidence"] or 0.0 for item in bins]
        axis.bar(centers, accuracies, width=1 / len(bins), edgecolor="black", alpha=0.75,
                 label="accuracy")
        axis.bar(centers, np.array(confidences) - np.array(accuracies), bottom=accuracies,
                 width=1 / len(bins), color="tab:red", alpha=0.3, label="gap")
        axis.plot([0, 1], [0, 1], "--", color="gray")
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.set_xlabel("confidence")
        axis.set_ylabel("accuracy")
        axis.set_title(f"{label}: ECE={metrics['ece']:.4f}, NLL={metrics['nll']:.4f}")
        axis.legend(loc="upper left")
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def plot_risk_coverage(curves, path, title):
    """Plot selective risk against coverage for one or more scores."""
    figure, axis = plt.subplots(figsize=(6, 4.5))
    for label, (coverage, risks, aurc) in curves.items():
        axis.plot(coverage, risks, label=f"{label} (AURC={aurc:.4f})")
    axis.set_xlabel("coverage")
    axis.set_ylabel("selective risk")
    axis.set_title(title)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def plot_ood_histogram(scores, method, threshold, path, title):
    """Plot in-distribution and OOD score distributions with the threshold."""
    figure, axis = plt.subplots(figsize=(7, 4.5))
    labels = (("test", "in-distribution test"), ("near_ood_test", "near-OOD"),
              ("far_ood_test", "far-OOD"))
    for split, label in labels:
        axis.hist(scores[split][method], bins=50, alpha=0.5, density=True, label=label)
    axis.axvline(threshold, color="black", linestyle="--", label="threshold")
    axis.set_xlabel(f"{method} score (higher = in-distribution)")
    axis.set_title(title)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def plot_set_sizes(size_map, path, title):
    """Plot the distribution of conformal prediction-set sizes."""
    figure, axis = plt.subplots(figsize=(7, 4.5))
    bins = np.arange(0.5, config.NUM_CLASSES + 1.5)
    for label, sizes in size_map.items():
        axis.hist(sizes, bins=bins, alpha=0.5, density=True, label=label)
    axis.set_xlabel("prediction set size")
    axis.set_title(title)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
