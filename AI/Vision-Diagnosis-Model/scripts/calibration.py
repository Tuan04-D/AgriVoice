"""Temperature scaling, OOD scoring, conformal prediction and gate mapping."""

import math

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

import config


def softmax_np(logits, temperature=1.0):
    """Return softmax probabilities as a NumPy array."""
    return torch.softmax(logits.float() / temperature, dim=1).numpy()


def fit_temperature(logits, labels):
    """Fit a single temperature by minimizing NLL, with a grid-search fallback."""
    logits = logits.float()
    labels = labels.long()
    log_temperature = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [log_temperature], lr=0.05, max_iter=500, line_search_fn="strong_wolfe"
    )

    def closure():
        optimizer.zero_grad()
        loss = F.cross_entropy(logits / log_temperature.exp(), labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    candidates = [log_temperature.detach().exp().item()]
    grid = torch.logspace(-2, 1.3, 400)
    grid_nll = torch.stack([F.cross_entropy(logits / value, labels) for value in grid])
    candidates.append(float(grid[grid_nll.argmin()]))
    candidates = [value for value in candidates if math.isfinite(value) and value > 0]
    return min(candidates, key=lambda value: F.cross_entropy(logits / value, labels).item())


def knn_similarity(features, bank, k, device):
    """Return the similarity to the k-th nearest neighbour in ``bank``."""
    bank = bank.to(device).float()
    k = min(k, len(bank))
    outputs = []
    for chunk in features.split(2048):
        similarity = chunk.to(device).float() @ bank.T
        outputs.append(similarity.topk(k, dim=1).values[:, -1].cpu())
    return torch.cat(outputs).numpy()


def ood_scores(logits, features, temperature, bank, device):
    """Return OOD scores where higher values mean more in-distribution."""
    return {
        "msp": softmax_np(logits, temperature).max(1),
        "energy": torch.logsumexp(logits.float(), dim=1).numpy(),
        "knn": knn_similarity(features, bank, config.OOD["knn_k"], device),
    }


def ood_metrics(id_scores, out_scores, threshold):
    """Return AUROC, AUPR, FPR at the target TPR and detection rates."""
    targets = np.concatenate([np.ones(len(id_scores)), np.zeros(len(out_scores))])
    scores = np.concatenate([id_scores, out_scores])
    false_positive, true_positive, _ = roc_curve(targets, scores)
    index = min(
        len(false_positive) - 1,
        int(np.searchsorted(true_positive, config.OOD["tpr"], side="left")),
    )
    return {
        "auroc": float(roc_auc_score(targets, scores)),
        "aupr_in": float(average_precision_score(targets, scores)),
        "aupr_out": float(average_precision_score(1 - targets, -scores)),
        "fpr_at_tpr": float(false_positive[index]),
        "id_flagged_rate": float((id_scores < threshold).mean()),
        "ood_detection_rate": float((out_scores < threshold).mean()),
        "n_id": int(len(id_scores)),
        "n_ood": int(len(out_scores)),
    }


def ood_thresholds(val_scores):
    """Return the score threshold per method at the target in-distribution TPR."""
    return {
        method: float(np.quantile(val_scores[method], 1 - config.OOD["tpr"]))
        for method in config.OOD["methods"]
    }


def select_ood_method(validation):
    """Pick the OOD method with the best mean AUROC over the validation pools."""
    return max(
        config.OOD["methods"],
        key=lambda method: np.mean(
            [validation[method]["near"]["auroc"], validation[method]["far"]["auroc"]]
        ),
    )


def raps_k_reg(probs, labels, alpha):
    """Return the RAPS regularization rank from calibration label ranks."""
    order = np.argsort(-probs, axis=1, kind="stable")
    ranks = np.argmax(order == labels[:, None], axis=1) + 1
    return max(1, int(np.quantile(ranks, 1 - alpha, method="higher")))


def conformal_calibrate(probs, labels, method, alpha, lam, k_reg):
    """Return the conformal quantile for THR, APS or RAPS scores."""
    count = len(labels)
    if method == "thr":
        scores = 1.0 - probs[np.arange(count), labels]
    else:
        order = np.argsort(-probs, axis=1, kind="stable")
        cumulative = np.cumsum(np.take_along_axis(probs, order, axis=1), axis=1)
        ranks = np.argmax(order == labels[:, None], axis=1)
        scores = cumulative[np.arange(count), ranks]
        if method == "raps":
            scores = scores + lam * np.maximum(0, ranks + 1 - k_reg)
    level = min(1.0, math.ceil((count + 1) * (1 - alpha)) / count)
    return float(np.quantile(scores, level, method="higher"))


def conformal_sets(probs, qhat, method, lam, k_reg):
    """Return the boolean prediction-set matrix, always including the top class."""
    count, classes = probs.shape
    if method == "thr":
        included = (1.0 - probs) <= qhat
    else:
        order = np.argsort(-probs, axis=1, kind="stable")
        cumulative = np.cumsum(np.take_along_axis(probs, order, axis=1), axis=1)
        if method == "raps":
            cumulative = cumulative + lam * np.maximum(0, np.arange(1, classes + 1) - k_reg)
        included_sorted = cumulative <= qhat
        included = np.zeros_like(included_sorted)
        np.put_along_axis(included, order, included_sorted, axis=1)
    included[np.arange(count), probs.argmax(1)] = True
    return included


def conformal_summary(sets, labels):
    """Return coverage and set-size statistics for prediction sets."""
    sizes = sets.sum(1)
    covered = sets[np.arange(len(labels)), labels]
    by_size = {
        int(size): {
            "count": int((sizes == size).sum()),
            "coverage": float(covered[sizes == size].mean()),
        }
        for size in np.unique(sizes)
    }
    class_coverage = {
        config.CLASS_NAMES[index]: float(covered[labels == index].mean())
        for index in range(config.NUM_CLASSES) if (labels == index).any()
    }
    return {
        "coverage": float(covered.mean()),
        "avg_set_size": float(sizes.mean()),
        "median_set_size": float(np.median(sizes)),
        "singleton_rate": float((sizes == 1).mean()),
        "by_set_size": by_size,
        "class_conditional_coverage": class_coverage,
        "min_class_coverage": float(min(class_coverage.values())),
    }


def gate_levels(probs, sets, is_ood):
    """Map probabilities, prediction sets and the OOD flag to gate levels."""
    sizes = sets.sum(1)
    top = probs.max(1)
    high = (sizes == 1) & (top >= config.GATE_PREVIEW["high_prob_threshold"])
    low = is_ood | (sizes > config.GATE_PREVIEW["max_medium_set_size"])
    return np.where(low, "low", np.where(high, "high", "medium"))


def gate_summary(levels, correct=None):
    """Return the gate level distribution and, optionally, accuracy per level."""
    summary = {
        "distribution": {
            level: float((levels == level).mean()) for level in ("high", "medium", "low")
        }
    }
    if correct is not None:
        summary["accuracy_by_level"] = {
            level: float(correct[levels == level].mean()) if (levels == level).any() else None
            for level in ("high", "medium", "low")
        }
    return summary
