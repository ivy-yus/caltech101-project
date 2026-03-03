import os
import json
import csv
from typing import List, Dict, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

RESULTS_DIR = "results"
PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

# -----------------------------
# Global plot defaults (paper-ish)
# -----------------------------
def set_paper_defaults():
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["figure.dpi"] = 120
    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.titlesize"] = 11
    plt.rcParams["axes.labelsize"] = 10
    plt.rcParams["legend.fontsize"] = 9
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.25
    plt.rcParams["grid.linestyle"] = "--"
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False

set_paper_defaults()

# -----------------------------
# Helpers
# -----------------------------
def ema_smooth(values: List[float], alpha: float = 0.35) -> np.ndarray:
    """Exponential moving average smoothing for visualization only."""
    v = np.array(values, dtype=float)
    if len(v) == 0:
        return v
    out = np.zeros_like(v)
    out[0] = v[0]
    for i in range(1, len(v)):
        out[i] = alpha * v[i] + (1 - alpha) * out[i - 1]
    return out

def safe_float(x):
    return None if x is None else float(x)

def save_csv(rows: List[Dict], path: str):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "exp", "arch", "test_acc", "top5_acc", "macro_f1", "weighted_f1",
                "augment", "freeze_backbone"
            ],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

def get_row_map(rows: List[Dict]) -> Dict[str, Dict]:
    return {r["exp"]: r for r in rows}

def grouped_bar_two_metrics(
    rows: List[Dict],
    exps: List[str],
    metric_a: str,
    metric_b: str,
    filename: str,
    title: str,
    ylabel: str,
    name_map: Optional[Dict[str, str]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    legend_labels: Tuple[str, str] = ("Metric A", "Metric B"),
):
    """
    Create a single 'paper-style' grouped bar chart with two metrics per model.
    - metric_a/metric_b: keys like 'test_acc', 'macro_f1'
    """
    row_map = get_row_map(rows)

    labels = []
    a_vals = []
    b_vals = []
    for e in exps:
        if e not in row_map:
            continue
        ra = safe_float(row_map[e].get(metric_a))
        rb = safe_float(row_map[e].get(metric_b))
        if ra is None or rb is None:
            continue
        labels.append(name_map.get(e, e) if name_map else e)
        a_vals.append(ra)
        b_vals.append(rb)

    x = np.arange(len(labels))
    width = 0.38

    plt.figure(figsize=(7.0, 3.0))
    plt.bar(x - width / 2, a_vals, width, label=legend_labels[0])
    plt.bar(x + width / 2, b_vals, width, label=legend_labels[1])

    plt.xticks(x, labels, rotation=20, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
    if ylim:
        plt.ylim(*ylim)
    plt.grid(axis="y")
    plt.legend(frameon=False, ncol=2, loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, filename))
    plt.close()

def plot_curves_paper(
    exp_name: str,
    smooth_alpha: float = 0.18,
    acc_ylim: Optional[Tuple[float, float]] = None,   # None => auto
    loss_ylim: Optional[Tuple[float, float]] = None,  # None => auto
):
    """
    Clean paper-style curves:
    - smoothed only
    - no raw curves
    - auto y-limits to avoid "missing lines"
    """
    path = os.path.join(RESULTS_DIR, f"{exp_name}_metrics.json")
    if not os.path.exists(path):
        print(f"[WARN] metrics file not found: {path}")
        return

    with open(path, "r") as f:
        metrics = json.load(f)

    hist = metrics.get("history", None)
    if not hist:
        print(f"[WARN] no history found for {exp_name}")
        return

    epochs = np.arange(1, len(hist["train_loss"]) + 1)

    train_loss_s = ema_smooth(hist["train_loss"], alpha=smooth_alpha)
    val_loss_s   = ema_smooth(hist["val_loss"],   alpha=smooth_alpha)
    train_acc_s  = ema_smooth(hist["train_acc"],  alpha=smooth_alpha)
    val_acc_s    = ema_smooth(hist["val_acc"],    alpha=smooth_alpha)

    # ---- auto y-limits (paper-friendly)
    def auto_ylim(vals, pad_ratio=0.08, hard_min=None, hard_max=None):
        vmin = float(np.min(vals))
        vmax = float(np.max(vals))
        if np.isclose(vmin, vmax):
            vmin -= 1.0
            vmax += 1.0
        span = vmax - vmin
        lo = vmin - pad_ratio * span
        hi = vmax + pad_ratio * span
        if hard_min is not None:
            lo = max(lo, hard_min)
        if hard_max is not None:
            hi = min(hi, hard_max)
        return (lo, hi)

    # ----- Loss plot -----
    plt.figure(figsize=(5.4, 3.0))
    plt.plot(epochs, train_loss_s, linewidth=2.0, label="Train")
    plt.plot(epochs, val_loss_s,   linewidth=2.0, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{exp_name}: Training Loss")
    if loss_ylim is None:
        plt.ylim(*auto_ylim(np.concatenate([train_loss_s, val_loss_s])))
    else:
        plt.ylim(*loss_ylim)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"{exp_name}_loss.png"))
    plt.close()

    # ----- Accuracy plot -----
    plt.figure(figsize=(5.4, 3.0))
    plt.plot(epochs, train_acc_s, linewidth=2.0, label="Train")
    plt.plot(epochs, val_acc_s,   linewidth=2.0, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title(f"{exp_name}: Training Accuracy")
    if acc_ylim is None:
        # accuracy has natural bounds
        plt.ylim(*auto_ylim(np.concatenate([train_acc_s, val_acc_s]), hard_min=0, hard_max=100))
    else:
        plt.ylim(*acc_ylim)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"{exp_name}_acc.png"))
    plt.close()

# -----------------------------
# Load metrics
# -----------------------------
rows = []

for fname in os.listdir(RESULTS_DIR):
    if not fname.endswith("_metrics.json"):
        continue

    exp_name = fname.replace("_metrics.json", "")
    path = os.path.join(RESULTS_DIR, fname)

    with open(path, "r") as f:
        metrics = json.load(f)

    cfg = metrics.get("config", {})

    acc = metrics.get("test_accuracy", None)
    top5 = metrics.get("top5_accuracy", None)

    cr = metrics.get("classification_report", {})
    macro_f1 = cr.get("macro avg", {}).get("f1-score", None)
    weighted_f1 = cr.get("weighted avg", {}).get("f1-score", None)

    rows.append(
        {
            "exp": exp_name,
            "arch": cfg.get("arch", "unknown"),
            "test_acc": acc,
            "top5_acc": top5,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "augment": cfg.get("augment", None),
            "freeze_backbone": cfg.get("freeze_backbone", None),
        }
    )

def acc_key(r):
    return float(r["test_acc"]) if r["test_acc"] is not None else -1.0

rows = sorted(rows, key=acc_key, reverse=True)

print("=== Summary (Accuracy / Macro-F1 / Weighted-F1) ===")
for r in rows:
    print(
        f"{r['exp']:>25s} | "
        f"arch={r['arch']:<18s} | "
        f"acc={r['test_acc']:.4f} | "
        f"macro_f1={r['macro_f1']:.4f} | "
        f"weighted_f1={r['weighted_f1']:.4f}"
    )

csv_path = os.path.join(PLOTS_DIR, "summary_metrics.csv")
save_csv(rows, csv_path)
print(f"[OK] Saved summary CSV: {csv_path}")

# -----------------------------
# Names
# -----------------------------
pretty_names = {
    "hog_svm": "HOG + SVM",
    "effnet_b0_main": "EffNet-B0",
    "resnet18_main": "ResNet-18",
    "resnet50_main": "ResNet-50",
    "vit_b16_main": "ViT-B/16 (head)",
    "vit_b16_noaug": "ViT-B/16 (head, no aug)",
    "vit_b16_ft_all": "ViT-B/16 (full FT)",
    "vit_b16_strongaug_ft_all": "ViT-B/16 (full FT, strong aug)",
    "vit_b16_ft_all_reg": "ViT-B/16 (full FT + reg)",
}

vit_names = {
    "vit_b16_main": "Head + weak aug",
    "vit_b16_noaug": "Head + no aug",
    "vit_b16_ft_all": "Full FT + weak aug",
    "vit_b16_strongaug_ft_all": "Full FT + strong aug",
    "vit_b16_ft_all_reg": "Full FT + reg",
}

# -----------------------------
# 1) Overall comparison: ONE grouped bar figure
# -----------------------------
main_exps = [
    "hog_svm",
    "effnet_b0_main",
    "resnet18_main",
    "resnet50_main",
    "vit_b16_main",
    "vit_b16_ft_all",
    "vit_b16_ft_all_reg",
]

grouped_bar_two_metrics(
    rows,
    main_exps,
    metric_a="test_acc",
    metric_b="macro_f1",
    filename="model_comparison_acc_macroF1.png",
    title="Model Comparison on Caltech-101",
    ylabel="Score",
    name_map=pretty_names,
    ylim=(0.40, 1.00),
    legend_labels=("Test Accuracy", "Macro F1"),
)

# -----------------------------
# 2) ViT ablation: ONE grouped bar figure
# -----------------------------
vit_exps = [
    "vit_b16_main",
    "vit_b16_noaug",
    "vit_b16_ft_all",
    "vit_b16_strongaug_ft_all",
    "vit_b16_ft_all_reg",
]

grouped_bar_two_metrics(
    rows,
    vit_exps,
    metric_a="test_acc",
    metric_b="macro_f1",
    filename="vit_ablation_acc_macroF1.png",
    title="ViT-B/16 Ablation",
    ylabel="Score",
    name_map=vit_names,
    ylim=(0.85, 1.00),
    legend_labels=("Test Accuracy", "Macro F1"),
)

# -----------------------------
# 3) Curves: paper-ish, raw + smoothed + best epoch line
# -----------------------------
plot_curves_paper("resnet50_main", smooth_alpha=0.18)       
plot_curves_paper("vit_b16_ft_all", smooth_alpha=0.18)      
plot_curves_paper("vit_b16_ft_all_reg", smooth_alpha=0.18)  

print(f"[OK] Plots saved to: {PLOTS_DIR}")