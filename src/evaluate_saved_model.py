import os
import json
import csv
import argparse
import numpy as np

import torch
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score

import matplotlib.pyplot as plt

# Import your existing training utilities (same split logic!)
# This works because you're running: python src/evaluate_saved_model.py
from deep_experiments import get_dataloaders, build_model, set_seed, SEED, BATCH_SIZE


RESULTS_DIR = "results"
MODELS_DIR = "saved_models"
PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def paper_style():
    """Set a clean, paper-like matplotlib style."""
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 120,
        "savefig.dpi": 300,
    })


def get_device(device_str: str):
    if device_str == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device_str)


def load_metrics_config(exp: str):
    """Load training config from results/<exp>_metrics.json if exists."""
    path = os.path.join(RESULTS_DIR, f"{exp}_metrics.json")
    if not os.path.exists(path):
        return None, None

    with open(path, "r") as f:
        m = json.load(f)
    cfg = m.get("config", {})
    return m, cfg


def load_model_from_exp(exp: str, arch: str, img_size: int, freeze_backbone: bool, device):
    # build model with correct #classes (based on dataset)
    # NOTE: we call get_dataloaders() just to get num_classes & test_loader (split is deterministic by SEED)
    train_loader, val_loader, test_loader, num_classes = get_dataloaders(
        img_size=img_size,
        batch_size=BATCH_SIZE,
        augment=False,  # eval-time transform (no aug). Split stays the same due to fixed SEED.
    )

    model = build_model(arch, num_classes, freeze_backbone=freeze_backbone).to(device)

    ckpt = os.path.join(MODELS_DIR, f"best_{exp}.pth")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"Cannot find checkpoint: {ckpt}")

    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # Class names order from ImageFolder
    # We can retrieve it from the underlying dataset of test_loader
    # test_loader.dataset is Subset(full_eval, indices); full_eval is ImageFolder
    class_names = test_loader.dataset.dataset.classes

    return model, test_loader, class_names, num_classes


@torch.no_grad()
def run_inference(model, test_loader, device):
    all_labels = []
    all_preds = []
    all_probs = []

    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        all_labels.append(labels.cpu().numpy())
        all_preds.append(preds.cpu().numpy())
        all_probs.append(probs.cpu().numpy())

    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)
    y_prob = np.concatenate(all_probs)
    return y_true, y_pred, y_prob


def topk_accuracy(y_true, y_prob, k=5):
    topk = np.argsort(y_prob, axis=1)[:, -k:]
    hits = [(y_true[i] in topk[i]) for i in range(len(y_true))]
    return float(np.mean(hits))


def compute_per_class_accuracy(cm):
    # cm: rows = true class, cols = predicted
    denom = cm.sum(axis=1)
    denom = np.where(denom == 0, 1, denom)
    return cm.diagonal() / denom


def plot_confusion_matrix_paper(cm, class_names, out_prefix, normalize="true"):
    """
    Paper-grade confusion matrix:
    - normalize="true" => row-normalized (recommended for many classes)
    - sparse ticks to avoid clutter
    """
    paper_style()

    if normalize == "true":
        denom = cm.sum(axis=1, keepdims=True)
        denom = np.where(denom == 0, 1, denom)
        cm_show = cm / denom
        title = "Confusion Matrix (row-normalized)"
        vmin, vmax = 0.0, 1.0
    else:
        cm_show = cm
        title = "Confusion Matrix"
        vmin, vmax = None, None

    n = cm.shape[0]

    # Wider figure because 102x102 is dense; still readable in paper as overview.
    fig_w, fig_h = 6.5, 5.6  # ~ 2-column width-ish
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(cm_show, interpolation="nearest", aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")

    # Sparse tick labels (indices), no class names (too many)
    step = 10 if n >= 60 else 5
    ticks = np.arange(0, n, step)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks])
    ax.set_yticklabels([str(t) for t in ticks])

    # Light grid lines (subtle)
    ax.set_xticks(np.arange(-.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-.5, n, 1), minor=True)
    ax.grid(which="minor", linestyle="-", linewidth=0.2, alpha=0.08)
    ax.tick_params(which="minor", bottom=False, left=False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.ax.set_ylabel("Rate" if normalize == "true" else "Count", rotation=90)

    fig.tight_layout()

    fig.savefig(os.path.join(PLOTS_DIR, f"{out_prefix}_cm_norm.png"))
    fig.savefig(os.path.join(PLOTS_DIR, f"{out_prefix}_cm_norm.pdf"))
    plt.close(fig)


def plot_per_class_accuracy_paper(per_class_acc, out_prefix):
    """
    Paper-grade per-class accuracy distribution:
    - sorted curve / bar to show spread
    - highlight tails (worst/best)
    """
    paper_style()

    acc_sorted = np.sort(per_class_acc)
    n = len(acc_sorted)

    fig, ax = plt.subplots(figsize=(5.8, 3.0))  # single-column friendly
    ax.plot(np.arange(n), acc_sorted, linewidth=2.0)
    ax.set_xlabel("Classes (sorted)")
    ax.set_ylabel("Per-class accuracy")
    ax.set_title("Per-class Accuracy Distribution")
    ax.set_ylim(0.0, 1.0)

    # Add reference lines (subtle)
    for y in [0.5, 0.75, 0.9]:
        ax.axhline(y, linestyle="--", linewidth=1.0, alpha=0.25)

    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, f"{out_prefix}_per_class_acc.png"))
    fig.savefig(os.path.join(PLOTS_DIR, f"{out_prefix}_per_class_acc.pdf"))
    plt.close(fig)


def save_top_confusions(cm, class_names, out_csv, top_k=20):
    """
    Save top confusing (true -> predicted) pairs by normalized confusion rate.
    Uses row-normalized off-diagonal rates.
    """
    n = cm.shape[0]
    denom = cm.sum(axis=1, keepdims=True)
    denom = np.where(denom == 0, 1, denom)
    cm_norm = cm / denom

    pairs = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            rate = cm_norm[i, j]
            if rate > 0:
                pairs.append((rate, i, j, int(cm[i, j])))

    pairs.sort(reverse=True, key=lambda x: x[0])
    pairs = pairs[:top_k]

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rate(row_norm)", "true_idx", "pred_idx", "count"])
        writer.writerows(pairs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, required=True,
                        help="Experiment name prefix, e.g. vit_b16_ft_all_reg (matches results/<exp>_metrics.json and saved_models/best_<exp>.pth)")
    parser.add_argument("--device", type=str, default="auto",
                        help="auto | cpu | cuda | mps")
    parser.add_argument("--arch", type=str, default=None,
                        help="Override arch if no metrics json. e.g. vit_b_16 / resnet50 / resnet18 / efficientnet_b0")
    parser.add_argument("--img_size", type=int, default=None,
                        help="Override img_size if no metrics json. e.g. 224 for ViT")
    parser.add_argument("--freeze_backbone", action="store_true",
                        help="Override freeze_backbone if no metrics json.")
    args = parser.parse_args()

    set_seed(SEED)
    device = get_device(args.device)
    print(f"[eval] device={device}")

    metrics_json, cfg = load_metrics_config(args.exp)

    # Resolve config
    if cfg:
        arch = cfg.get("arch")
        img_size = int(cfg.get("img_size"))
        freeze_backbone = bool(cfg.get("freeze_backbone", True))
    else:
        # fallback to CLI
        if args.arch is None or args.img_size is None:
            raise ValueError("No results/<exp>_metrics.json found. Please provide --arch and --img_size.")
        arch = args.arch
        img_size = args.img_size
        freeze_backbone = args.freeze_backbone

    print(f"[eval] exp={args.exp} arch={arch} img_size={img_size} freeze_backbone={freeze_backbone}")

    model, test_loader, class_names, num_classes = load_model_from_exp(
        args.exp, arch, img_size, freeze_backbone, device
    )

    y_true, y_pred, y_prob = run_inference(model, test_loader, device)

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    top5 = topk_accuracy(y_true, y_prob, k=5)

    cm = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))
    per_class_acc = compute_per_class_accuracy(cm)

    # Save extra eval json
    extra = {
        "exp": args.exp,
        "arch": arch,
        "img_size": img_size,
        "freeze_backbone": freeze_backbone,
        "test_accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "top5_accuracy": float(top5),
        "per_class_accuracy": per_class_acc.tolist(),
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(y_true, y_pred, output_dict=True, zero_division=0),
    }
    out_json = os.path.join(RESULTS_DIR, f"{args.exp}_extra_eval.json")
    with open(out_json, "w") as f:
        json.dump(extra, f, indent=2)
    print(f"[eval] wrote: {out_json}")

    # Plots (paper-grade)
    out_prefix = args.exp
    plot_confusion_matrix_paper(cm, class_names, out_prefix, normalize="true")
    plot_per_class_accuracy_paper(per_class_acc, out_prefix)

    # Top confusions csv
    out_csv = os.path.join(PLOTS_DIR, f"{out_prefix}_top_confusions.csv")
    save_top_confusions(cm, class_names, out_csv, top_k=25)
    print(f"[eval] wrote: {out_csv}")

    print(f"[eval] Acc={acc:.4f} Macro-F1={macro_f1:.4f} Weighted-F1={weighted_f1:.4f} Top-5={top5:.4f}")
    print(f"[eval] plots: {PLOTS_DIR}/{out_prefix}_cm_norm.(png|pdf), {PLOTS_DIR}/{out_prefix}_per_class_acc.(png|pdf)")


if __name__ == "__main__":
    main()