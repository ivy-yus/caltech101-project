import os
import json
import numpy as np
from typing import Tuple, Dict

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# ------------------
# Global config
# ------------------
DATA_DIR = "./caltech-101"
DEFAULT_IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-4
NUM_WORKERS = 4
SEED = 42


def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ------------------
# Data loaders
# ------------------
def get_dataloaders(
    img_size: int = DEFAULT_IMG_SIZE,
    batch_size: int = BATCH_SIZE,
    augment: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader, int]:

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    if augment:

        train_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    img_size,
                    scale=(0.8, 1.0),
                    ratio=(0.75, 1.33),
                ),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                ),
                transforms.ToTensor(),
                normalize,
                transforms.RandomErasing(p=0.25),
            ]
        )
    else:
        train_transform = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                normalize,
            ]
        )

    eval_transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            normalize,
        ]
    )

    full_train = datasets.ImageFolder(DATA_DIR, transform=train_transform)
    full_eval = datasets.ImageFolder(DATA_DIR, transform=eval_transform)

    targets = full_train.targets
    indices = np.arange(len(targets))

    train_idx, temp_idx = train_test_split(
        indices,
        test_size=0.30,
        stratify=targets,
        random_state=SEED,
    )
    temp_targets = [targets[i] for i in temp_idx]
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.50,
        stratify=temp_targets,
        random_state=SEED,
    )

    train_loader = DataLoader(
        Subset(full_train, train_idx),
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )
    val_loader = DataLoader(
        Subset(full_eval, val_idx),
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )
    test_loader = DataLoader(
        Subset(full_eval, test_idx),
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    num_classes = len(full_train.classes)
    return train_loader, val_loader, test_loader, num_classes


# ------------------
# Model factory
# ------------------
def build_model(arch: str, num_classes: int, freeze_backbone: bool = True) -> nn.Module:

    if arch == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    elif arch == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    elif arch == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)

    elif arch == "vit_b_16":
        model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
        in_features = model.heads.head.in_features
        model.heads.head = nn.Linear(in_features, num_classes)

    else:
        raise ValueError(f"Unknown architecture: {arch}")

    if freeze_backbone:
        for p in model.parameters():
            p.requires_grad = False

        if arch in ["resnet50", "resnet18"]:
            for p in model.fc.parameters():
                p.requires_grad = True
        elif arch == "efficientnet_b0":
            for p in model.classifier.parameters():
                p.requires_grad = True
        elif arch == "vit_b_16":
            for p in model.heads.parameters():
                p.requires_grad = True

    return model

# ------------------
# Train & eval loops
# ------------------
def train_one_model(
    arch: str,
    img_size: int = DEFAULT_IMG_SIZE,
    augment: bool = True,
    exp_name: str = "exp",
    freeze_backbone: bool = True,
    weight_decay: float = 0.0,
    label_smoothing: float = 0.0,
) -> Dict:
    set_seed(SEED)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running {exp_name} on device: {device}")

    train_loader, val_loader, test_loader, num_classes = get_dataloaders(
        img_size=img_size,
        batch_size=BATCH_SIZE,
        augment=augment,
    )
    print(f"Num classes: {num_classes}")

    model = build_model(arch, num_classes, freeze_backbone=freeze_backbone).to(device)

    params_to_optimize = [p for p in model.parameters() if p.requires_grad]
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = optim.Adam(params_to_optimize, lr=LR, weight_decay=weight_decay)

    best_val_acc = 0.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    os.makedirs("saved_models", exist_ok=True)
    ckpt_path = os.path.join("saved_models", f"best_{exp_name}.pth")

    for epoch in range(EPOCHS):
        # Train
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_loss = running_loss / total
        train_acc = 100.0 * correct / total

        # Validate
        model.eval()
        val_running_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_running_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_loss = val_running_loss / val_total
        val_acc = 100.0 * val_correct / val_total

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"[{exp_name}] Epoch [{epoch+1}/{EPOCHS}] "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), ckpt_path)

    print(f"[{exp_name}] Finished training. Best Val Acc: {best_val_acc:.2f}%")

    # -------- Test evaluation --------
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    acc = accuracy_score(all_labels, all_preds)
    report_dict = classification_report(
        all_labels,
        all_preds,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(all_labels, all_preds)

    # Top-5 accuracy
    top5_correct = 0
    for probs, label in zip(all_probs, all_labels):
        top5 = np.argsort(probs)[-5:]
        if label in top5:
            top5_correct += 1
    top5_acc = top5_correct / len(all_labels)

    metrics = {
        "test_accuracy": acc,
        "top5_accuracy": top5_acc,
        "classification_report": report_dict,
        "confusion_matrix": cm.tolist(),
        "best_val_acc": best_val_acc / 100.0,
        "history": history,
        "config": {
            "arch": arch,
            "img_size": img_size,
            "augment": augment,
            "freeze_backbone": freeze_backbone,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "lr": LR,
        },
    }

    os.makedirs("results", exist_ok=True)
    with open(os.path.join("results", f"{exp_name}_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"[{exp_name}] Test Accuracy: {acc * 100:.2f}%, Top-5: {top5_acc * 100:.2f}%")
    return metrics

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp",
        type=str,
        default="resnet50_main",
        choices=[
            "resnet50_main",
            "resnet18_main",
            "effnet_b0_main",
            "vit_b16_main",
            "vit_b16_noaug",
            "vit_b16_ft_all",
            "resnet50_res64",
            "resnet50_noaug",
            "vit_b16_strongaug_ft_all",
            "vit_b16_ft_all_reg",
        ],
        help="Which experiment to run",
    )
    args = parser.parse_args()

    if args.exp == "resnet50_main":
        train_one_model("resnet50", img_size=128, augment=True, exp_name="resnet50_main")

    elif args.exp == "resnet18_main":
        train_one_model("resnet18", img_size=128, augment=True, exp_name="resnet18_main")

    elif args.exp == "effnet_b0_main":
        train_one_model("efficientnet_b0", img_size=128, augment=True, exp_name="effnet_b0_main")

    elif args.exp == "vit_b16_main":
        # baseline: frozen backbone + augmentation
        train_one_model(
            "vit_b_16",
            img_size=224,
            augment=True,
            exp_name="vit_b16_main",
            freeze_backbone=True,
        )

    elif args.exp == "vit_b16_noaug":
        # Ablation A: no augmentation
        train_one_model(
            "vit_b_16",
            img_size=224,
            augment=False,
            exp_name="vit_b16_noaug",
            freeze_backbone=True,
        )

    elif args.exp == "vit_b16_ft_all":
        # Ablation B: full fine-tuning
        train_one_model(
            "vit_b_16",
            img_size=224,
            augment=True,
            exp_name="vit_b16_ft_all",
            freeze_backbone=False,
        )
    elif args.exp == "vit_b16_strongaug_ft_all":
        train_one_model(
            "vit_b_16",
            img_size=224,
            augment=True,
            exp_name="vit_b16_strongaug_ft_all",
            freeze_backbone=False,     # full fine-tune
        )

    elif args.exp == "vit_b16_ft_all_reg":
        train_one_model(
            "vit_b_16",
            img_size=224,
            augment=True,
            exp_name="vit_b16_ft_all_reg",
            freeze_backbone=False,     # full fine-tune
            weight_decay=1e-4,
            label_smoothing=0.1,
        )