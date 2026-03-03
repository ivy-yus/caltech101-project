import os
import json
import numpy as np
from typing import Dict, Tuple

from PIL import Image
from skimage.color import rgb2gray
from skimage.feature import hog

from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from torchvision.datasets import ImageFolder

DATA_DIR = "./caltech-101"
IMG_SIZE = 128
SEED = 42


def get_splits(num_samples: int, targets: list) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    跟 deep_experiments 里一样：70/15/15 stratified split
    """
    indices = np.arange(num_samples)
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
    return train_idx, val_idx, test_idx


def extract_hog_features(dataset: ImageFolder, indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    从 ImageFolder 某些样本中提取 HOG 特征。
    """
    X, y = [], []
    for i in indices:
        path, label = dataset.samples[i]
        img = Image.open(path).convert("RGB")
        img = img.resize((IMG_SIZE, IMG_SIZE))
        gray = rgb2gray(np.array(img))

        # HOG 特征（参数可以调，这里给一个常见配置）
        feat = hog(
            gray,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            orientations=9,
            block_norm="L2-Hys",
        )
        X.append(feat)
        y.append(label)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    return X, y


def run_hog_svm() -> Dict:
    dataset = ImageFolder(DATA_DIR)  # 不用 transforms，这里自己处理
    num_samples = len(dataset.samples)
    targets = [s[1] for s in dataset.samples]

    train_idx, val_idx, test_idx = get_splits(num_samples, targets)

    print(f"Total samples: {num_samples}")
    print(f"Train/Val/Test: {len(train_idx)}/{len(val_idx)}/{len(test_idx)}")

    print("Extracting HOG features for train/val/test...")
    X_train, y_train = extract_hog_features(dataset, train_idx)
    X_val, y_val = extract_hog_features(dataset, val_idx)
    X_test, y_test = extract_hog_features(dataset, test_idx)

    print(f"Feature dim: {X_train.shape[1]}")

    # 这里用 LinearSVC，速度比 RBF SVC 好很多
    clf = LinearSVC(random_state=SEED, max_iter=5000)
    clf.fit(X_train, y_train)

    # 在验证集上看一下
    val_pred = clf.predict(X_val)
    val_acc = accuracy_score(y_val, val_pred)
    print(f"Val Accuracy (HOG + LinearSVC): {val_acc * 100:.2f}%")

    # 最终在测试集上评估
    test_pred = clf.predict(X_test)
    test_acc = accuracy_score(y_test, test_pred)
    report_dict = classification_report(
        y_test, test_pred, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_test, test_pred)

    print(f"Test Accuracy (HOG + LinearSVC): {test_acc * 100:.2f}%")

    metrics = {
        "test_accuracy": float(test_acc),
        "classification_report": report_dict,
        "confusion_matrix": cm.tolist(),
        "config": {
            "arch": "hog_svm",
            "img_size": IMG_SIZE,
            "feature": "HOG",
            "classifier": "LinearSVC",
        },
    }

    os.makedirs("results", exist_ok=True)
    with open(os.path.join("results", "hog_svm_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


if __name__ == "__main__":
    run_hog_svm()