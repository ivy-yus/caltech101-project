import numpy as np
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report
from skimage.feature import hog
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split

# Configurations
DATA_DIR = './caltech-101'
IMG_SIZE = 64  # Use 64x64 for SVM to speed up feature extraction and training

def extract_hog_features(dataset):
    """Extract HOG features and labels from a PyTorch dataset."""
    features = []
    labels = []
    
    print(f"Extracting HOG features for {len(dataset)} images... (This may take a minute)")
    for i in range(len(dataset)):
        # dataset[i] returns (tensor_image, label)
        img_tensor, label = dataset[i]
        
        # Convert tensor (C, H, W) to numpy array (H, W, C) for skimage
        img_np = img_tensor.numpy().transpose((1, 2, 0))
        
        # Extract HOG features
        # channel_axis=-1 handles RGB images directly
        fd = hog(img_np, orientations=9, pixels_per_cell=(8, 8),
                 cells_per_block=(2, 2), visualize=False, channel_axis=-1)
        
        features.append(fd)
        labels.append(label)
        
        if (i + 1) % 1000 == 0:
            print(f"Processed {i + 1}/{len(dataset)} images...")
            
    return np.array(features), np.array(labels)

def main():
    # 1. Load data (Resize and convert to Tensor, no augmentation for standard SVM)
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor()
    ])
    
    full_dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
    targets = full_dataset.targets
    
    # 2. Stratified Split (70% Train, 15% Val, 15% Test)
    train_idx, temp_idx = train_test_split(np.arange(len(targets)), test_size=0.30, stratify=targets, random_state=42)
    temp_targets = [targets[i] for i in temp_idx]
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.50, stratify=temp_targets, random_state=42)

    train_dataset = Subset(full_dataset, train_idx)
    test_dataset = Subset(full_dataset, test_idx) # For SVM, we can directly evaluate on Test or Val

    # 3. Extract Features
    print("--- Training Set ---")
    X_train, y_train = extract_hog_features(train_dataset)
    print("--- Test Set ---")
    X_test, y_test = extract_hog_features(test_dataset)

    # 4. Train Linear SVM
    print(f"\nTraining Linear SVM on {len(X_train)} samples with {X_train.shape[1]} features...")
    # dual="auto" is recommended for newer scikit-learn versions to optimize speed
    svm_clf = LinearSVC(C=1.0, max_iter=2000, dual=False, random_state=42)
    svm_clf.fit(X_train, y_train)

    # 5. Evaluate
    print("\nEvaluating model...")
    y_pred = svm_clf.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"SVM Test Accuracy: {acc * 100:.2f}%\n")
    
    # Print detailed report (Precision, Recall, F1-Score)
    print("Classification Report (Subset):")
    # We only print the overall averages to keep console clean, 
    # but you can remove output_dict to see all 101 classes
    report = classification_report(y_test, y_pred, zero_division=0)
    print(report)

if __name__ == '__main__':
    main()