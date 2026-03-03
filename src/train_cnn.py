import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report # Added for final evaluation
import numpy as np

# Configurations
DATA_DIR = './caltech-101'
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-4

def get_dataloaders():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        normalize,
    ])
    
    eval_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        normalize,
    ])

    full_train = datasets.ImageFolder(DATA_DIR, transform=train_transform)
    full_eval = datasets.ImageFolder(DATA_DIR, transform=eval_transform)

    targets = full_train.targets
    train_idx, temp_idx = train_test_split(np.arange(len(targets)), test_size=0.30, stratify=targets, random_state=42)
    
    temp_targets = [targets[i] for i in temp_idx]
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.50, stratify=temp_targets, random_state=42)

    train_loader = DataLoader(Subset(full_train, train_idx), batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(Subset(full_eval, val_idx), batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    test_loader = DataLoader(Subset(full_eval, test_idx), batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    num_classes = len(full_train.classes)
    return train_loader, val_loader, test_loader, num_classes

def main():
    # 1. Setup Device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Load Data
    train_loader, val_loader, test_loader, num_classes = get_dataloaders()
    print(f"Data loaded. Number of classes: {num_classes}")

    # 3. Setup Model (ResNet50)
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    
    # Freeze backbone
    for param in model.parameters():
        param.requires_grad = False
        
    # Replace classification head
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    # 4. Loss & Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=LR)

    # 5. Training Loop
    best_val_acc = 0.0
    model_save_path = 'saved_models/best_resnet50.pth'
    
    for epoch in range(EPOCHS):
        # Training Phase
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
            
        train_acc = 100. * correct / total
        train_loss = running_loss / total
        
        # Validation Phase
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
        val_acc = 100. * correct / total
        val_loss = val_loss / total
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs('saved_models', exist_ok=True)
            torch.save(model.state_dict(), model_save_path)
            
    print(f"Training complete. Best Validation Accuracy: {best_val_acc:.2f}%")

    # ==========================================
    # 6. Final Evaluation on Test Set
    # ==========================================
    print("\nEvaluating on Test Set...")
    model.load_state_dict(torch.load(model_save_path))
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    test_acc = accuracy_score(all_labels, all_preds)
    print(f"Test Accuracy: {test_acc * 100:.2f}%\n")
    
    report = classification_report(all_labels, all_preds, zero_division=0)
    print("Classification Report:")
    print(report)

if __name__ == '__main__':
    main()