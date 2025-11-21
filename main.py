import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt

plt.switch_backend("Agg")

from data import create_kfold_splits, get_dataloader
from model import CNN


def plot_learning_curves(history, fold_idx, save_dir="results"):
    os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(12, 5))

    # 绘制 Loss 曲线
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.title(f"Fold {fold_idx + 1} Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    # 绘制 Accuracy 曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_acc"], label="Train Acc")
    plt.plot(epochs, history["val_acc"], label="Val Acc")
    plt.title(f"Fold {fold_idx + 1} Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    save_path = os.path.join(save_dir, f"fold_{fold_idx + 1}.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved learning curves to {save_path}")


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss, total_correct, total_samples = 0, 0, 0

    for images, labels in tqdm(dataloader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        total_correct += preds.eq(labels).sum().item()
        total_samples += labels.size(0)

    return total_loss / total_samples, total_correct / total_samples


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss, total_correct, total_samples = 0, 0, 0

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validating", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            total_correct += preds.eq(labels).sum().item()
            total_samples += labels.size(0)

    return total_loss / total_samples, total_correct / total_samples


def train(data_root, epochs, batch_size, lr, num_folds=10):
    print(f"\n------ 训练开始 ------")

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU for training\n")
    else:
        device = torch.device("cpu")
        print("Using CPU for training\n")

    create_kfold_splits(data_root)

    transform = {
        "train": transforms.Compose(
            [
                transforms.RandomResizedCrop(224, scale=(0.5, 1.0), interpolation=3),
                transforms.RandomHorizontalFlip(),
                # transforms.TrivialAugmentWide(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
                # transforms.RandomErasing(
                #     p=0.5,
                #     scale=(0.02, 0.2),
                #     ratio=(0.3, 3.0),
                #     value="random",
                #     inplace=True,
                # ),
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        ),
    }

    fold_results = []

    for fold_idx in range(num_folds):
        print(f"\n------ 启动 Fold {fold_idx + 1} ------")

        train_loader, val_loader = get_dataloader(
            data_root=data_root,
            fold_idx=fold_idx,
            transform=transform,
            batch_size=batch_size,
            pin_memory=True,
        )

        # 模型、损失函数、优化器
        model = CNN(num_classes=200).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
        best_val_acc = 0.0
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")

            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )
            val_loss, val_acc = validate(model, val_loader, criterion, device)
            best_val_acc = max(best_val_acc, val_acc)

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")

        scheduler.step()
        fold_results.append(best_val_acc)
        plot_learning_curves(history, fold_idx)
        print(f"Best Val Acc: {best_val_acc:.4f}")

    print("\n------ 最终结果 ------")
    print(f"Average Acc: {sum(fold_results)/len(fold_results):.4f}")
    print(f"Best    Acc: {max(fold_results):.4f}\n")


if __name__ == "__main__":
    data_root = "data/CUB_200_2011"
    epochs = 100
    batch_size = 256
    lr = 1e-3
    num_folds = 1

    train(
        data_root=data_root,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        num_folds=num_folds,
    )
