import os, copy, numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score
)
import matplotlib.pyplot as plt
from efficientnet_pytorch import EfficientNet
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR

class EfficientNetClassifier(nn.Module):
    def __init__(self, num_classes, use_meta=False,
                 meta_input_dim=0, backbone_name="efficientnet-b4"):
        super().__init__()
        self.use_meta = use_meta

        self.backbone = EfficientNet.from_pretrained(backbone_name)
        in_features = self.backbone._fc.in_features
        self.backbone._fc = nn.Identity()

        if self.use_meta and meta_input_dim > 0:
            self.meta_net = nn.Sequential(
                nn.Linear(meta_input_dim, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(inplace=True),
            )
            in_features += 64
        else:
            self.meta_net = None

        self.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x, x_meta=None):
        x = self.backbone.extract_features(x)
        x = F.adaptive_avg_pool2d(x, 1).squeeze(-1).squeeze(-1)
        if self.use_meta and self.meta_net is not None and x_meta is not None:
            x = torch.cat([x, self.meta_net(x_meta)], dim=1)
        return self.classifier(x)

# Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        if isinstance(alpha, (list, tuple)):
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
        else:
            self.alpha = torch.tensor([alpha, 1.0 - alpha], dtype=torch.float32)
        self.gamma = gamma
        self.reduction = reduction
        self.nll_loss = nn.NLLLoss(weight=self.alpha, reduction=self.reduction)

    def forward(self, logits, target):
        if logits.device != self.nll_loss.weight.device:
            self.nll_loss.weight = self.nll_loss.weight.to(logits.device)
        probs = F.softmax(logits, dim=1)
        log_p = torch.log(probs)
        focal_weight = (1 - probs) ** self.gamma
        focal_log_p = focal_weight * log_p
        loss = self.nll_loss(focal_log_p, target)
        return loss

# training function
def train_model(model, optimizer, criterion,
                train_loader, val_loader, config,
                fold=None, scheduler=None, writer=None):
    best_score = -float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0

    train_losses = []
    val_losses = []
    val_scores = []
    lrs = []

    for epoch in range(config.epochs):
        model.train()
        running_loss = 0.0

        loader_iter = tqdm(train_loader, desc=f"Fold {fold+1} Epoch {epoch+1}/{config.epochs}") if config.use_tqdm else train_loader

        for batch in loader_iter:
            if config.use_meta:
                images, meta, labels = batch
                images, meta, labels = images.to(config.device), meta.to(config.device), labels.to(config.device)
                outputs = model(images, meta)
            else:
                images, labels = batch
                images, labels = images.to(config.device), labels.to(config.device)
                outputs = model(images)

            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * labels.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(train_loss)

        # validation
        model.eval()
        val_running_loss, all_labels, all_probs = 0.0, [], []
        with torch.no_grad():
            for batch in val_loader:
                if config.use_meta:
                    images, meta, labels = batch
                    images, meta, labels = images.to(config.device), meta.to(config.device), labels.to(config.device)
                    outputs = model(images, meta)
                else:
                    images, labels = batch
                    images, labels = images.to(config.device), labels.to(config.device)
                    outputs = model(images)

                val_running_loss += criterion(outputs, labels).item() * labels.size(0)
                probs = F.softmax(outputs, dim=1)[:, 1]
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_loss = val_running_loss / len(val_loader.dataset)
        val_losses.append(val_loss)
        all_probs, all_labels = np.array(all_probs), np.array(all_labels)

        try:
            val_auc = roc_auc_score(all_labels, all_probs)
        except ValueError:
            val_auc = 0.0
        val_scores.append(val_auc)

        val_preds = (all_probs >= 0.5).astype(int)
        val_acc = accuracy_score(all_labels, val_preds)
        current_lr = optimizer.param_groups[0]['lr']
        lrs.append(current_lr)

        print(f"[Fold {fold+1}] Epoch {epoch+1} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, AUC: {val_auc:.4f}, Acc: {val_acc:.4f}, LR: {current_lr:.6f}")

        if val_auc > best_score:
            best_score = val_auc
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if config.use_scheduler and scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_auc)
            else:
                scheduler.step()

        if epochs_no_improve >= config.patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    model.load_state_dict(best_model_wts)
    fold_name = f"fold{fold+1}" if fold is not None else "full"
    save_training_curves(train_losses, val_losses, val_scores, fold_name, config.images_dir)
    save_lr_curve(lrs, fold_name, config.images_dir)
    return model, best_score

# validation and save
def save_training_curves(train_losses, val_losses, val_scores, fold_name, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    epochs_range = range(1, len(train_losses)+1)

    plt.figure()
    plt.plot(epochs_range, train_losses, label="Train Loss")
    plt.plot(epochs_range, val_losses, label="Val Loss")
    plt.title(f"Loss ({fold_name})")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(save_dir, f"loss_curve_{fold_name}.png"))
    plt.close()

    plt.figure()
    plt.plot(epochs_range, val_scores, label="Val AUC")
    plt.title(f"AUC ({fold_name})")
    plt.xlabel("Epoch")
    plt.ylabel("AUC")
    plt.legend()
    plt.savefig(os.path.join(save_dir, f"auc_curve_{fold_name}.png"))
    plt.close()

def save_lr_curve(lrs, fold_name, save_dir):
    plt.figure()
    plt.plot(range(1, len(lrs)+1), lrs, label="LR")
    plt.title(f"Learning Rate Schedule ({fold_name})")
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.savefig(os.path.join(save_dir, f"lr_curve_{fold_name}.png"))
    plt.close()
