import os, numpy as np, torch, torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, LinearLR, SequentialLR

from A.train     import train_model, FocalLoss
from A.datasets  import LesionDataset, get_transforms
from A.model     import EfficientNetClassifier

# from amls2final7.amls2folder.A.train     import train_model, FocalLoss
# from amls2final7.amls2folder.A.datasets  import LesionDataset, get_transforms
# from amls2final7.amls2folder.A.model     import EfficientNetClassifier

def cross_validate(train_df, cfg):
    print(f"Starting {cfg.num_folds}-fold CV ...")
    skf = StratifiedKFold(n_splits=cfg.num_folds, shuffle=True, random_state=cfg.random_seed)
    splits = list(skf.split(train_df, train_df.target))
    folds = 1 if cfg.test_mode else cfg.num_folds

    for fold in range(folds):
        print(f"\n── Fold {fold+1}/{folds} ──")
        tr_idx, val_idx = splits[fold]
        df_tr, df_val = train_df.iloc[tr_idx].reset_index(drop=True), train_df.iloc[val_idx].reset_index(drop=True)

        weak, strong, val_tf = get_transforms(cfg)
        tr_set = LesionDataset(df_tr, weak, strong, cfg.use_meta, cfg.use_preprocessed, cfg.tensor_dir)
        val_set = LesionDataset(df_val, val_tf, val_tf, cfg.use_meta, cfg.use_preprocessed, cfg.tensor_dir)

        tr_loader = DataLoader(tr_set, cfg.batch_size, True,  num_workers=cfg.num_workers, pin_memory=True)
        val_loader= DataLoader(val_set, cfg.batch_size, False, num_workers=cfg.num_workers, pin_memory=True)

        meta_dim = len(tr_set.meta_cols) if cfg.use_meta else 0
        model = EfficientNetClassifier(cfg.num_classes, cfg.use_meta, meta_dim,
                                       backbone_name=cfg.backbone,
                                       dropout=cfg.classifier_dropout).to(cfg.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

        # Scheduler
        scheduler = None
        if cfg.use_scheduler:
            if cfg.lr_scheduler_type == "plateau":
                scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.4, patience=2)
            else:
                scheduler = SequentialLR(
                    optimizer,
                    schedulers=[
                        LinearLR(optimizer, start_factor=0.1, total_iters=cfg.warmup_epochs),
                        CosineAnnealingLR(optimizer, T_max=cfg.cosine_epochs)
                    ],
                    milestones=[cfg.warmup_epochs]
                )

        # Loss
        if cfg.use_focal_loss:
            criterion = FocalLoss(alpha=0.25, gamma=2.0)
        else:
            if cfg.use_class_weights:
                cnt = df_tr.target.value_counts().to_dict(); tot = sum(cnt.values())
                w = torch.tensor([tot/(2*cnt[i]) for i in range(2)], dtype=torch.float).to(cfg.device)
            else:
                w = None
            criterion = nn.CrossEntropyLoss(weight=w, label_smoothing=cfg.label_smoothing)

        best_model, _ = train_model(model, optimizer, criterion,
                                    tr_loader, val_loader,
                                    cfg, fold, scheduler)
        save_path = os.path.join(cfg.weights_dir, f"best_model_fold{fold}.pth")
        torch.save(best_model.state_dict(), save_path)
        print("Saved:", save_path)
