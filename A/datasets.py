import os, warnings, numpy as np, pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

warnings.filterwarnings("ignore", message=".*weights_only=False.*", category=FutureWarning)

def load_data(config):
    df = pd.read_csv(config.csv_path)
    df["image_path"] = df["image_name"].apply(lambda x: os.path.join(config.data_dir, f"{x}.jpg"))

    if config.use_meta:
        # metadata 
        dummies = pd.get_dummies(df["anatom_site_general_challenge"], dummy_na=True,
                                 dtype=np.uint8, prefix="site")
        df = pd.concat([df, dummies], axis=1)

        df["sex"] = df["sex"].map({"male":1, "female":0})
        df["sex"] = df["sex"].fillna(-1)
        df["age_approx"] = (df["age_approx"] / 90).fillna(0)

        df.drop(columns=["anatom_site_general_challenge"], inplace=True, errors="ignore")
    return df

def make_balanced_augmented(df, random_state=42, factor=5):
    df_pos = df[df["target"] == 1].copy()
    df_pos["aug_dup"] = 0

    dup_list = []
    for _ in range(factor - 1):
        d = df_pos.copy()
        d["aug_dup"] = 1
        dup_list.append(d)
    df_pos_aug = pd.concat(dup_list, ignore_index=True)

    total_pos = len(df_pos) * factor

    df_neg = df[df["target"] == 0].sample(n=total_pos,
                                          random_state=random_state).copy()
    df_neg["aug_dup"] = 0

    df_bal = (pd.concat([df_pos, df_pos_aug, df_neg], axis=0)
                .sample(frac=1, random_state=random_state)
                .reset_index(drop=True))
    return df_bal

def stratified_split(df, test_ratio=0.1, random_state=42):
    train_df, test_df = train_test_split(df, test_size=test_ratio,
                                         stratify=df["target"], random_state=random_state)
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)

def get_transforms(config):
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    # normal argumentation
    train_weak = transforms.Compose([
        transforms.Resize((config.img_size, config.img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # strong argumentation
    train_strong = transforms.Compose([
        transforms.RandomResizedCrop(config.img_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(brightness=0.1, contrast=0.1,
                               saturation=0.05, hue=0.02), 
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.RandomErasing(p=0.5, value='random'),
    ])

    # validation
    val_transform = transforms.Compose([
        transforms.Resize((config.img_size, config.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return train_weak, train_strong, val_transform

class LesionDataset(Dataset):
    """
    若 `aug_dup`==1 且 strong_transform!=None，则使用强增强；否则用 weak_transform。
    """
    def __init__(self, df, weak_transform=None, strong_transform=None,
                 use_meta=False, use_preprocessed=False, tensor_dir=None):
        self.df = df
        self.weak_transform   = weak_transform
        self.strong_transform = strong_transform
        self.use_meta = use_meta
        self.use_preprocessed = use_preprocessed
        self.tensor_dir = tensor_dir

        if use_meta:
            self.meta_cols = [c for c in df.columns
                              if c.startswith("sex") or c.startswith("site_") or c == "age_approx"]
        else:
            self.meta_cols = []

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_name = row["image_name"]
        label = torch.tensor(row["target"], dtype=torch.long)

        # preprocess images
        if self.use_preprocessed:
            sample = torch.load(os.path.join(self.tensor_dir, f"{image_name}.pt"),
                                weights_only=False)
            img = sample["image"]
        else:
            img = Image.open(row["image_path"]).convert("RGB")

        # select transform
        if self.use_preprocessed:
            pass
        else:
            if row.get("aug_dup", 0) == 1 and self.strong_transform:
                img = self.strong_transform(img)
            elif self.weak_transform:
                img = self.weak_transform(img)

        # -------- metadata --------
        if self.use_meta:
            meta = torch.tensor(row[self.meta_cols].values.astype(np.float32))
            return img, meta, label
        else:
            return img, label
