import os
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import torchvision.transforms as T
import torchvision.transforms.functional as F
import torch


class RSNADataset(Dataset):
    def __init__(self, data_dir, csv_file=None, df=None, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        self.label_cols = [
            "epidural",
            "intraparenchymal",
            "intraventricular",
            "subarachnoid",
            "subdural",
            "any"
        ]

        # Load labels
        if df is not None:
            self.df = df.reset_index(drop=True)
            return

        df = pd.read_csv(csv_file)

        df["id"] = df["ID"].apply(lambda x: x.split("_")[1])
        df["type"] = df["ID"].apply(lambda x: x.split("_")[2])
        df = df.pivot(index="id", columns="type", values="Label")

        df = df[self.label_cols]
        self.df = df.reset_index()

    def __len__(self):
        return len(self.df)


    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = row['id']
        img_path = os.path.join(self.data_dir, f"ID_{img_id}_frame0.png")

        image = Image.open(img_path).convert("L")

        if row.get("is_augmented", False):
            image = T.RandomRotation(degrees=30, fill=0)(image)
            image = T.RandomHorizontalFlip()(image)
            image = T.RandomVerticalFlip()(image)
            image = T.ColorJitter(brightness=0.1, contrast=0.1)(image)

        if self.transform:
            image = self.transform(image)

        labels = torch.tensor(row[self.label_cols].values.astype(float), dtype=torch.float32)
        return image, labels