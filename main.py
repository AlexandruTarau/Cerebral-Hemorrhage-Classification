import json
import os
import warnings

import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from Preprocess import Preprocess
from MonaiPreprocess import MonaiPreprocess
from RSNADataset import RSNADataset
from BaselineCNN import BaselineCNN
from Trainer import Trainer
from Trainer import FocalLoss
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import seaborn as sns
import shutil
import random
import numpy as np
import torchvision.models as models
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from torch.utils.data import WeightedRandomSampler
from collections import Counter
from torch.utils.data import ConcatDataset
import pandas as pd
import monai


def random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    monai.utils.set_determinism(seed=seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def split_dataset_k_fold(df, config):
    skf = StratifiedKFold(
        n_splits=config['k_folds'],
        shuffle=True,
        random_state=config['seed']
    )

    folds = []
    strat_labels = df["any"].values

    for train_index, val_index in skf.split(df, strat_labels):
        train_df = df.iloc[train_index].reset_index(drop=True)
        val_df = df.iloc[val_index].reset_index(drop=True)
        folds.append((train_df, val_df))

    return folds


def mean_std(values):
    return float(np.mean(values)), float(np.std(values))


def count_labels_in_one_epoch(loader, label_cols):
    counter = Counter()

    for _, targets in loader:
        targets = targets.numpy()
        for y in targets:
            for i, v in enumerate(y):
                if v == 1:
                    counter[label_cols[i]] += 1

    return counter


if __name__ == "__main__":
    # Load config
    with open("config.json", "r") as f:
        CONFIG = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random_seed(CONFIG['seed'])

    # Datasets
    train_dataset = RSNADataset(
        data_dir=CONFIG['train_images'],
        csv_file=CONFIG['train_csv'],
        transform=None
    )

    transform = Preprocess(config_path='config.json', training=False)

    # [Uncomment for Monai augmentations] (Replace the above line)
    """
    transform = MonaiPreprocess("config.json", augmentation_set="basic")
    """

    test_dataset = RSNADataset(
        data_dir=CONFIG['test_images'],
        csv_file=CONFIG['test_csv'],
        transform=transform
    )

    g = torch.Generator()
    g.manual_seed(CONFIG['seed'])

    test_loader = DataLoader(
        test_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=CONFIG['num_workers'],
        worker_init_fn=seed_worker,
        generator=g,
    )

    # Split
    folds = split_dataset_k_fold(train_dataset.df, CONFIG)
    fold_metrics = []

    for idx, (train_df, val_df) in enumerate(folds):
        print(f"\n===== FOLD {idx + 1} / {CONFIG['k_folds']} =====")

        # RESEED for determinism
        random_seed(CONFIG['seed'])

        # Datasets
        transform = Preprocess(config_path='config.json', training=True)

        # [Uncomment for Monai augmentations] (Replace the above line)
        """
        transform = MonaiPreprocess("config.json", augmentation_set="hybrid")
        """

        # [Uncomment for oversampling minority classes]
        """
        # Identify the minority classes
        minority_mask = (train_df['epidural'] == 1) | (train_df['intraventricular'] == 1)
        minority_rows = train_df[minority_mask].copy()
        minority_ids = minority_rows["id"].unique()

        # Debug prints
        total_orig_imgs = train_df["id"].nunique() # Changed from image_id to id
        total_min_imgs = len(minority_ids)
        print(f"[DEBUG] Original Training Images: {total_orig_imgs}")
        print(f"[DEBUG] Minority Images Found (Epidural/Intraventricular): {total_min_imgs}")
        print(f"[DEBUG] New Augmented Rows added: {len(minority_rows)}")

        # Add 'is_augmented' flag (True = augmented, False = original)
        minority_rows["is_augmented"] = True

        # Combine the original and augmented data
        combined_train_df = pd.concat([train_df, minority_rows], ignore_index=True)
        
        train_dataset = RSNADataset(
            data_dir=CONFIG['train_images'],
            df=combined_train_df,
            transform=transform
        )
        """

        train_dataset = RSNADataset(
            data_dir=CONFIG['train_images'],
            df=train_df,
            transform=transform
        )

        transform = Preprocess(config_path='config.json', training=False)

        # Uncomment for Monai augmentations (Replace the above line)
        """
        transform = MonaiPreprocess("config.json", augmentation_set="basic")
        """

        val_dataset = RSNADataset(
            data_dir=CONFIG['train_images'],
            df=val_df,
            transform=transform
        )

        # Dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=CONFIG['batch_size'],
            shuffle=True,
            num_workers=CONFIG['num_workers'],
            worker_init_fn=seed_worker,
            generator=g,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=CONFIG['batch_size'],
            shuffle=False,
            num_workers=CONFIG['num_workers'],
            worker_init_fn=seed_worker,
            generator=g,
        )

        # DEBUG prints for class distribution
        label_distribution = count_labels_in_one_epoch(
            train_loader,
            train_dataset.label_cols
        )

        print("Class distributions after oversampling:")
        for k, v in label_distribution.items():
            print(f"{k}: {v}")

        # Model and trainer (Choose between Resnet18 and EfficientNet)

        # Resnet18
        #"""
        model = models.resnet18(weights=None)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.fc = nn.Linear(model.fc.in_features, CONFIG['model']['num_classes'])
        model = model.to(device)
        #"""

        # EfficientNet
        """
        model = models.efficientnet_b0(weights='IMAGENET1K_V1')
        old_conv = model.features[0][0]
        model.features[0][0] = nn.Conv2d(
            in_channels=1,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False
        )
        
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, CONFIG['model']['num_classes'])
        model = model.to(device)
        """

        # [Uncomment to Freeze Backbone for EfficientNet]
        """
        # Freeze backbone
        for param in model.parameters():
            param.requires_grad = False
        
        # Unfreeze the input layer
        for param in model.features[0][0].parameters():
            param.requires_grad = True
        
        # Unfreeze the classifier
        for param in model.classifier[1].parameters():
            param.requires_grad = True
        """


        pos_weights = None

        # [Uncomment for Loss with pos weights]
        """
        labels = train_df[train_dataset.label_cols].values
        class_counts = labels.sum(axis=0)
        print(class_counts)

        pos_weights = (len(labels) - class_counts) / class_counts
        print(pos_weights)

        pos_weights = torch.tensor(pos_weights, dtype=torch.float32).to(device)
        """

        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weights)

        # [Uncomment for FocalLoss] (Replace the above line)
        """
        criterion = FocalLoss(alpha=0.5, gamma=2.0)
        """

        # Adam Optimizer
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=CONFIG['learning_rate'],
            weight_decay=CONFIG['weight_decay']
        )

        # SGD Optimizer
        """
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=CONFIG['learning_rate'],
            momentum=0.9,
            weight_decay=CONFIG['weight_decay']
        )
        """

        # RMSprop Optimizer
        """
        optimizer = torch.optim.RMSprop(
            model.parameters(),
            lr=CONFIG['learning_rate'],
            alpha=0.99,
            eps=1e-08,
            weight_decay=CONFIG['weight_decay']
        )
        """

        scheduler = None

        # [Uncomment for LR Scheduler]
        """
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.1,
            patience=2
        )
        """

        fold_save_dir = os.path.join(CONFIG['save_dir'], f"fold_{idx + 1}")
        os.makedirs(fold_save_dir, exist_ok=True)

        trainer = Trainer(model, optimizer, criterion, device, save_dir=fold_save_dir, scheduler=scheduler)

        resume_fold = False
        trainer.train(train_loader, val_loader, num_epochs=CONFIG['num_epochs'], resume=resume_fold)

        val_stats = trainer.evaluate(val_loader)

        test_stats = trainer.evaluate(test_loader)

        fold_metrics.append({
            "fold": idx + 1,

            "val_accuracy": val_stats["subset_accuracy"],
            "val_auc": val_stats["macro_auc"],
            "val_hamming": val_stats["hamming_score"],
            "val_precision": np.mean(val_stats["per_label_precision"]),
            "val_recall": np.mean(val_stats["per_label_recall"]),
            "val_f1": val_stats["macro_f1"],

            "test_accuracy": test_stats['subset_accuracy'],
            "test_auc": test_stats['macro_auc'],
            "test_hamming": test_stats['hamming_score'],
            "test_precision": np.mean(test_stats['per_label_precision']),
            "test_recall": np.mean(test_stats['per_label_recall']),
            "test_f1": test_stats['macro_f1']
        })

    summary = {
        "val_accuracy": mean_std([f["val_accuracy"] for f in fold_metrics]),
        "val_auc": mean_std([f["val_auc"] for f in fold_metrics]),
        "val_hamming": mean_std([f["val_hamming"] for f in fold_metrics]),
        "val_precision": mean_std([f["val_precision"] for f in fold_metrics]),
        "val_recall": mean_std([f["val_recall"] for f in fold_metrics]),
        "val_f1": mean_std([f["val_f1"] for f in fold_metrics]),

        "test_accuracy": mean_std([f["test_accuracy"] for f in fold_metrics]),
        "test_auc": mean_std([f["test_auc"] for f in fold_metrics]),
        "test_hamming": mean_std([f["test_hamming"] for f in fold_metrics]),
        "test_precision": mean_std([f["test_precision"] for f in fold_metrics]),
        "test_recall": mean_std([f["test_recall"] for f in fold_metrics]),
        "test_f1": mean_std([f["test_f1"] for f in fold_metrics]),
    }

    final_results = {
        "per_fold": fold_metrics,
        "summary": summary
    }

    with open(os.path.join(CONFIG["save_dir"], "k_fold_results.json"), "w") as f:
        json.dump(final_results, f, indent=4)