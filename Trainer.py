import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import json
import time
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt)**self.gamma * BCE_loss
        return F_loss.mean()


class Trainer:
    def __init__(self, model, optimizer, criterion, device="cuda", save_dir="checkpoints", scheduler=None):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.scheduler = scheduler

        self.metrics = {
            "train_loss": [], "val_loss": [],
            "train_macro_f1": [], "val_macro_f1": [],
            "train_accuracy": [], "val_accuracy": [],
            "train_hamming_score": [], "val_hamming_score": [],
            "training_time_seconds": 0.0
        }

        self.best_val_loss = np.inf
        self.best_f1 = 0.0
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def _calculate_hamming_score(self, y_true, y_pred):
        intersection = np.logical_and(y_true, y_pred).sum(axis=1)
        union = np.logical_or(y_true, y_pred).sum(axis=1)
        score = np.ones(len(union))
        mask = union > 0
        score[mask] = intersection[mask] / union[mask]

        return np.mean(score)

    def save_checkpoint(self, epoch, is_best=False):
        checkpoint = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "metrics": self.metrics,
            "best_f1": self.best_f1
        }
        torch.save(checkpoint, f"{self.save_dir}/last_checkpoint.pth")
        if is_best:
            torch.save(checkpoint, f"{self.save_dir}/best_model.pth")

    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.metrics = checkpoint["metrics"]
        self.best_f1 = checkpoint["best_f1"]
        start_epoch = checkpoint["epoch"] + 1
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        return start_epoch

    def train_epoch(self, loader, threshold=0.42):
        self.model.train()
        running_loss = 0.0
        all_targets, all_preds = [], []

        progress = tqdm(loader, desc="Training", leave=False)

        for images, labels in progress:
            images, labels = images.to(self.device), labels.to(self.device).float()

            self.optimizer.zero_grad()
            logits = self.model(images)
            loss = self.criterion(logits, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

            probs = torch.sigmoid(logits)
            preds = (probs >= threshold).float()

            all_targets.append(labels.cpu().numpy())
            all_preds.append(preds.cpu().numpy())

        avg_loss = running_loss / len(loader)
        all_targets = np.vstack(all_targets)
        all_preds = np.vstack(all_preds)

        macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
        avg_accuracy = accuracy_score(all_targets, all_preds)
        h_score = self._calculate_hamming_score(all_targets, all_preds)

        return avg_loss, macro_f1, avg_accuracy, h_score

    def evaluate(self, loader, threshold=0.42):
        self.model.eval()
        running_loss = 0.0
        all_targets, all_preds, all_probs = [], [], []

        with torch.no_grad():
            progress = tqdm(loader, desc="Evaluating", leave=False)

            for images, labels in progress:
                images, labels = images.to(self.device), labels.to(self.device).float()
                logits = self.model(images)
                loss = self.criterion(logits, labels)

                running_loss += loss.item()

                probs = torch.sigmoid(logits)

                # Postprocessing
                processed_probs = probs.clone()
                for i in range(len(processed_probs)):
                    any_p = processed_probs[i, 5]

                    # If 'any' is below threshold, force everything to 0
                    if any_p < threshold:
                        processed_probs[i, :] = 0.0
                    else:
                        # If 'any' is 1, at least one subtype MUST be 1.
                        # If none are above threshold, force the most likely one to 1.
                        if torch.max(processed_probs[i, :5]) < threshold:
                            max_idx = torch.argmax(processed_probs[i, :5])
                            processed_probs[i, max_idx] = 1.0

                all_targets.append(labels.cpu().numpy())
                all_preds.append((processed_probs >= threshold).float().cpu().numpy())
                all_probs.append(probs.cpu().numpy())

        avg_loss = running_loss / len(loader)
        all_targets = np.vstack(all_targets)
        all_preds = np.vstack(all_preds)
        all_probs = np.vstack(all_probs)

        # Calculate AUC
        try:
            auc_macro = roc_auc_score(all_targets, all_probs, average="macro")
        except ValueError:
            auc_macro = 0.0

        return {
            "avg_loss": avg_loss,
            "macro_f1": f1_score(all_targets, all_preds, average="macro", zero_division=0),
            "macro_auc": auc_macro,
            "subset_accuracy": accuracy_score(all_targets, all_preds),
            "hamming_score": self._calculate_hamming_score(all_targets, all_preds), # Added here
            "per_label_accuracy": np.mean(all_targets == all_preds, axis=0),
            "overall_accuracy": np.mean(all_targets == all_preds),
            "per_label_precision": precision_score(all_targets, all_preds, average=None, zero_division=0),
            "per_label_recall": recall_score(all_targets, all_preds, average=None, zero_division=0),
            "per_label_f1": f1_score(all_targets, all_preds, average=None, zero_division=0),
            "all_targets": all_targets,
            "all_preds": all_preds,
            "all_probs": all_probs
        }

    def train(self, train_loader, val_loader, num_epochs=20, resume=False):
        start_epoch = 1
        checkpoint_path = f"{self.save_dir}/last_checkpoint.pth"

        # Early Stopping State
        patience = 4
        epochs_no_improve = 0

        if resume and os.path.exists(checkpoint_path):
            start_epoch = self.load_checkpoint(checkpoint_path)

        # Start Training Timer
        start_time = time.time()

        for epoch in range(start_epoch, num_epochs + 1):
            print(f"\n===== EPOCH {epoch}/{num_epochs} =====")

            train_loss, train_f1, train_acc, train_ham = self.train_epoch(train_loader)
            val_stats = self.evaluate(val_loader)

            val_loss = val_stats["avg_loss"]
            val_f1 = val_stats["macro_f1"]
            val_acc = val_stats["subset_accuracy"]
            val_ham = val_stats["hamming_score"]

            print(f"Train Loss: {train_loss:.4f} | Train F1: {train_f1:.4f} | Train Ham: {train_ham:.4f} | Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f} | Val Ham: {val_ham:.4f} | Val Acc: {val_acc:.4f}")

            # Add metrics
            self.metrics["train_loss"].append(train_loss)
            self.metrics["val_loss"].append(val_loss)
            self.metrics["train_macro_f1"].append(train_f1)
            self.metrics["val_macro_f1"].append(val_f1)
            self.metrics["train_accuracy"].append(train_acc)
            self.metrics["val_accuracy"].append(val_acc)
            self.metrics["train_hamming_score"].append(train_ham)
            self.metrics["val_hamming_score"].append(val_ham)

            if self.scheduler is not None:
                self.scheduler.step(val_loss)
                print(f"Current LR: {self.optimizer.param_groups[0]['lr']}")

            # Save Best Model based on F1-Score
            if val_f1 > self.best_f1:
                print(f"New Best F1 Score: {val_f1:.4f}! Saving best_model.pth...")
                self.best_f1 = val_f1
                self.save_checkpoint(epoch, is_best=True)

            # [Uncomment for early stopping based on Val Loss]
            """
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                epochs_no_improve = 0
                print(f"Val Loss improved to {val_loss:.4f}")
            else:
                epochs_no_improve += 1
                print(f"No improvement in Val Loss. ({epochs_no_improve}/{patience})")
            
            # Check if we should stop
            if epochs_no_improve >= patience:
                print(f"EARLY STOPPING triggered at epoch {epoch}. Stopping training.")
                break
            """

            # Save regular checkpoint
            self.save_checkpoint(epoch, is_best=False)

        # End Timer
        total_time = time.time() - start_time
        self.metrics["training_time_seconds"] = total_time
        print(f"\nTotal Training Time: {total_time:.2f} seconds")

        with open(f"{self.save_dir}/metrics.json", "w") as f:
            json.dump(self.metrics, f, indent=4)

    def get_metrics(self):
        return self.metrics