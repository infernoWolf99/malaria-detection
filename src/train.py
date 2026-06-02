import gc
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm import tqdm

class ModelTrainer:
    """
    Manages the execution pipeline for training, validating, and
    tracking performance metrics of a malaria detection model.
    """

    def __init__(
        self, model: nn.Module, train_loader, val_loader, lr: float = 0.005
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader

        # optimizer
        self.optimizer = torch.optim.SGD(
            self.model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005
        )
        self.scaler = GradScaler(enabled=(self.device.type == "cuda"))

        # Metric state tracking
        self.results = []

    def run_epoch(self, epoch_idx: int) -> dict:
        """Runs one full training and validation pass."""
        print(f"\n➔ Epoch [{epoch_idx + 1}]")

        # training loss
        avg_train_loss = self._train_one_epoch()
        print(f"[Train Complete] Average Loss: {avg_train_loss:.4f}")
        # validation loss
        avg_val_loss = self._validate_loss()
        print(f"   [Val Loss Complete] Val Loss:  {avg_val_loss:.4f}")
        # validation mAP
        mAP_50 = self._validate_one_epoch()
        print(f"[Val Complete]   mAP@0.50:     {mAP_50:.4f}")

        epoch_summary = {
            "epoch": epoch_idx + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "mAP_50": mAP_50,
        }
        self.results.append(epoch_summary)
        return epoch_summary

    def _train_one_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        pbar = tqdm(self.train_loader, desc="   Training", leave=False)

        for images, targets in pbar:
            # Shift payload components cleanly to the targeted hardware
            images = [img.to(self.device) for img in images]
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

            self.optimizer.zero_grad()

            # Forward pass using Mixed Precision (AMP)
            with autocast(device_type=self.device.type):
                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

            # Backpropagation via Scaled Gradient
            self.scaler.scale(losses).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += losses.item()
            pbar.set_postfix({"batch_loss": f"{losses.item():.4f}"})

        return total_loss / len(self.train_loader)

    def _validate_loss(self) -> float:
        """
        Computes validation loss by forcing the model to calculate
        losses without updating weights.
        """
        self.model.train()
        total_val_loss = 0.0

        pbar = tqdm(self.val_loader, desc="   Val Loss", leave=False)

        with torch.no_grad():
            for images, targets in pbar:
                images = [img.to(self.device) for img in images]
                targets = [
                    {k: v.to(self.device) for k, v in t.items()} for t in targets
                ]

                with autocast(device_type=self.device.type):
                    loss_dict = self.model(images, targets)
                    losses = sum(loss for loss in loss_dict.values())

                total_val_loss += losses.item()

        return total_val_loss / len(self.val_loader)

    def _validate_one_epoch(self) -> float:
        self.model.eval()
        metric_evaluator = MeanAveragePrecision(iou_type="bbox")
        pbar = tqdm(self.val_loader, desc="   Validating", leave=False)

        with torch.no_grad():
            for images, targets in pbar:
                images = [img.to(self.device) for img in images]

                with autocast(device_type=self.device.type):
                    predictions = self.model(images)

                cpu_preds = [{k: v.cpu() for k, v in p.items()} for p in predictions]
                cpu_targets = [{k: v.cpu() for k, v in t.items()} for t in targets]
                metric_evaluator.update(cpu_preds, cpu_targets)

        final_metrics = metric_evaluator.compute()
        return final_metrics["map_50"].item()

    def cleanup(self):
        """Purges hardware registers and resets Python cache explicitly."""
        del self.model, self.optimizer, self.scaler
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
