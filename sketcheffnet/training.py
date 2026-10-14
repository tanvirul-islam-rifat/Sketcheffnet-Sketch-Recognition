"""
Shared training and evaluation loops.

One :func:`fit` loop covers every experiment. Each experiment keeps the exact
protocol it was run with (scheduler, checkpointing, early stopping), passed in as
arguments, so the shared code does not change any recorded result.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(it, **_):
        return it


def train_one_epoch(model, loader, criterion, optimizer, device, desc="train"):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in tqdm(loader, desc=desc, leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        total += labels.size(0)
        correct += outputs.argmax(1).eq(labels).sum().item()
    return running_loss / max(len(loader), 1), 100.0 * correct / max(total, 1)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        running_loss += criterion(outputs, labels).item()
        total += labels.size(0)
        correct += outputs.argmax(1).eq(labels).sum().item()
    return running_loss / max(len(loader), 1), 100.0 * correct / max(total, 1)


@torch.no_grad()
def predict(model, loader, device):
    """Return (predictions, labels) as NumPy arrays."""
    model.eval()
    preds, labels = [], []
    for images, y in loader:
        preds.append(model(images.to(device)).argmax(1).cpu().numpy())
        labels.append(np.asarray(y))
    return np.concatenate(preds), np.concatenate(labels)


def score(labels, preds) -> tuple[float, float]:
    """(accuracy %, weighted F1 %) - the two metrics every experiment reports."""
    return (accuracy_score(labels, preds) * 100,
            f1_score(labels, preds, average="weighted", zero_division=0) * 100)


def evaluate(model, loader, device) -> tuple[float, float]:
    preds, labels = predict(model, loader, device)
    return score(labels, preds)


def fit(model, train_loader, val_loader, criterion, optimizer, epochs, device, *,
        scheduler=None, scheduler_on_metric=False, checkpoint_path: Path | None = None,
        patience: int | None = None, log_every: int = 1, name: str = "model"):
    """Train for ``epochs`` epochs, validating after each one.

    scheduler_on_metric  call ``scheduler.step(val_acc)`` (ReduceLROnPlateau) instead of ``step()``
    checkpoint_path      save {epoch, model_state_dict, optimizer_state_dict, val_acc, train_acc}
                         whenever validation accuracy improves
    patience             stop after this many epochs without improvement

    Returns a dict: history, best_val_acc, best_epoch (1-based), epochs_run, time_minutes.
    """
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}
    best_val, best_epoch, stale = 0.0, 0, 0
    start = time.time()
    for epoch in range(epochs):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device,
                                          desc=f"{name} {epoch + 1}/{epochs}")
        va_loss, va_acc = validate(model, val_loader, criterion, device)
        if scheduler is not None:
            scheduler.step(va_acc) if scheduler_on_metric else scheduler.step()
        lr = optimizer.param_groups[0]["lr"]
        for k, v in zip(history, (tr_loss, tr_acc, va_loss, va_acc, lr)):
            history[k].append(v)
        if (epoch + 1) % log_every == 0 or epoch == 0:
            print(f"  [{name}] epoch {epoch + 1:>2}/{epochs}  train {tr_loss:.4f} / {tr_acc:.2f}%  "
                  f"val {va_loss:.4f} / {va_acc:.2f}%  lr {lr:.6f}")
        if va_acc > best_val:
            best_val, best_epoch, stale = va_acc, epoch + 1, 0
            if checkpoint_path is not None:
                Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
                torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "val_acc": va_acc, "train_acc": tr_acc}, checkpoint_path)
        else:
            stale += 1
            if patience is not None and stale >= patience:
                print(f"  [{name}] early stopping: {patience} epochs without improvement")
                break
    return {"history": history, "best_val_acc": best_val, "best_epoch": best_epoch,
            "epochs_run": len(history["val_acc"]), "time_minutes": (time.time() - start) / 60}


def load_checkpoint(model, path, device="cpu"):
    """Load a checkpoint written by :func:`fit` (or a bare state_dict) into ``model``."""
    state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state.get("model_state_dict", state) if isinstance(state, dict) else state)
    return model
