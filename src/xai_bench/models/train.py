"""Training loop with AdamW, ReduceLROnPlateau, early stopping, AMP, checkpointing."""
from __future__ import annotations
from pathlib import Path
from typing import Dict


def _make_loader(dataset, batch_size, shuffle, num_workers, seed):
    import torch
    from ..seeds import seed_worker
    g = torch.Generator(); g.manual_seed(seed)
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
        pin_memory=torch.cuda.is_available(), worker_init_fn=seed_worker, generator=g,
    )


def evaluate_logits(model, loader, device):
    """Return (probs[N,2], labels[N]) as numpy arrays."""
    probs, logits, labels = collect_probs_logits(model, loader, device)
    return probs, labels


def collect_probs_logits(model, loader, device):
    """Return (probs[N,2], logits[N,2], labels[N]) as numpy arrays.

    Logits are needed for post-hoc temperature scaling (proposal 3.9.2).
    """
    import numpy as np
    import torch
    model.eval()
    probs, logits, labels = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            lo = model(x)
            probs.append(torch.softmax(lo, dim=1).cpu().numpy())
            logits.append(lo.cpu().numpy())
            labels.append(y.numpy())
    return np.concatenate(probs), np.concatenate(logits), np.concatenate(labels)


def train_model(model, train_ds, val_ds, cfg: Dict, device, class_weight=None,
                ckpt_path: str | Path = "checkpoints/model.pt") -> Dict:
    """Train and return a history dict; best checkpoint saved to ckpt_path."""
    import torch
    import torch.nn as nn

    t = cfg["train"]
    seed = cfg["experiment"]["seed"]
    epochs = 1 if t.get("quick_debug") else int(t["epochs"])
    bs = int(t["batch_size"])
    nw = int(cfg["data"].get("num_workers", 2))

    train_loader = _make_loader(train_ds, bs, True, nw, seed)
    val_loader = _make_loader(val_ds, bs, False, nw, seed)

    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(t["lr"]),
                            weight_decay=float(t["weight_decay"]))
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=2)
    weight = class_weight.to(device) if (class_weight is not None and t.get("class_weighted_loss")) else None
    criterion = nn.CrossEntropyLoss(weight=weight)
    use_amp = bool(t.get("amp")) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_val = float("inf"); patience = int(t.get("early_stopping_patience", 5)); bad = 0
    history = {"train_loss": [], "val_loss": []}
    Path(ckpt_path).parent.mkdir(parents=True, exist_ok=True)

    for ep in range(epochs):
        model.train(); running = 0.0; nb = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            running += loss.item(); nb += 1
            if t.get("quick_debug") and nb >= 3:
                break
        tr_loss = running / max(nb, 1)

        model.eval(); vrun = 0.0; vb = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                vrun += criterion(model(x), y).item(); vb += 1
                if t.get("quick_debug") and vb >= 3:
                    break
        val_loss = vrun / max(vb, 1)
        sched.step(val_loss)
        history["train_loss"].append(tr_loss); history["val_loss"].append(val_loss)
        print(f"[epoch {ep+1}/{epochs}] train_loss={tr_loss:.4f} val_loss={val_loss:.4f}")

        if val_loss < best_val - 1e-4:
            best_val = val_loss; bad = 0
            torch.save({"model_state": model.state_dict(), "epoch": ep, "val_loss": val_loss}, ckpt_path)
        else:
            bad += 1
            if bad >= patience:
                print(f"Early stopping at epoch {ep+1}.")
                break

    if Path(ckpt_path).exists():
        model.load_state_dict(torch.load(ckpt_path, map_location=device)["model_state"])
    history["best_val_loss"] = best_val
    return history
