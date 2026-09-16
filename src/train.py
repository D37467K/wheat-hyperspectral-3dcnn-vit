import json
import random
from pathlib import Path

import kagglehub
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from dataset import HyperLeafDataset, META_COLUMNS
from preprocessing import image_lookup, preprocess_cube
from hybrid_model import HybridSpectralCNNViT
from evaluate import evaluate, calculate_metrics, print_test_report

CLASS_COLUMNS = ["Heerup", "Kvium", "Rembrandt", "Sheriff"]

IMAGE_SIZE = 96
BATCH_SIZE = 4
EPOCHS = 30
PATIENCE = 15
SEED = 42

CACHE_DIR = Path("/content/hyperleaf_preprocessed")
OUTPUT_DIR = Path("/content/hyperleaf_output")
CACHE_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def prepare_data():
    hyperleaf2024_path = kagglehub.competition_download("HyperLeaf2024")
    data_dir = Path(hyperleaf2024_path)

    frame = pd.read_csv(data_dir / "train.csv")
    lookup = image_lookup(data_dir / "images")

    frame["image_path"] = frame["ImageId"].map(lookup)
    frame = frame.dropna(subset=["image_path"]).copy().reset_index(drop=True)
    frame["image_path"] = frame["image_path"].astype(str)
    frame["label"] = frame[CLASS_COLUMNS].to_numpy().argmax(axis=1)
    frame["cache_path"] = [
        str(CACHE_DIR / f"{int(x):05d}.npy")
        for x in frame.ImageId
    ]

    print(f"Matched {len(frame)} labelled images")

    for row in tqdm(
        frame.itertuples(),
        total=len(frame),
        desc="Preprocessing",
    ):
        out = Path(row.cache_path)
        if not out.exists():
            np.save(
                out,
                preprocess_cube(
                    row.image_path,
                    image_size=IMAGE_SIZE,
                ),
            )

    train, holdout = train_test_split(
        frame,
        test_size=0.30,
        stratify=frame.label,
        random_state=SEED,
    )
    val, test = train_test_split(
        holdout,
        test_size=0.50,
        stratify=holdout.label,
        random_state=SEED,
    )

    scaler = StandardScaler().fit(train[META_COLUMNS])

    for split in (train, val, test):
        split.loc[:, META_COLUMNS] = scaler.transform(split[META_COLUMNS])

    return train, val, test, scaler

def make_loaders(train, val, test):
    train_loader = DataLoader(
        HyperLeafDataset(train, True),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        HyperLeafDataset(val),
        batch_size=BATCH_SIZE * 2,
        num_workers=2,
        pin_memory=True,
    )
    test_loader = DataLoader(
        HyperLeafDataset(test),
        batch_size=BATCH_SIZE * 2,
        num_workers=2,
        pin_memory=True,
    )
    return train_loader, val_loader, test_loader

def validate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    y_true, y_pred = [], []

    with torch.no_grad():
        for image, meta, target in loader:
            image = image.to(device)
            meta = meta.to(device)
            target = target.to(device)

            with torch.cuda.amp.autocast(
                enabled=device.type == "cuda"
            ):
                output = model(image, meta)
                loss = criterion(output, target)

            pred = output.argmax(dim=1)

            total_loss += loss.item() * target.size(0)
            correct += (pred == target).sum().item()
            total += target.size(0)

            y_true.extend(target.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())

    return (
        total_loss / total,
        correct / total,
        f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
    )

def train_model():
    set_seed(SEED)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("Using:", device)

    train, val, test, scaler = prepare_data()
    train_loader, val_loader, test_loader = make_loaders(
        train, val, test
    )

    counts = np.bincount(train.label, minlength=4)
    weights = torch.tensor(
        len(train) / (4 * counts),
        dtype=torch.float32,
        device=device,
    )

    model = HybridSpectralCNNViT(
        fine_tune_blocks=1
    ).to(device)

    criterion = nn.CrossEntropyLoss(
        weight=weights,
        label_smoothing=0.03,
    )

    optimizer = torch.optim.AdamW(
        filter(
            lambda p: p.requires_grad,
            model.parameters(),
        ),
        lr=1e-4,
        weight_decay=1e-4,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=5,
    )

    scaler_amp = torch.cuda.amp.GradScaler(
        enabled=device.type == "cuda"
    )

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "val_f1": [],
    }

    best_f1, stale = -1.0, 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        loss_sum, correct, total = 0.0, 0, 0

        for image, meta, target in train_loader:
            image = image.to(device)
            meta = meta.to(device)
            target = target.to(device)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(
                enabled=device.type == "cuda"
            ):
                output = model(image, meta)
                loss = criterion(output, target)

            scaler_amp.scale(loss).backward()
            scaler_amp.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler_amp.step(optimizer)
            scaler_amp.update()

            prediction = output.argmax(dim=1)
            loss_sum += loss.item() * target.size(0)
            correct += (prediction == target).sum().item()
            total += target.size(0)

        train_loss = loss_sum / total
        train_acc = correct / total
        val_loss, val_acc, val_f1 = validate(
            model,
            val_loader,
            criterion,
            device,
        )

        scheduler.step(val_f1)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        print(
            f"Epoch {epoch:03d} | "
            f"train loss: {train_loss:.4f} | "
            f"train acc: {train_acc:.4f} | "
            f"val loss: {val_loss:.4f} | "
            f"val acc: {val_acc:.4f} | "
            f"val macro-F1: {val_f1:.4f}"
        )

        if val_f1 > best_f1:
            best_f1, stale = val_f1, 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "scaler": scaler,
                    "classes": CLASS_COLUMNS,
                },
                OUTPUT_DIR / "best_model.pt",
            )
        else:
            stale += 1
            if stale >= PATIENCE:
                print("Early stopping")
                break

    checkpoint = torch.load(
        OUTPUT_DIR / "best_model.pt",
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model"])

    y_test, p_test = evaluate(
        model,
        test_loader,
        device,
    )
    print_test_report(
        y_test,
        p_test,
        CLASS_COLUMNS,
    )

    metrics = calculate_metrics(y_test, p_test)

    for name, value in metrics.items():
        print(f"{name:18s}: {value:.4f}")

    (OUTPUT_DIR / "performance_metrics.json").write_text(
        json.dumps(metrics, indent=2)
    )

    return model, history, metrics

if __name__ == "__main__":
    train_model()
