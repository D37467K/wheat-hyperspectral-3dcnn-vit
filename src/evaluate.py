import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    y_true, y_pred = [], []

    for image, meta, target in loader:
        image = image.to(device)
        meta = meta.to(device)

        output = model(image, meta)
        pred = output.argmax(dim=1)

        y_true.extend(target.cpu().numpy())
        y_pred.extend(pred.cpu().numpy())

    return np.asarray(y_true), np.asarray(y_pred)

def calculate_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "recall_macro": recall_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "f1_macro": f1_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "f1_weighted": f1_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
    }

def print_test_report(y_true, y_pred, class_names):
    print("Held-out test results:")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=class_names,
            digits=4,
            zero_division=0,
        )
    )
    print(
        "Confusion matrix:\n",
        confusion_matrix(
            y_true,
            y_pred,
            labels=np.arange(len(class_names)),
        ),
    )
