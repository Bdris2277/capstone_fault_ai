from __future__ import annotations

import numpy as np


def train_one_epoch(*, model, train_loader, criterion, optimizer, device) -> tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += float(loss.item())
        _, predicted = outputs.max(1)
        total += int(labels.size(0))
        correct += int((predicted == labels).sum().item())

    return running_loss / max(len(train_loader), 1), (100.0 * correct / max(total, 1))


def predict_classes(*, model, loader, device) -> np.ndarray:
    model.eval()
    all_preds: list[int] = []

    import torch

    with torch.no_grad():
        for inputs, _labels in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            all_preds.extend([int(x) for x in predicted.cpu().numpy().tolist()])

    return np.asarray(all_preds, dtype=np.int64)

