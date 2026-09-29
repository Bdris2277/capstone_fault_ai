from __future__ import annotations


def build_fault_mlp(*, input_size: int, num_classes: int = 5):
    try:
        import torch.nn as nn
        import torch.nn.functional as F
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError(
            "PyTorch is required. Install torch on your Jetson (JetPack-specific build)."
        ) from exc

    class FaultMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(input_size, 64)
            self.dropout1 = nn.Dropout(0.2)
            self.fc2 = nn.Linear(64, 32)
            self.dropout2 = nn.Dropout(0.2)
            self.fc3 = nn.Linear(32, num_classes)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = self.dropout1(x)
            x = F.relu(self.fc2(x))
            x = self.dropout2(x)
            return self.fc3(x)

    return FaultMLP()

