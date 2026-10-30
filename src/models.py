"""Modelos recurrentes LSTM y GRU para pronóstico de consumo a corto plazo."""
import torch
from torch import nn

import config as C


class RNNForecaster(nn.Module):
    def __init__(self, cell: str = "lstm", n_inputs: int = C.N_INPUTS, hidden: int = C.HIDDEN,
                 num_layers: int = C.NUM_LAYERS, dropout: float = C.DROPOUT):
        super().__init__()
        self.cell = cell
        rnn_cls = {"lstm": nn.LSTM, "gru": nn.GRU}[cell]
        self.rnn = rnn_cls(n_inputs, hidden, num_layers=num_layers, batch_first=True,
                           dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):                    # x: (B, LOOKBACK, n_inputs)
        out, _ = self.rnn(x)
        return self.head(out[:, -1]).squeeze(-1)   # consumo medio escalado del próximo bloque


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def flops_per_forecast(cell: str, n_in=C.N_INPUTS, h=C.HIDDEN, layers=C.NUM_LAYERS, T=C.LOOKBACK) -> int:
    """FLOPs aproximados (multiplicación+suma = 2) de un pronóstico."""
    gates = {"lstm": 4, "gru": 3}[cell]
    total, inp = 0, n_in
    for _ in range(layers):
        total += T * gates * 2 * (inp * h + h * h + h)
        inp = h
    total += 2 * (h * 32 + 32)
    return int(total)
