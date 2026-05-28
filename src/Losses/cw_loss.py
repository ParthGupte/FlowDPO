import torch
import torch.nn as nn


class NishantLossCW(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a, b):
        return ((a * b).sum(dim=-1) ** 2).mean()