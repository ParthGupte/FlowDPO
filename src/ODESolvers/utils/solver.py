import torch
import torch.nn as nn
from abc import abstractmethod


class Solver:
    def __init__(self, flow_model):
        self.flow_model = flow_model
        pass

    @abstractmethod
    def step(self, z_t, c, t, dt):
        pass

    def solve(self, X_0, c, n_steps):
        time_steps = torch.linspace(0, 1, n_steps + 1, device=self.flow_model.device)
        dt = time_steps[1] - time_steps[0]
        z_t = X_0
        for t in time_steps[:-1]:
            z_t = self.step(z_t ,c, t, dt)
        return z_t
    
    def solve_reverse(self,X_1, c, n_steps):
        time_steps = torch.linspace(1, 0, n_steps + 1, device=self.flow_model.device)
        dt = time_steps[1] - time_steps[0]
        z_t = X_1
        for t in time_steps[:-1]:
            z_t = self.step(z_t ,c, t, dt)
        return z_t