import torch
from torch import nn
from ODESolvers.utils.solver import Solver


class RK4Solver(Solver):
    def __init__(self, flow_model: nn.Module):
        super().__init__(flow_model)
        self.flow_model = flow_model

    def step(self, z_t, c, t, dt):
        k1 = self.flow_model(z_t,t,c)
        k2 = self.flow_model(z_t + 0.5 * dt * k1, (t + 0.5 * dt),c)
        k3 = self.flow_model(z_t + 0.5 * dt * k2, (t + 0.5 * dt),c)
        k4 = self.flow_model(z_t + dt * k3, (t + dt),c)

        z_t_1 = z_t + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return z_t_1
