import torch
import torch.nn as nn
from ODESolvers.utils.solver import Solver


class EulerSolver(Solver):
    def __init__(self, flowmodel: nn.Module):
        super().__init__(flowmodel)
        self.flow_model = flowmodel

    def step(self, z_t, c, t, dt):
        V_z_t = self.flow_model(z_t,t,c)
        z_t_1 = z_t + V_z_t * dt
        return z_t_1
