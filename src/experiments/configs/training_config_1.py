import torch
from ODESolvers.euler_solver import EulerSolver
from ODESolvers.rangakutta4_solver import RK4Solver

DATA_PATH = ".src/data"
BATCH_SIZE = 128
NUM_WORKERS = 10
CHECK_VAL_EVERY_N_EPOCHS = 1
DEVICES = [0]
MAX_EPOCHS = 100
ACCUMULATE_GRAD_BATCHES = 1
IMAGE_SIZE = 28
LATENT_SIZE = (1,28,28)
OPTIMIZER_CLASS = torch.optim.Adam
LR = 1e-4
LOSS_FN_CLASS = torch.nn.MSELoss
SOLVER_CLASS_FWD = RK4Solver
SOLVER_STEPS = 10
SAVE_N_IMAGES = 10
NUM_CLASS_EMBEDS = 10