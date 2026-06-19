import torch
from ODESolvers.rangakutta4_solver import RK4Solver
from Losses.dpo_flow import FlowDPOLoss, FlowDPOLossMultiSample, FlowDPOLossWallace

DATA_PATH = "data/hpdv3_pairs.pt"
BATCH_SIZE = 10
NUM_WORKERS = 2
CHECK_VAL_EVERY_N_EPOCHS = 1
DEVICES = [1]
MAX_EPOCHS = 100
ACCUMULATE_GRAD_BATCHES = 1
IMAGE_SIZE = 28
LATENT_SIZE = (1,28,28)
OPTIMIZER_CLASS = torch.optim.Adam
LR = 1e-5
LOSS_FN_CLASS = FlowDPOLossMultiSample
SOLVER_CLASS_FWD = RK4Solver
SOLVER_STEPS = 10
SAVE_N_IMAGES = 10
NUM_CLASS_EMBEDS = 10