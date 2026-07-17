import torch
from ODESolvers.rangakutta4_solver import RK4Solver
from Losses.dpo_flow import FlowDPOLoss, FlowDPOLossMultiSample, FlowDPOLossWallace
from inference_tests.utils.load_sd3_model import load_sd3_model_wallace

DATA_PATH = "data/hpdv3_pairs.pt"
BATCH_SIZE = 1  # TODO: restore to 10 once GPU memory allows (see training_step OOM debugging)
# validation runs a full SD3 generation + HPSv3 reward pass per image (much
# more GPU memory per example than a training step), so it needs its own,
# smaller batch size
VAL_BATCH_SIZE = 10
NUM_WORKERS = 10
CHECK_VAL_EVERY_N_EPOCHS = 1
DEVICES = [0]
MAX_EPOCHS = 100
ACCUMULATE_GRAD_BATCHES = 3
IMAGE_SIZE = 1024
LATENT_SIZE = (1,28,28)
OPTIMIZER_CLASS = torch.optim.Adam
LR = 1e-5
LOSS_FN_CLASS = FlowDPOLossWallace
SOLVER_CLASS_FWD = RK4Solver
SOLVER_STEPS = 10
SAVE_N_IMAGES = 10
NUM_CLASS_EMBEDS = 10
LOSS_KWARGS = {'N':100}

# which model-building function experiment3_dpo.py should call; swap to
# load_sd3_model (plain FlowDPOSD3) or another loader to change model classes
# without touching the experiment script.
LOAD_FN = load_sd3_model_wallace

USE_LORA = True
LORA_RANK = 16
LORA_ALPHA = 16
LORA_DROPOUT = 0.0
# peft matches these by suffix against the transformer's submodule names;
# attention projections only (standard SD3 LoRA target set).
LORA_TARGET_MODULES = ["to_q", "to_k", "to_v", "to_out.0"]