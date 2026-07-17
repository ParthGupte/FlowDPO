import os
# training_step's double-backward (create_graph=True through both transformer
# and ref_transformer) drives the allocator close to the GPU's full capacity,
# where PyTorch's caching allocator fragments into blocks too small to satisfy
# a single large allocation even though enough total memory is reserved.
# Must be set before any CUDA context/allocator initialization.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from datamodule.DPO_loader import FlowDPOSD3DataModule
import lightning as L
import torch
from experiments.configs import training_config_3_dpo as training_config_dpo, model_config_1 as model_config_dpo
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from inference_tests.utils.load_sd3_model import load_lora_weights
from models.sd3 import pipe
import sys
from seed_all import set_seed


seed_value = 42
L.seed_everything(seed_value)
set_seed(seed_value)

torch.set_float32_matmul_precision('high')

def use_double_backward_safe_attention():
    # FlowDPOLossMultiSample backprops through the transformer twice
    # (autograd.grad with create_graph=True), which flash/mem-efficient SDPA
    # kernels don't support, so training needs the math backend. Validation
    # only does a single forward pass and OOMs under math attention's full
    # O(N^2) attention matrix at 1024x1024, so it should keep the default
    # (flash/mem-efficient) backends.
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)

cont = len(sys.argv) > 1 and sys.argv[1] == 'cont'

if cont:
    if len(sys.argv) < 4:
        raise ValueError("Usage: python experiment3_dpo.py cont <wandb_run_id> <ckpt_path>")
    run_id = sys.argv[2]
    ckpt_path = sys.argv[3]
    wandb_logger = WandbLogger(project="flow_dpo_training_logs",resume="allow",id=run_id)
else:
    wandb_logger = WandbLogger(project="flow_dpo_training_logs")

data_module = FlowDPOSD3DataModule(
    data_path = training_config_dpo.DATA_PATH,
    batch_size = training_config_dpo.BATCH_SIZE,
    val_batch_size = training_config_dpo.VAL_BATCH_SIZE,
    num_workers = training_config_dpo.NUM_WORKERS,
)

checkpoint_callback_avg_loss = ModelCheckpoint(
    monitor="train_epoch_avg_loss",  # The metric to monitor (you log this yourself)
    mode="min",  # Lower loss is better
    save_top_k=1,  # Keep top 3 checkpoints
    filename="FlowDPOSD3-{epoch:02d}-{train_epoch_avg_loss:.4f}",
    # only state_dict (already filtered to transformer/LoRA weights in
    # on_save_checkpoint) is worth persisting -- skip optimizer/scheduler state.
    save_weights_only=True,
    every_n_epochs=1
)

checkpoint_callback_FID = ModelCheckpoint(
    monitor="FID",  # The metric to monitor (you log this yourself)
    mode="min",  # Lower loss is better
    save_top_k=3,  # Keep top 3 checkpoints
    filename="FlowDPOSD3-{epoch:02d}-{FID:.4f}",
    save_weights_only=True,
    every_n_epochs=training_config_dpo.CHECK_VAL_EVERY_N_EPOCHS
)

trainer = L.Trainer(
    devices=training_config_dpo.DEVICES,
    max_epochs=training_config_dpo.MAX_EPOCHS,
    accumulate_grad_batches=training_config_dpo.ACCUMULATE_GRAD_BATCHES,
    logger=wandb_logger,
    check_val_every_n_epoch=training_config_dpo.CHECK_VAL_EVERY_N_EPOCHS,
    callbacks=[checkpoint_callback_avg_loss, checkpoint_callback_FID],
    limit_val_batches=20,
    fast_dev_run=True
)

flowmodel_dpo = training_config_dpo.LOAD_FN(pipe, model_config_dpo, training_config_dpo)

if cont:
    # there's no point validating freshly-initialized weights beforehand, since
    # the checkpoint's trained weights get loaded either way below.
    use_double_backward_safe_attention()
    if training_config_dpo.USE_LORA:
        # LoRA checkpoints (save_weights_only=True, state_dict filtered to
        # "lora_" keys in on_save_checkpoint) carry no optimizer/epoch state to
        # resume, so load the adapter weights directly and start a fresh
        # trainer.fit() rather than going through ckpt_path='s full restore.
        load_lora_weights(flowmodel_dpo, ckpt_path)
        trainer.fit(flowmodel_dpo,datamodule=data_module)
    else:
        # full fine-tune checkpoints (transformer.* weights) still resume
        # via Lightning's own restore.
        trainer.fit(flowmodel_dpo,datamodule=data_module,ckpt_path=ckpt_path)
else:
    trainer.validate(flowmodel_dpo,datamodule=data_module)
    use_double_backward_safe_attention()
    trainer.fit(flowmodel_dpo,datamodule=data_module)
