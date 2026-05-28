from datamodule.DPO_loader import FlowDPODataModule
import lightning as L
import torch
from diffusers import UNet2DModel
from experiments.configs import training_config_1 as training_config_ref, training_config_1_dpo as training_config_dpo
from lightning.pytorch.loggers import WandbLogger
from models.flow_dpo_module import FlowDPOModule
from models.unet_flow import FlowModel
from lightning.pytorch.callbacks import ModelCheckpoint
import sys
from seed_all import set_seed
from experiments.configs import model_config_1 as model_config_ref, model_config_1 as model_config_dpo
import copy
from lightning.pytorch.profilers import AdvancedProfiler


seed_value = 42
L.seed_everything(seed_value)
set_seed(seed_value)

torch.set_float32_matmul_precision('high')

torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

if len(sys.argv) > 1:
    if sys.argv[1] == 'cont':
        run_id = "ip7m8d1g"
        wandb_logger = WandbLogger(project="flow_dpo_training_logs",resume="allow",id=run_id)
else:
    wandb_logger = WandbLogger(project="flow_dpo_training_logs")

data_module = FlowDPODataModule(
    data_path = training_config_dpo.DATA_PATH,
    batch_size = training_config_dpo.BATCH_SIZE,
    num_workers = training_config_dpo.NUM_WORKERS,
)

checkpoint_callback_avg_loss = ModelCheckpoint(
    monitor="train_epoch_avg_loss",  # The metric to monitor (you log this yourself)
    mode="min",  # Lower loss is better
    save_top_k=1,  # Keep top 3 checkpoints
    filename="FlowModel_dpo-{epoch:02d}-{train_epoch_avg_loss:.4f}",
    save_weights_only=False,
    every_n_epochs=1
)

checkpoint_callback_FID = ModelCheckpoint(
    monitor="FID",  # The metric to monitor (you log this yourself)
    mode="min",  # Lower loss is better
    save_top_k=3,  # Keep top 3 checkpoints
    filename="FlowModel_dpo-{epoch:02d}-{FID:.4f}",
    save_weights_only=False,
    every_n_epochs=training_config_dpo.CHECK_VAL_EVERY_N_EPOCHS
)

# profiler = AdvancedProfiler(dirpath="./src/profiles", filename="profiler.txt")

trainer = L.Trainer(
    devices=training_config_dpo.DEVICES,
    max_epochs=training_config_dpo.MAX_EPOCHS,
    accumulate_grad_batches=training_config_dpo.ACCUMULATE_GRAD_BATCHES,
    logger=wandb_logger,
    check_val_every_n_epoch=training_config_dpo.CHECK_VAL_EVERY_N_EPOCHS,
    callbacks=[checkpoint_callback_avg_loss,checkpoint_callback_FID],
    # profiler=profiler,
    fast_dev_run=False
)

unet = UNet2DModel(
    sample_size = training_config_ref.LATENT_SIZE,
    in_channels = model_config_ref.IN_CHANNELS,
    out_channels = model_config_ref.OUT_CHANNELS,
    down_block_types = model_config_ref.DOWN_BLOCK_TYPES,
    up_block_types = model_config_ref.UP_BLOCK_TYPES,
    block_out_channels = model_config_ref.BLOCK_OUT_CHANNELS,
    layers_per_block = model_config_ref.LAYERS_PER_BLOCK,
    num_class_embeds = training_config_ref.NUM_CLASS_EMBEDS
)

ref_model_path = "flow_dpo_training_logs/gdx9e3sy/checkpoints/FlowModel-epoch=94-FID=14.2297.ckpt"

flowmodel_ref = FlowModel.load_from_checkpoint(checkpoint_path = ref_model_path,
                                               unet = unet,
                                               model_config = model_config_ref,
                                               training_config = training_config_ref
                                               )

unet_ref = UNet2DModel(
    sample_size = training_config_ref.LATENT_SIZE,
    in_channels = model_config_ref.IN_CHANNELS,
    out_channels = model_config_ref.OUT_CHANNELS,
    down_block_types = model_config_ref.DOWN_BLOCK_TYPES,
    up_block_types = model_config_ref.UP_BLOCK_TYPES,
    block_out_channels = model_config_ref.BLOCK_OUT_CHANNELS,
    layers_per_block = model_config_ref.LAYERS_PER_BLOCK,
    num_class_embeds = training_config_ref.NUM_CLASS_EMBEDS
)

unet_ref.load_state_dict(flowmodel_ref.unet.state_dict())
unet_train = copy.deepcopy(unet_ref)

flowmodel_dpo = FlowDPOModule(unet_train,unet_ref,model_config_dpo,training_config_dpo)


if len(sys.argv) > 1:
    if sys.argv[1] == 'cont':
        trainer.fit(flowmodel_dpo,datamodule=data_module,ckpt_path="flow_dpo_training_logs/ip7m8d1g/checkpoints/FlowModel_dpo-epoch=83-train_epoch_avg_loss=0.0000.ckpt")
else:
    trainer.fit(flowmodel_dpo,datamodule=data_module)

