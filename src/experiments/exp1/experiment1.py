from datamodule.MNIST_loader import FlowMNISTDataModule
import lightning as L
import torch
from diffusers import UNet2DModel
from experiments.configs import training_config_1 as training_config
from lightning.pytorch.loggers import WandbLogger
from models.unet_flow import FlowModel
from lightning.pytorch.callbacks import ModelCheckpoint
import sys
from seed_all import set_seed
from experiments.configs import model_config_1 as model_config

seed_value = 42
L.seed_everything(seed_value)
set_seed(seed_value)

torch.set_float32_matmul_precision('high')

# if sys.argv[2] == "cont":
#     run_id = "o2aw64cq"
#     wandb_logger = WandbLogger(project="fundus_seg_training_logs",resume="allow",id=run_id)
# else:
wandb_logger = WandbLogger(project="flow_dpo_training_logs")

data_module = FlowMNISTDataModule(
    data_dir = training_config.DATA_PATH,
    batch_size = training_config.BATCH_SIZE,
    num_workers = training_config.NUM_WORKERS
)

checkpoint_callback_avg_loss = ModelCheckpoint(
    monitor="train_epoch_avg_loss",  # The metric to monitor (you log this yourself)
    mode="min",  # Lower loss is better
    save_top_k=1,  # Keep top 3 checkpoints
    filename="FlowModel-{epoch:02d}-{train_epoch_avg_loss:.4f}",
    save_weights_only=False,
    every_n_epochs=1
)

checkpoint_callback_FID = ModelCheckpoint(
    monitor="FID",  # The metric to monitor (you log this yourself)
    mode="min",  # Lower loss is better
    save_top_k=3,  # Keep top 3 checkpoints
    filename="FlowModel-{epoch:02d}-{FID:.4f}",
    save_weights_only=False,
    every_n_epochs=training_config.CHECK_VAL_EVERY_N_EPOCHS
)

trainer = L.Trainer(
    devices=training_config.DEVICES,
    max_epochs=training_config.MAX_EPOCHS,
    accumulate_grad_batches=training_config.ACCUMULATE_GRAD_BATCHES,
    logger=wandb_logger,
    check_val_every_n_epoch=training_config.CHECK_VAL_EVERY_N_EPOCHS,
    callbacks=[checkpoint_callback_avg_loss,checkpoint_callback_FID],
    fast_dev_run=False
)


unet = UNet2DModel(
    sample_size = training_config.LATENT_SIZE,
    in_channels = model_config.IN_CHANNELS,
    out_channels = model_config.OUT_CHANNELS,
    down_block_types = model_config.DOWN_BLOCK_TYPES,
    up_block_types = model_config.UP_BLOCK_TYPES,
    block_out_channels = model_config.BLOCK_OUT_CHANNELS,
    layers_per_block = model_config.LAYERS_PER_BLOCK,
    num_class_embeds = training_config.NUM_CLASS_EMBEDS
)

flowmodel = FlowModel(unet,model_config,training_config)
trainer.fit(flowmodel,datamodule=data_module)
