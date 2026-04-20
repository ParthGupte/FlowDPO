from models.unet_flow import FlowModel, UNet2DModel

def load_model_mnist(ckpt_path,model_config,training_config):
    unet_model = UNet2DModel(
        sample_size = training_config.LATENT_SIZE,
        in_channels = model_config.IN_CHANNELS,
        out_channels = model_config.OUT_CHANNELS,
        down_block_types = model_config.DOWN_BLOCK_TYPES,
        up_block_types = model_config.UP_BLOCK_TYPES,
        block_out_channels = model_config.BLOCK_OUT_CHANNELS,
        layers_per_block = model_config.LAYERS_PER_BLOCK,
        num_class_embeds = training_config.NUM_CLASS_EMBEDS
    )
    flow_model = FlowModel.load_from_checkpoint(checkpoint_path= ckpt_path,
                                                unet = unet_model,
                                                model_config = model_config,
                                                training_config = training_config
                                                )
    return flow_model
    