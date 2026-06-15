from models.unet_flow import FlowModel, UNet2DModel
from models.flow_dpo_module import FlowDPOModule


def load_dpo_module(ckpt_path, model_config, training_config):
    """Load a FlowDPOModule from checkpoint (returns the DPO module)."""
    # create unet placeholders for both train and reference
    unet_train = UNet2DModel(
        sample_size = training_config.LATENT_SIZE,
        in_channels = model_config.IN_CHANNELS,
        out_channels = model_config.OUT_CHANNELS,
        down_block_types = model_config.DOWN_BLOCK_TYPES,
        up_block_types = model_config.UP_BLOCK_TYPES,
        block_out_channels = model_config.BLOCK_OUT_CHANNELS,
        layers_per_block = model_config.LAYERS_PER_BLOCK,
        num_class_embeds = training_config.NUM_CLASS_EMBEDS
    )

    unet_ref = UNet2DModel(
        sample_size = training_config.LATENT_SIZE,
        in_channels = model_config.IN_CHANNELS,
        out_channels = model_config.OUT_CHANNELS,
        down_block_types = model_config.DOWN_BLOCK_TYPES,
        up_block_types = model_config.UP_BLOCK_TYPES,
        block_out_channels = model_config.BLOCK_OUT_CHANNELS,
        layers_per_block = model_config.LAYERS_PER_BLOCK,
        num_class_embeds = training_config.NUM_CLASS_EMBEDS
    )

    dpo = FlowDPOModule.load_from_checkpoint(
        checkpoint_path=ckpt_path,
        unet=unet_train,
        unet_ref=unet_ref,
        model_config=model_config,
        training_config=training_config,
    )

    return dpo


def load_flowmodel_from_dpo_ckpt(ckpt_path, model_config, training_config):
    """Load a FlowDPOModule checkpoint and return an equivalent FlowModel using the trained `unet`.

    This is useful when downstream code expects a plain `FlowModel` but the checkpoint was saved from a DPO module.
    """
    dpo = load_dpo_module(ckpt_path, model_config, training_config)

    # extract trained unet from the DPO module and create a FlowModel wrapper
    trained_unet = dpo.unet
    flow_model = FlowModel(trained_unet, model_config, training_config)
    return flow_model

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
    