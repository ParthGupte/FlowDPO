from models.sd3_dpo_module import FlowDPOSD3, FLowDPOSD3Wallace
from diffusers import StableDiffusion3Pipeline
import copy
import torch
from PIL import Image
from torchvision import transforms

def load_image(
    path,
    image_size=1024,
    device="cuda",
    dtype=torch.float16,
):

    transform = transforms.Compose([

        transforms.Resize(
            (image_size, image_size)
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            [0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5]
        )

    ])

    img = Image.open(path).convert("RGB")

    img = transform(img)

    img = img.unsqueeze(0)

    img = img.to(
        device=device,
        dtype=dtype
    )

    return img

def load_sd3_model(pipe:StableDiffusion3Pipeline,model_config,training_config):
    ref_transformer = copy.deepcopy(pipe.transformer)
    flow_dpo_sd3 = FlowDPOSD3(pipe,ref_transformer,model_config,training_config)
    return flow_dpo_sd3.to("cuda")

def load_sd3_model_wallace(pipe:StableDiffusion3Pipeline,model_config,training_config):
    ref_transformer = copy.deepcopy(pipe.transformer)
    flow_dpo_sd3 = FLowDPOSD3Wallace(pipe,ref_transformer,model_config,training_config)
    return flow_dpo_sd3.to("cuda")

def load_lora_weights(model, ckpt_path):
    """Loads just the LoRA adapter weights from a checkpoint saved by
    FlowDPOSD3.on_save_checkpoint into an already-constructed model (i.e. one
    whose transformer already has the matching LoRA adapter injected via
    add_adapter in __init__). Used for the "cont" flow instead of Lightning's
    ckpt_path= resume, since that checkpoint has no optimizer/epoch state to
    restore anyway (save_weights_only=True) and only the base transformer's
    LoRA deltas -- not ref_transformer, fid_metric, etc. -- need loading.
    """
    if not model.use_lora:
        raise ValueError("load_lora_weights requires a model built with USE_LORA=True")

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    lora_state_dict = {
        k[len("transformer."):]: v
        for k, v in checkpoint["state_dict"].items()
        if k.startswith("transformer.") and "lora_" in k
    }
    if not lora_state_dict:
        raise ValueError(f"No LoRA weights found in checkpoint: {ckpt_path}")

    _, unexpected = model.transformer.load_state_dict(lora_state_dict, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected LoRA keys not found in model: {unexpected}")
    return model

# from models.sd3 import pipe
# from experiments.configs import training_config_1_dpo as training_config, model_config_1 as model_config
# from datamodule.HPDv3_dataloader import sample

# flow_dpo_sd3 = load_sd3_model(pipe,model_config,training_config)


# cond = flow_dpo_sd3.encode_prompt("a photo of black musclular semi naked man holding a sign that says hello world")
# # IN_CHANNELS = flow_dpo_sd3.transformer.config.in_channels
# # X_0 = flow_dpo_sd3.pipe.prepare_latents(1,IN_CHANNELS,1024,1024,cond[0].dtype,cond[0].device,None)
# # gen_image_latents = flow_dpo_sd3.generate_image(X_0,cond)
# # print(gen_image_latents.shape)
# # final_img = flow_dpo_sd3.decode_image(gen_image_latents).detach()
# # final_img_pil = flow_dpo_sd3.pipe.image_processor.postprocess(final_img)[0]
# # final_img_pil.save("model_load_test.png")
# final_img = load_image("model_load_test.png")
# encoded = flow_dpo_sd3.encode_image(final_img)
# final_img_2 = flow_dpo_sd3.decode_image(encoded)
# final_img_pil_2 = flow_dpo_sd3.pipe.image_processor.postprocess(final_img_2.detach())[0]
# final_img_pil_2.save("decoded.png")
# # out = StableDiffusion3PipelineOutput(images=final_img_pil)
# # final_img_pil = flow_dpo_sd3.minimal_generate("a photo of black musclular semi naked man holding a sign that says hello world")

# # print(final_img.shape)

# final_img = load_image("model_load_test.png")
# encoded = flow_dpo_sd3.encode_image(final_img)
# X_0 = flow_dpo_sd3.solve_noise(encoded,cond)
# img_pil = flow_dpo_sd3.minimal_generate("a photo of black musclular semi naked man holding a sign that says hello world",X_0)
# img_pil.save("recon_from_noise.png")