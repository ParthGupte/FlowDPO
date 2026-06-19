from __future__ import annotations

import argparse
import inspect
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from datasets import load_dataset
from PIL import Image
from torchvision import transforms


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
MODELS_ROOT = SRC_ROOT / "models"

for path in (str(SRC_ROOT), str(MODELS_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from diffusers import StableDiffusion3Pipeline  # noqa: E402
from experiments.configs import model_config_1, training_config_1_dpo  # noqa: E402
from models.sd3_dpo_module import FlowDPOSD3  # noqa: E402


DEFAULT_HPDV3_ROOT = Path(
    "/home/chaksuai/.cache/huggingface/hub/"
    "datasets--MizzenAI--HPDv3/snapshots/5ac4b5b5f2ef42c4613dec612473170b131552db"
)
DEFAULT_MODEL_ID = "stabilityai/stable-diffusion-3-medium-diffusers"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Round-trip test for FlowDPOSD3 using HPDv3 sample 0."
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="float16", choices=["float16", "float32", "bfloat16"])
    parser.add_argument("--dataset-root", default=str(DEFAULT_HPDV3_ROOT))
    parser.add_argument("--split", default="train")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--image-side", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--output-dir", default="tmp/sd3_inverse_test")
    return parser.parse_args()


def get_dtype(name: str) -> torch.dtype:
    return getattr(torch, name)


def load_sample(dataset_root: Path, split: str, index: int, image_side: int) -> dict[str, Any]:
    ds = load_dataset("MizzenAI/HPDv3", split=split)
    sample = ds[index]

    path1 = dataset_root / sample["path1"]
    path2 = dataset_root / sample["path2"]

    image_transform = transforms.Compose(
        [
            transforms.Resize((image_side, image_side)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    image1 = image_transform(Image.open(path1).convert("RGB"))
    image2 = image_transform(Image.open(path2).convert("RGB"))

    choice_dist = sample["choice_dist"]
    if choice_dist is None:
        raise ValueError(f"sample {index} has no choice_dist")
    if choice_dist[0] == choice_dist[1]:
        raise ValueError(f"sample {index} is a tie: {choice_dist}")

    if choice_dist[0] > choice_dist[1]:
        winner, loser = image1, image2
        winner_path, loser_path = path1, path2
    else:
        winner, loser = image2, image1
        winner_path, loser_path = path2, path1

    return {
        "prompt": sample["prompt"],
        "choice_dist": choice_dist,
        "winner": winner,
        "loser": loser,
        "winner_path": winner_path,
        "loser_path": loser_path,
    }


def build_module(model_id: str, device: str, dtype: torch.dtype) -> tuple[FlowDPOSD3, StableDiffusion3Pipeline]:
    pipe = StableDiffusion3Pipeline.from_pretrained(model_id, torch_dtype=dtype)
    pipe.to(device)

    ref_transformer = pipe.transformer
    module = FlowDPOSD3(pipe, ref_transformer, model_config_1, training_config_1_dpo)
    module.to(device)
    module.eval()
    return module, pipe


def encode_prompt(pipe: StableDiffusion3Pipeline, prompt: str, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    prompt_embeds, _, pooled_prompt_embeds, _ = pipe.encode_prompt(
        prompt=prompt,
        prompt_2=prompt,
        prompt_3=prompt,
        device=torch.device(device),
        num_images_per_prompt=1,
        do_classifier_free_guidance=False,
    )
    return prompt_embeds, pooled_prompt_embeds


def call_solve_noise(
    module: FlowDPOSD3,
    latent: torch.Tensor,
    cond: tuple[torch.Tensor, torch.Tensor],
    steps: int,
) -> torch.Tensor:
    signature = inspect.signature(module.solve_noise)
    if "num_inference_steps" in signature.parameters:
        return module.solve_noise(latent, cond, num_inference_steps=steps)
    return module.solve_noise(latent, cond, steps)


def call_generate_image(
    module: FlowDPOSD3,
    solved_noise: torch.Tensor,
    cond: tuple[torch.Tensor, torch.Tensor],
    steps: int,
):
    signature = inspect.signature(module.generate_image)
    params = [p for p in signature.parameters.values() if p.name != "self"]

    if len(params) == 3:
        return module.generate_image(solved_noise, cond, steps)
    if len(params) == 2:
        return module.generate_image(solved_noise, cond)
    if len(params) == 1:
        return module.generate_image(cond)

    raise TypeError(f"Unsupported generate_image signature: {signature}")


def save_tensor_image(tensor: torch.Tensor, path: Path) -> None:
    from torchvision.utils import save_image

    path.parent.mkdir(parents=True, exist_ok=True)
    save_image(((tensor.detach().cpu() + 1) / 2).clamp(0, 1), str(path))


def main() -> None:
    args = parse_args()
    dtype = get_dtype(args.dtype)
    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sample = load_sample(dataset_root, args.split, args.index, args.image_side)
    module, pipe = build_module(args.model_id, args.device, dtype)

    winner = sample["winner"].unsqueeze(0).to(args.device, dtype=torch.float16)
    prompt_embeds, pooled_prompt_embeds = encode_prompt(pipe, sample["prompt"], args.device)
    cond = (prompt_embeds, pooled_prompt_embeds)

    with torch.no_grad():
        target_latent = module.encode_image(winner)
        solved_noise = call_solve_noise(module, target_latent, cond, args.steps)

    reconstructed = call_generate_image(module, solved_noise, cond, args.steps)

    print(f"prompt: {sample['prompt'][:120]}...")
    print(f"choice_dist: {sample['choice_dist']}")
    print(f"winner_path: {sample['winner_path']}")
    print(f"loser_path: {sample['loser_path']}")
    print(f"target_latent shape: {tuple(target_latent.shape)}")
    print(f"solved_noise shape: {tuple(solved_noise.shape)}")

    if reconstructed is None:
        raise RuntimeError(
            "generate_image returned None. Update FlowDPOSD3.generate_image to return the generated "
            "image or latent tensor so the inverse test can complete."
        )

    if isinstance(reconstructed, tuple):
        raise RuntimeError(
            f"generate_image returned a tuple instead of an image tensor: {type(reconstructed)}"
        )

    recon = reconstructed.detach().to(torch.float16)
    if recon.ndim == 3:
        recon = recon.unsqueeze(0)

    if recon.shape == target_latent.shape:
        recon_latent = recon
        recon_image = module.decode_image(recon_latent).detach().to(torch.float16)
    elif recon.shape == winner.shape:
        recon_image = recon
        recon_latent = module.encode_image(recon_image).detach().to(torch.float16)
    else:
        raise RuntimeError(
            "generate_image returned an unexpected shape. "
            f"winner={tuple(winner.shape)} latent={tuple(target_latent.shape)} output={tuple(recon.shape)}"
        )

    target_latent_f16 = target_latent.to(torch.float16)
    latent_mae = F.l1_loss(recon_latent, target_latent_f16).item()
    latent_mse = F.mse_loss(recon_latent, target_latent_f16).item()
    image_mae = F.l1_loss(recon_image, winner).item()
    image_mse = F.mse_loss(recon_image, winner).item()

    print(f"reconstructed shape: {tuple(recon.shape)}")
    print(f"latent_reconstruction_mae: {latent_mae:.6f}")
    print(f"latent_reconstruction_mse: {latent_mse:.6f}")
    print(f"image_reconstruction_mae: {image_mae:.6f}")
    print(f"image_reconstruction_mse: {image_mse:.6f}")

    save_tensor_image(winner[0], output_dir / "winner.png")
    save_tensor_image(recon_image[0], output_dir / "reconstruction.png")
    save_tensor_image((winner[0] - recon_image[0]).abs(), output_dir / "abs_diff.png")
    print(f"saved outputs to {output_dir}")


if __name__ == "__main__":
    main()
