import os
import tempfile

import torch
from torchvision.utils import save_image
from hpsv3 import HPSv3RewardInferencer


class HPSv3Metric:
    """Scores generated images against their text prompts with HPSv3.

    HPSv3RewardInferencer.reward() only accepts image file paths (no PIL/tensor
    input), so images are written to a temp dir per call and cleaned up right
    after scoring.
    """

    def __init__(self, device="cuda"):
        self.inferencer = HPSv3RewardInferencer(device=device)

    def to(self, device):
        # HPSv3RewardInferencer._prepare_input moves batch tensors using its
        # own stored .device string rather than the model's actual device, so
        # both must be updated together or reward() breaks with a device mismatch.
        self.inferencer.model.to(device)
        self.inferencer.device = device
        return self

    @torch.no_grad()
    def score(self, prompts, images):
        """prompts: list[str]; images: float tensor [B,C,H,W] in [0,1]. Returns tensor [B]."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            image_paths = []
            for i, image in enumerate(images):
                path = os.path.join(tmp_dir, f"{i}.png")
                save_image(image, path)
                image_paths.append(path)

            rewards = self.inferencer.reward(image_paths, list(prompts))

        return torch.tensor([reward[0].item() for reward in rewards])
