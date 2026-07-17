import os

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from datasets import load_dataset
from PIL import Image

ALLOWED_MODELS = {"flux","kolors","sd3","hunyuan","real_images"}

ROOT = "/home/chaksuai/.cache/huggingface/hub/datasets--MizzenAI--HPDv3/snapshots/5ac4b5b5f2ef42c4613dec612473170b131552db"


class HPDv3Dataset(Dataset):
    def __init__(self,root=ROOT,split="train",image_size=1024):
        """
        Args:
            root: path to downloaded HPDv3 snapshot
            split: "train" or "test"
            image_size: resize images to (image_size, image_size)
        """
        assert split in ["train", "test"]
        self.root = root
        self.ds = load_dataset(
            "MizzenAI/HPDv3"
        )[split]
        self.ds = self.ds.filter(lambda ex:ex["model1"] in ALLOWED_MODELS and ex["model2"] in ALLOWED_MODELS)
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.5, 0.5, 0.5],
                [0.5, 0.5, 0.5]
            ),
            transforms.ConvertImageDtype(torch.float16),
        ])

    def __len__(self):
        return len(self.ds)

    def _load_image(self, rel_path):
        img = Image.open(
            os.path.join(
                self.root,
                rel_path
            )
        ).convert("RGB")
        return self.transform(img)

    def __getitem__(self, idx):
        ex = self.ds[idx]
        path1 = ex["path1"]
        path2 = ex["path2"]

        votes1 = ex["choice_dist"][0]
        votes2 = ex["choice_dist"][1]


        # Determine winner/loser
        if votes1 >= votes2:

            winner_path = path1
            loser_path = path2

            winner_votes = votes1
            loser_votes = votes2

            winner_model = ex["model1"]
            loser_model = ex["model2"]

        else:

            winner_path = path2
            loser_path = path1

            winner_votes = votes2
            loser_votes = votes1

            winner_model = ex["model2"]
            loser_model = ex["model1"]


        winner = self._load_image(
            winner_path
        )

        loser = self._load_image(
            loser_path
        )


        return {

            "winner": winner,

            "loser": loser,

            "prompt": ex["prompt"],

            "winner_votes": torch.tensor(
                winner_votes,
                dtype=torch.long
            ),

            "loser_votes": torch.tensor(
                loser_votes,
                dtype=torch.long
            ),

            "confidence": torch.tensor(
                ex["confidence"],
                dtype=torch.float32
            ),

            "winner_model": winner_model,

            "loser_model": loser_model,
        }


# if __name__ == "__main__":

    
#     dataset = HPDv3Dataset(
#         ROOT,
#         split="train",
#         image_size=1024,
#     )
#     print("Total comparisons:",len(dataset))

#     hpdv3_loader = DataLoader(
#         dataset,
#         batch_size=4,
#         shuffle=True,
#         num_workers=4,
#         pin_memory=True,
#     )

#     batch = next(iter(hpdv3_loader))

#     print(batch["winner"].shape)
#     print(batch["loser"].shape)

#     print(batch["prompt"][0])

#     print(batch["winner_votes"])
#     print(batch["loser_votes"])

#     print(batch["winner_model"])
#     print(batch["loser_model"])