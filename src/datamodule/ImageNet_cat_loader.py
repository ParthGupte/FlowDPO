import torch
from torch.utils.data import IterableDataset, DataLoader
from torchvision import transforms
from datasets import load_dataset
import lightning as L

# Standard ILSVRC2012 label indices (0-999) for cat classes
CAT_LABELS = {281, 282, 283, 284, 285}  # tabby, tiger cat, Persian, Siamese, Egyptian cat

# -----------------------------
# ImageNet-1k Cat Base Dataset (streaming)
# -----------------------------
def get_imagenet_cat_dataset(split="train", image_size=64):
    dataset = load_dataset(
        "imagenet-1k",
        split=split
    )

    label_names = dataset.features["label"].names

    dataset = dataset.filter(lambda ex: ex["label"] in CAT_LABELS)

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),                    # [0,1]
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # -> [-1,1]
    ])

    def apply_transform(ex):
        ex["image"] = transform(ex["image"].convert("RGB"))
        ex["label"] = label_names[ex["label"]]
        return ex

    dataset = dataset.map(apply_transform)

    return dataset


# -----------------------------
# Flow Wrapper Dataset
# -----------------------------
class FlowImageNetCatDataset(IterableDataset):
    def __init__(self, base_dataset):
        self.dataset = base_dataset

    def __iter__(self):
        for ex in self.dataset:
            x1 = ex["image"]
            label = ex["label"]

            x0 = torch.randn_like(x1)      # Gaussian noise
            t = torch.rand(())              # scalar time

            yield x1, label, x0, t


# -----------------------------
# Lightning DataModule
# -----------------------------
class FlowImageNetCatDataModule(L.LightningDataModule):
    def __init__(self, batch_size=64, num_workers=2, image_size=64):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            train_base = get_imagenet_cat_dataset(split="train", image_size=self.image_size)
            val_base = get_imagenet_cat_dataset(split="validation", image_size=self.image_size)

            self.train_dataset = FlowImageNetCatDataset(train_base)
            self.val_dataset = FlowImageNetCatDataset(val_base)

        if stage == "test" or stage is None:
            test_base = get_imagenet_cat_dataset(split="test", image_size=self.image_size)
            self.test_dataset = FlowImageNetCatDataset(test_base)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True
        )


# -----------------------------
# Example Usage
# -----------------------------
if __name__ == "__main__":
    dm = FlowImageNetCatDataModule(batch_size=32)
    dm.setup()

    batch = next(iter(dm.train_dataloader()))
    x1, label, x0, t = batch

    print("x1 shape:", x1.shape)
    print("label:", label)
    print("x0 shape:", x0.shape)
    print("t shape:", t.shape)
