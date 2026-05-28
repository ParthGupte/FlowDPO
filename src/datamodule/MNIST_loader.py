import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import lightning as L

flip = transforms.RandomHorizontalFlip(p=0.5)
# -----------------------------
# MNIST Base Dataset
# -----------------------------
def get_mnist_dataset(train=True):
    transform = transforms.Compose([
        transforms.ToTensor(),                    # [0,1]
        transforms.Normalize((0.5,), (0.5,))      # -> [-1,1]
    ])
    
    dataset = datasets.MNIST(
        root="./data",
        train=train,
        download=True,
        transform=transform
    )
    
    return dataset


# -----------------------------
# Flow Wrapper Dataset
# -----------------------------
class FlowMNISTDataset(Dataset):
    def __init__(self, base_dataset, flipping = False):
        self.dataset = base_dataset
        self.flipping = flipping

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x1, label = self.dataset[idx]
        if self.flipping:
            x1 = flip(x1)


        x0 = torch.randn_like(x1)      # Gaussian noise
        t = torch.rand(())              # scalar time

        return x1, label, x0, t


# -----------------------------
# Lightning DataModule
# -----------------------------
class FlowMNISTDataModule(L.LightningDataModule):
    def __init__(self, batch_size=64, num_workers=2, data_dir="./data",flipping = False):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_dir = data_dir
        self.flipping = flipping

    def setup(self, stage=None):
        # Called on every GPU separately in DDP
        if stage == "fit" or stage is None:
            train_base = get_mnist_dataset(train=True)
            val_base = get_mnist_dataset(train=False)

            self.train_dataset = FlowMNISTDataset(train_base,flipping=self.flipping)
            self.val_dataset = FlowMNISTDataset(val_base,flipping=self.flipping)

        if stage == "test" or stage is None:
            test_base = get_mnist_dataset(train=False)
            self.test_dataset = FlowMNISTDataset(test_base,flipping=self.flipping)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )


# -----------------------------
# Example Usage
# -----------------------------
if __name__ == "__main__":
    dm = FlowMNISTDataModule(batch_size=32)
    dm.setup()

    batch = next(iter(dm.train_dataloader()))

    print("x0 shape:", batch["x0"].shape)
    print("x1 shape:", batch["x1"].shape)
    print("t shape:", batch["t"].shape)