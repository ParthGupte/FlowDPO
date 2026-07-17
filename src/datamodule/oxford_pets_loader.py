import kagglehub
from pathlib import Path
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import torch
import lightning as L
from torch.utils.data import random_split

generator = torch.Generator().manual_seed(42)

# Download dataset
root = Path(kagglehub.dataset_download("tanlikesmath/the-oxfordiiit-pet-dataset"))
print(f"Dataset path: {root}")

CAT_BREEDS = ["Abyssinian","Bengal","Birman","Bombay","British Shorthair","Egyptian Mau","Maine Coon","Persian","Ragdoll","Russian Blue","Siamese","Sphynx"]

class OxfordPetCats(Dataset):
    def __init__(self, root=root, image_size =256):
        self.root = Path(root)
        self.transform = transforms.Compose([transforms.Resize((image_size, image_size)),transforms.ToTensor(),])

        # Find image directory
        image_dir = self.root / "images"
        if not image_dir.exists():
            image_dir = self.root

        # All jpg images
        image_paths = sorted(image_dir.glob("*.jpg"))

        self.samples = []

        for img_path in image_paths:
            stem = img_path.stem  # e.g. "British_Shorthair_123"

            # Remove trailing image number
            breed = "_".join(stem.split("_")[:-1]).replace("_", " ")
            if breed not in CAT_BREEDS:
                continue

            self.samples.append(
                {
                    "image_path": img_path,
                    "breed": breed,
                }
            )

        # Create contiguous breed IDs
        breeds = sorted({sample["breed"] for sample in self.samples})
        self.breed_to_id = {b: i for i, b in enumerate(breeds)}
        self.id_to_breed = {i: b for b, i in self.breed_to_id.items()}

        for sample in self.samples:
            sample["breed_id"] = self.breed_to_id[sample["breed"]]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        image = Image.open(sample["image_path"]).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return {
            "image": image,
            "breed": sample["breed"],
            "breed_id": sample["breed_id"],
        }




# -----------------------------
# Flow Wrapper Dataset
# -----------------------------
class FlowOxfordPetCatsDataset(Dataset):
    def __init__(self, base_dataset):
        self.dataset = base_dataset
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self,idx):
        ex = self.dataset[idx]
        x1 = ex["image"]
        label = ex["breed"]
        x0 = torch.randn_like(x1)      # Gaussian noise
        t = torch.rand(())              # scalar time
        return x1, label, x0, t

# -----------------------------
# Lightning DataModule
# -----------------------------
class FlowOxfordPetCatsDataModule(L.LightningDataModule):
    def __init__(self, batch_size=64, num_workers=2, image_size=64):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size

    def setup(self, stage=None):
        dataset = OxfordPetCats(image_size=self.image_size)
        if stage == "fit" or stage is None:
            train_base, val_base = random_split(dataset,[0.9,0.1], generator=generator) 
            self.train_dataset = FlowOxfordPetCatsDataset(train_base)
            self.val_dataset = FlowOxfordPetCatsDataset(val_base)

        if stage == "test" or stage is None:
            train_base, test_base = random_split(dataset, [0.9,0.1], generator=generator) 
            self.test_dataset = FlowOxfordPetCatsDataset(test_base)

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

# dataset = OxfordPetCats(root,transform=transforms.ToTensor())

# print(f"Number of images: {len(dataset)}")
# print(f"Number of breeds: {len(dataset.breed_to_id)}")
# shapes = []
# for i in range(len(dataset)):
#     sample = dataset[i]
#     if sample["image"].shape not in shapes:
#         shapes.append(sample["image"].shape)
#         print(sample["image"].shape)
#         print(sample["breed"])
#         print(sample["breed_id"])

# print("\nBreed counts:")
# counts = Counter(s["breed"] for s in dataset.samples)
# for breed, count in sorted(counts.items()):
#     print(f"{breed:20s} {count}")