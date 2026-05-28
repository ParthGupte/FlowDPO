import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import lightning as L
from torch.utils.data import random_split
import math

def standard_normal_logprob(x, dim=None):
    """
    Computes log p(x) under standard normal N(0, I)

    Args:
        x: tensor of any shape
        dim: dimensions to sum over (default: all except batch)

    Returns:
        log_prob: tensor of shape (batch,) if dim is specified
    """
    if dim is None:
        dim = tuple(range(1, x.ndim))  # sum over all non-batch dims
    
    log_z = -0.5 * math.log(2 * math.pi)
    
    return (log_z - 0.5 * x**2).sum(dim=dim)



class CachedDPOMNISTDataset(Dataset):
    def __init__(self, path):
        # Filter only digit 9
        self.data = torch.load(path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row_dict = self.data[idx]
        y_neg = row_dict['x1']
        y_pos = row_dict['x1_flip']

        y0_neg = row_dict['x0']
        y0_pos = row_dict['x0_flip']

        x = row_dict["label"]
        
        return {
            "x": x,        # context (can be unused)
            "y_pos": y_pos,
            "y_neg": y_neg,
            "y0_neg": y0_neg,
            "y0_pos": y0_pos
        }
# -----------------------------
# Flow Wrapper Dataset
# -----------------------------
class FlowDPODataset(Dataset):
    def __init__(self, base_dataset):
        self.dataset = base_dataset
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        row_dict = self.dataset[idx]
        row_dict['t1'] = torch.rand(())
        row_dict['t2'] = row_dict['t1'] #torch.rand(())
        row_dict['t3'] = row_dict['t1'] #torch.rand(())
        row_dict['t4'] = row_dict['t1'] #torch.rand(())
        row_dict['t5'] = row_dict['t1'] #torch.rand(())
        return row_dict

# -----------------------------
# Lightning DataModule
# -----------------------------
class FlowDPODataModule(L.LightningDataModule):
    def __init__(self, batch_size=64, num_workers=2, data_path="data/mnist_9_flow_pairs.pt"):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_path = data_path
        

    def setup(self, stage=None):
        # Called on every GPU separately in DDP
        cached_dataset = CachedDPOMNISTDataset(self.data_path)
        train_dataset, test_dataset = random_split(cached_dataset,[0.8,0.2],torch.Generator().manual_seed(42))
        
        if stage == "fit" or stage is None:
            
            train_base = train_dataset
            val_base = test_dataset

            self.train_dataset = FlowDPODataset(train_base)
            self.val_dataset = FlowDPODataset(val_base)

        if stage == "test" or stage is None:
            test_base = test_dataset
            self.test_dataset = FlowDPODataset(test_base)

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

if __name__ == "__main__":
    dataset = CachedDPOMNISTDataset("data/mnist_9_flow_pairs.pt")
    for i in range(10):
        item = dataset[i]
        print(
            i,
            item["y_neg"].shape,
            item["y_pos"].shape,
            item["y0_neg"].shape,
            item["y0_pos"].shape,
            standard_normal_logprob(item["y0_neg"]),
            standard_normal_logprob(item["y0_pos"])
        )