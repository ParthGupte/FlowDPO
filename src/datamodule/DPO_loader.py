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
    
class CachedDPOHPDv3Dataset(Dataset):
    def __init__(self, path):
        # data/hpdv3_pairs.pt is ~500GB; mmap=True keeps tensor storages on
        # disk and pages them in per-access instead of loading everything
        # into RAM upfront, which otherwise OOM-kills the process.
        self.data = torch.load(path, mmap=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row_dict = self.data[idx]
        y_l_enc = row_dict['y_l_enc']
        y_w_enc = row_dict['y_w_enc']

        x0_l = row_dict['x0_l']
        x0_w = row_dict['x0_w']

        cond = row_dict["prompt_embeds"], row_dict["pooled_prompt_embeds"]

        return {
            "cond": cond,        # context (can be unused)
            "y_pos": y_w_enc,
            "y_neg": y_l_enc,
            "y0_neg": x0_l,
            "y0_pos": x0_w,
            "prompt": row_dict["prompt"]
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
        # match y_pos's dtype so y_t_til's interpolation doesn't upcast fp16
        # latents to float32 (which then mismatches the model's fp16 weights)
        dtype = row_dict['y_pos'].dtype
        row_dict['t1'] = torch.rand((), dtype=dtype)
        row_dict['t2'] = torch.rand((), dtype=dtype)
        row_dict['t3'] = torch.rand((), dtype=dtype)
        row_dict['t4'] = torch.rand((), dtype=dtype)
        row_dict['t5'] = torch.rand((), dtype=dtype)
        return row_dict

# -----------------------------
# Lightning DataModule
# -----------------------------
class FlowDPODataModule(L.LightningDataModule):
    def __init__(self, batch_size=64, num_workers=2, data_path="data/mnist_9_flow_pairs.pt",dataset_class = CachedDPOMNISTDataset):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_path = data_path
        self.dataset_class = dataset_class
        

    def setup(self, stage=None):
        # Called on every GPU separately in DDP
        cached_dataset = self.dataset_class(self.data_path)
        train_dataset, test_dataset = random_split(cached_dataset,[0.8,0.2],torch.Generator().manual_seed(42))
        
        if stage == "fit" or stage == "validate" or stage is None:
            
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

class FlowDPOSD3DataModule(L.LightningDataModule):
    def __init__(self, batch_size=64, num_workers=2, data_path="data/hpdv3_pairs.pt",dataset_class = CachedDPOHPDv3Dataset,
                 val_batch_size=None, test_batch_size=None):
        super().__init__()
        self.batch_size = batch_size
        # validation runs a full SD3 generation + HPSv3 reward pass per image,
        # so it needs a much smaller batch size than training to fit in GPU
        # memory; default to batch_size so callers that don't care can ignore this.
        self.val_batch_size = val_batch_size if val_batch_size is not None else batch_size
        self.test_batch_size = test_batch_size if test_batch_size is not None else batch_size
        self.num_workers = num_workers
        self.data_path = data_path
        self.dataset_class = dataset_class


    def setup(self, stage=None):
        # Called on every GPU separately in DDP
        cached_dataset = self.dataset_class(self.data_path)
        train_dataset, test_dataset = random_split(cached_dataset,[0.99,0.01],torch.Generator().manual_seed(42))
        
        if stage == "fit" or stage == "validate" or stage is None:
            
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
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

if __name__ == "__main__":
    ds = CachedDPOHPDv3Dataset("data/hpdv3_pairs.pt")
    # shapes = {tuple(ds[i]["cond"][0].shape) for i in range(len(ds))}
    # print(shapes)
    # ds = CachedDPOMNISTDataset("data/mnist_9_flow_pairs.pt")
    for i in range(10):
        item = ds[i]
        print(
            i,
            item["y_neg"].shape,
            item["y_pos"].shape,
            item["y0_neg"].shape,
            item["y0_pos"].shape,
            standard_normal_logprob(item["y0_neg"]),
            standard_normal_logprob(item["y0_pos"])
        )