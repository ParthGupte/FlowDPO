import os

import torch
from HPDv3_dataloader import HPDv3Dataset


def backfill_prompts(cache_path="data/hpdv3_pairs.pt"):
    results = torch.load(cache_path, map_location="cpu")
    # precompute_noise_hpdv3.py iterates this dataset with shuffle=False, so
    # results[i] always corresponds to dataset_train.ds[i] by index -- even if
    # the cache is only a partial run, the prefix that was written still lines
    # up with the dataset's prefix.
    dataset_train = HPDv3Dataset(split="train")

    if len(results) > len(dataset_train):
        raise ValueError(
            f"Cache has {len(results)} entries but HPDv3Dataset(train) only has "
            f"{len(dataset_train)} -- dataset filtering must have changed since "
            "precompute ran, so index-based backfill is not valid."
        )

    for i, row in enumerate(results):
        row["prompt"] = dataset_train.ds[i]["prompt"]

    tmp_path = cache_path + ".tmp"
    torch.save(results, tmp_path)
    os.replace(tmp_path, cache_path)
    print(f"Backfilled prompt field on {len(results)} entries")


if __name__ == "__main__":
    backfill_prompts()
