"""
Compares the number of pairs cached in data/hpdv3_pairs.pt (CachedDPOHPDv3Dataset)
against the number of pairs HPDv3Dataset would produce from the raw HPDv3 train split.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/

import torch

from datamodule.DPO_loader import CachedDPOHPDv3Dataset
from datamodule.HPDv3_dataloader import HPDv3Dataset

CACHE_PATH = "data/hpdv3_pairs.pt"
META_PATH = CACHE_PATH + ".meta"
BATCH_SIZE = 10  # matches training_config_3_dpo.BATCH_SIZE used during precompute


def main():
    if os.path.exists(META_PATH):
        meta = torch.load(META_PATH, map_location="cpu")
        last_batch = meta["last_batch"]
        print(f"meta: last_batch={last_batch} -> {last_batch + 1} batches saved "
              f"(<= {(last_batch + 1) * BATCH_SIZE} examples at batch_size={BATCH_SIZE})")
    else:
        print("No meta file found.")

    print(f"Loading cached dataset from {CACHE_PATH}...")
    try:
        cached = CachedDPOHPDv3Dataset(CACHE_PATH)
        cached_len = len(cached)
        print(f"Cached pairs: {cached_len}")
    except Exception as e:
        print(f"FAILED to load cache: {e}")
        cached_len = None

    print("Loading HPDv3Dataset(split='train') to get expected pair count...")
    raw = HPDv3Dataset(split="train")
    raw_len = len(raw)
    print(f"HPDv3Dataset(train) pairs: {raw_len}")

    print()
    if cached_len is None:
        print("Cannot compare: cache file failed to load (see error above).")
    elif cached_len == raw_len:
        print(f"MATCH: cache has all {raw_len} pairs.")
    else:
        diff = raw_len - cached_len
        print(f"MISMATCH: cache is missing {diff} pairs "
              f"({cached_len}/{raw_len}, {100 * cached_len / raw_len:.2f}% complete).")


if __name__ == "__main__":
    main()
