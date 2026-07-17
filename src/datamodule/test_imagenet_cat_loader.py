"""
Scans the streaming ImageNet-1k cat dataset (see ImageNet_cat_loader.py),
counts how many images fall into each of the 5 cat classes, and saves one
example image per class into a single labeled plot.

Since imagenet-1k is streamed, this walks the whole split to get exact counts,
which can take a long time. Pass --limit to cap how many cat images are
scanned (useful for a quick smoke test).
"""
import argparse
import sys
import os
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/

import matplotlib.pyplot as plt

from datamodule.ImageNet_cat_loader import get_imagenet_cat_dataset

OUT_PATH = "imagenet_cat_samples.png"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="train")
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--limit", type=int, default=None,
                         help="stop after this many cat images are found (default: scan whole split)")
    parser.add_argument("--progress-every", type=int, default=500)
    args = parser.parse_args()

    dataset = get_imagenet_cat_dataset(split=args.split, image_size=args.image_size)

    counts = Counter()
    example_image = {}

    scanned = 0
    for ex in dataset:
        scanned += 1
        label = ex["label"]

        counts[label] += 1
        if label not in example_image:
            img = (ex["image"] * 0.5 + 0.5).clamp(0, 1)  # undo the (0.5, 0.5, 0.5) normalize
            example_image[label] = img.permute(1, 2, 0).numpy()

        if scanned % args.progress_every == 0:
            print(f"found {scanned} cat images so far...")

        if args.limit is not None and scanned >= args.limit:
            break

    print(f"\nScanned {scanned} cat images from split='{args.split}'")
    print("Cat image counts per class:")
    for name in sorted(counts):
        print(f"  {name}: {counts[name]}")
    print(f"  TOTAL: {sum(counts.values())}")

    names = sorted(example_image)
    if not names:
        print("No cat images found, skipping plot.")
        return

    fig, axes = plt.subplots(1, len(names), figsize=(4 * len(names), 4))
    if len(names) == 1:
        axes = [axes]

    for ax, name in zip(axes, names):
        ax.imshow(example_image[name])
        ax.set_title(f"{name}\n(n={counts[name]})", fontsize=10)
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(OUT_PATH)
    print(f"\nSaved sample plot to {OUT_PATH}")


if __name__ == "__main__":
    main()
