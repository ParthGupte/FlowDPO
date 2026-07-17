"""
Model configuration variables for experiment 4
"""

IN_CHANNELS = 3
OUT_CHANNELS = 3
DOWN_BLOCK_TYPES = ("DownBlock2D",
                    "AttnDownBlock2D",
                    "AttnDownBlock2D",
                    "DownBlock2D")

UP_BLOCK_TYPES = ("UpBlock2D",
                  "AttnUpBlock2D",
                  "AttnUpBlock2D",
                  "UpBlock2D")

BLOCK_OUT_CHANNELS = (64,96,128,256)


LAYERS_PER_BLOCK = 2

ATTENTION_HEAD_DIM = 8 