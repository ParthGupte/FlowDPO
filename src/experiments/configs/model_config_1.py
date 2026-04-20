"""
Model configuration variables for experiment 1
"""

IN_CHANNELS = 1
OUT_CHANNELS = 1
DOWN_BLOCK_TYPES = ("DownBlock2D",
                    "AttnDownBlock2D",
                    "DownBlock2D")

UP_BLOCK_TYPES = ("UpBlock2D",
                  "AttnUpBlock2D",
                  "UpBlock2D")

BLOCK_OUT_CHANNELS = (32,64,128)


LAYERS_PER_BLOCK = 2

ATTENTION_HEAD_DIM = 8 