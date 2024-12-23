# clear_cache_callback.py

from transformers import TrainerCallback
import torch

class ClearCacheCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        torch.cuda.empty_cache()
