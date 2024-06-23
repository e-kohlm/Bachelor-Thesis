import gc
import torch

model = None
gc.collect()
torch.cuda.empty_cache()

with torch.no_grad():
    torch.cuda.empty_cache()
