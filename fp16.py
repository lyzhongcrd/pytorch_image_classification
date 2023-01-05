import torch
from torch import nn
import time
import pandas as pd

n_trials = 100
warmup_iters = 100
torch.backends.cudnn.benchmark = False

convs = [
    # ===      Input dimensions      ===   === Stride === ===           Kernel          ===
    {'n': 8, 'C': 64,  'H': 64,  'W': 64,  'u': 4, 'v': 4, 'K': 64,  'R': 4, 'S': 4, 'G': 1, 'pad': 1, 'bias': False,},
]

for conv in convs:
    model = nn.Conv2d(
        in_channels=conv['C'],
        out_channels=conv['K'],
        kernel_size=(conv['R'], conv['S']),
        stride=(conv['u'], conv['v']),
        padding=conv['pad'],
        groups=conv['G'],
        bias=conv['bias'],
    ).cuda().half()
    x = torch.randn((conv['n'], conv['C'], conv['H'], conv['W']),
                    device='cuda', dtype=torch.float16, requires_grad=True)
    # Select kernels, get y, dy
    for _ in range(warmup_iters):
        y = model.forward(x)
        dy = torch.randn_like(y)
        y.backward(dy)
    # Time forward pass
    torch.cuda.synchronize()
    t_start = time.perf_counter()
    for _ in range(n_trials):
        y = model.forward(x)
    torch.cuda.synchronize()
    t_end = time.perf_counter()
    dt_fwd = (t_end - t_start) / n_trials
    # Time backward pass
    torch.cuda.synchronize()
    dy = torch.randn_like(y)
    torch.cuda.synchronize()
    t_start = time.perf_counter()
    for _ in range(n_trials):
        y.backward(dy, retain_graph=True)
    torch.cuda.synchronize()
    t_end = time.perf_counter()
    dt_bwd = (t_end - t_start) / n_trials
    conv[f"fwd_{'fp16'}"] = int(dt_fwd*1e6)
    conv[f"bwd_{'fp16'}"] = int(dt_bwd*1e6)
df = pd.DataFrame(convs)
print(repr(df))