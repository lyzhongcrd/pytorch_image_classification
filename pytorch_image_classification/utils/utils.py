import pathlib
import random

import numpy as np
import torch
import yacs.config
# from contextlib import contextmanager


def set_seed(config: yacs.config.CfgNode) -> None:
    seed = config.train.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def setup_cudnn(config: yacs.config.CfgNode) -> None:
    torch.backends.cudnn.benchmark = config.cudnn.benchmark
    torch.backends.cudnn.deterministic = config.cudnn.deterministic


def save_config(config: yacs.config.CfgNode,
                output_path: pathlib.Path) -> None:
    with open(output_path, 'w') as f:
        f.write(str(config))

# @contextmanager
# def autocast_context_warp(enabled):
#     if enabled:
#         with torch.autocast(device_type='cuda', dtype=torch.float16):
#             yield
#     else:
#         yield

