import torch
from torch import distributed as dist

import functools
import os
from collections import OrderedDict
from math import cos, pi

def load_checkpoint(checkpoint, logger, model, strict=False):
    state_dict = torch.load(checkpoint, map_location='cpu')
    src_state_dict = state_dict["net"]
    target_state_dict = model.state_dict()
    skip_keys = []
    # skip mismatch size tensors in case of pretraining
    for k in src_state_dict.keys():
        if k not in target_state_dict:
            continue
        if src_state_dict[k].size() != target_state_dict[k].size():
            skip_keys.append(k)
    for k in skip_keys:
        del src_state_dict[k]
    missing_keys, unexpected_keys = model.load_state_dict(src_state_dict, strict=strict)
    if skip_keys:
        logger.info(f'removed keys in source state_dict due to size mismatch: {", ".join(skip_keys)}')
    if missing_keys:
        logger.info(f'missing keys in source state_dict: {", ".join(missing_keys)}')
    if unexpected_keys:
        logger.info(f'unexpected key in source state_dict: {", ".join(unexpected_keys)}')