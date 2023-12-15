import pdb

import numpy as np
import torch

DTYPE = torch.float
DEVICE = "cuda"

# -----------------------------------------------------------------------------#
# ------------------------------ numpy <--> torch -----------------------------#
# -----------------------------------------------------------------------------#


def to_np(x):
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    return x


def to_torch(x, dtype=None, device=None):
    dtype = dtype or DTYPE
    device = device or DEVICE
    if type(x) is dict:
        return {k: to_torch(v, dtype, device) for k, v in x.items()}
    elif torch.is_tensor(x):
        return x.to(device).type(dtype)
        # import pdb; pdb.set_trace()
    return torch.tensor(x, dtype=dtype, device=device)


def to_device(x, device=DEVICE):
    if torch.is_tensor(x):
        return x.to(device)
    elif type(x) is dict:
        return {k: to_device(v, device) for k, v in x.items()}
    else:
        print(f"Unrecognized type in `to_device`: {type(x)}")
        pdb.set_trace()
    # return [x.to(device) for x in xs]


# def atleast_2d(x, axis=0):
# 	'''
# 		works for both np arrays and torch tensors
# 	'''
# 	while len(x.shape) < 2:
# 		shape = (1, *x.shape) if axis == 0 else (*x.shape, 1)
# 		x = x.reshape(*shape)
# 	return x

# def to_2d(x):
# 	dim = x.shape[-1]
# 	return x.reshape(-1, dim)


def batchify(batch, device):
    """
    convert a single dataset item to a batch suitable for passing to a model by
            1) converting np arrays to torch tensors and
            2) and ensuring that everything has a batch dimension
    """
    fn = lambda x: to_torch(x[None], device=device)

    batched_vals = {}
    for field in batch.keys():
        val = batch[field]
        val = apply_dict(fn, val) if type(val) is dict else fn(val)
        batched_vals[field] = val
    return batched_vals


def apply_dict(fn, d, *args, **kwargs):
    return {k: fn(v, *args, **kwargs) for k, v in d.items()}


def remove_player_info(x, condition):
    if (
        "player_idxs" in condition
        and "player_hoop_sides" in condition
        and x.shape[-1] == 4
    ):  # must have add player info
        x = x[:, :, :, 1:-1]
    return x


def normalize(x):
    """
    scales `x` to [0, 1]
    """
    x = x - x.min()
    x = x / x.max()
    return x


def to_img(x):
    normalized = normalize(x)
    array = to_np(normalized)
    array = np.transpose(array, (1, 2, 0))
    return (array * 255).astype(np.uint8)


def set_device(device):
    global DEVICE
    DEVICE = device
    if "cuda" in device:
        torch.set_default_tensor_type(torch.cuda.FloatTensor)


def batch_to_device(batch, device="cuda:0"):
    device_batch = {k: to_device(batch[k], device) for k in batch.keys()}
    return device_batch


def _to_str(num):
    if num >= 1e6:
        return f"{(num/1e6):.2f} M"
    else:
        return f"{(num/1e3):.2f} k"


# -----------------------------------------------------------------------------#
# ----------------------------- parameter counting ----------------------------#
# -----------------------------------------------------------------------------#


def param_to_module(param):
    module_name = param[::-1].split(".", maxsplit=1)[-1][::-1]
    return module_name


def report_parameters(model, topk=10):
    counts = {k: p.numel() for k, p in model.named_parameters()}
    n_parameters = sum(counts.values())
    print(f"[ utils/arrays ] Total parameters: {_to_str(n_parameters)}")

    modules = dict(model.named_modules())
    sorted_keys = sorted(counts, key=lambda x: -counts[x])
    for i in range(min(topk, len(modules))):
        key = sorted_keys[i]
        count = counts[key]
        module = param_to_module(key)
        print(" " * 8, f"{key:10}: {_to_str(count)} | {modules[module]}")

    remaining_parameters = sum([counts[k] for k in sorted_keys[topk:]])
    print(
        " " * 8,
        f"... and {len(counts)-topk} others accounting for {_to_str(remaining_parameters)} parameters",
    )
    return n_parameters
