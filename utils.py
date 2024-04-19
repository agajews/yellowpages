import torch
import torch.nn.functional as F


def resolve_activation_fn(desc):
    if isinstance(desc, str):
        return getattr(F, desc)
    return desc


def as_tensor(x):
    if not isinstance(x, torch.Tensor):
        return torch.tensor(x)
    return x


def num_params(model):
    return sum(param.numel() for param in model.parameters())