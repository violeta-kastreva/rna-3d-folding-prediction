import torch
from torch import sin, cos, atan2, acos
from functools import wraps
from contextlib import contextmanager


def exists(val):
    return val is not None


def default(val, d):
    return val if val is not None else d


def cast_torch_tensor(fn):
    @wraps(fn)
    def inner(t):
        if not torch.is_tensor(t):
            t = torch.tensor(t, dtype = torch.get_default_dtype())
        return fn(t)
    return inner


@cast_torch_tensor
def rot_z(gamma):
    return torch.tensor([
        [cos(gamma), -sin(gamma), 0],
        [sin(gamma), cos(gamma),  0],
        [0,          0,           1]
    ], dtype=gamma.dtype)


@cast_torch_tensor
def rot_y(beta):
    return torch.tensor([
        [cos(beta),  0, sin(beta)],
        [0,          1, 0],
        [-sin(beta), 0, cos(beta)]
    ], dtype=beta.dtype)


def rot(alpha, beta, gamma):
    return rot_z(alpha) @ rot_y(beta) @ rot_z(gamma)


@contextmanager
def disable_tf32():
    orig_value = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    yield
    torch.backends.cuda.matmul.allow_tf32 = orig_value
