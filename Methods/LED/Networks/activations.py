#!/usr/bin/env python3
# # -*- coding: utf-8 -*-
"""Created by: Vlachas Pantelis, CSE-lab, ETH Zurich
"""
#!/usr/bin/env python

# TORCH
import torch
import torch.nn as nn

torch_activations = {
    "relu": nn.ReLU(),
    "selu": nn.SELU(),
    "elu": nn.ELU(),
    "celu": nn.CELU(),
    "tanh": nn.Tanh(),
    "sigmoid": nn.Sigmoid(),
    "identity": nn.Identity(),
}


def getActivation(str_):
    return torch_activations[str_]
