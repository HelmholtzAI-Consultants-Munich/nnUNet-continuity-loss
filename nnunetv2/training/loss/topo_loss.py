from typing import Callable

import torch

from nnunetv2.utilities.ddp_allgather import AllGatherGrad
from torch import nn
import torch_topological
import torch_topological.nn

class WassersteinPersistentHomologyLoss(nn.Module):
    def __init__(self, apply_nonlin: Callable = None, complex_class = torch_topological.nn.CubicalComplex):
        super(WassersteinPersistentHomologyLoss, self).__init__()

        self.apply_nonlin = apply_nonlin
        self.complex = complex_class()
        self.distance_metric = torch_topological.nn.WassersteinDistance()

    def forward(self, x, y, loss_mask=None):
        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        dim = x.ndim-2
        self.complex.dim = dim


        with torch.no_grad():
            if x.ndim != y.ndim:
                y = y.view((y.shape[0], 1, *y.shape[1:]))

            if x.shape == y.shape:
                y_onehot = y
            else:
                y_onehot = torch.zeros(x.shape, device=x.device, dtype=torch.bool)
                y_onehot.scatter_(1, y.long(), 1)
        

        # it makes no sense to do background for topology

        y_onehot = y_onehot.type_as(x)
        if dim == 2:
            x_top_inp = torch.nn.functional.adaptive_avg_pool2d(x[:, 1:], (32, 32))
            y_top_np = torch.nn.functional.adaptive_avg_pool2d(y_onehot[:, 1:], (32, 32))
        elif dim == 3:
            x_top_inp = torch.nn.functional.adaptive_avg_pool3d(x[:, 1:], (4, 16, 16))
            y_top_np = torch.nn.functional.adaptive_avg_pool3d(y_onehot[:, 1:], (4, 16, 16))
        complex_x = self.complex(x_top_inp)
        complex_y = self.complex(y_top_np)

        distance = torch.tensor(0.0, device=x.device)
        for layer_x, layer_y in zip(complex_x, complex_y):
            for diagram_x, diagram_y in zip(layer_x, layer_y):
                distance = distance + self.distance_metric(diagram_x, diagram_y)

        distance = distance / len(complex_x)
        return distance