from typing import Callable

import torch
from nnunetv2.utilities.ddp_allgather import AllGatherGrad
from torch import nn


class SoftCLDiceLoss(nn.Module):
    def __init__(self, apply_nonlin: Callable = None, batch_dice: bool = False, do_bg: bool = True, k: int = 10,
                  ddp: bool = True, eps: float = 1e-8, simplify_diff: bool = False):
        super(SoftCLDiceLoss, self).__init__()

        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.k = k
        self.ddp = ddp
        self.eps = eps
        self.simplify_diff = simplify_diff

    def forward(self, x, y, loss_mask=None):
        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        axes = tuple(range(2, x.ndim))

        with torch.no_grad():
            if x.ndim != y.ndim:
                y = y.view((y.shape[0], 1, *y.shape[1:]))

            if x.shape == y.shape:
                # if this is the case then gt is probably already a one hot encoding
                y_onehot = y
            else:
                y_onehot = torch.zeros(x.shape, device=x.device, dtype=torch.bool)
                y_onehot.scatter_(1, y.long(), 1)

            if not self.do_bg:
                y_onehot = y_onehot[:, 1:]

        if not self.do_bg:
            x = x[:, 1:]

        if self.simplify_diff:
            cl_label = self._soft_centerline(y_onehot)
            missing_cl = cl_label * (1-x)
            missing_cl = missing_cl.sum(axes) / (cl_label.sum(axes) + self.eps)
            if self.batch_dice:
                if self.ddp:
                    missing_cl = AllGatherGrad.apply(missing_cl).sum(0)
                missing_cl = missing_cl.sum(0)
            
            return missing_cl
        else:
            # computes the full CL dice
            cl_pred = self._soft_centerline(x)
            cl_label = self._soft_centerline(y_onehot)

            if loss_mask is None:
                cl_p_2vol_l = ((cl_pred * y_onehot).sum(axes) + self.eps) / ((cl_pred).sum(axes) + self.eps)
                cl_l_2vol_p = ((cl_label* x).sum(axes) + self.eps) / ((cl_label).sum(axes) + self.eps)
            else:
                cl_p_2vol_l = ((cl_pred * y_onehot * loss_mask).sum(axes) + self.eps) / ((cl_pred* loss_mask).sum(axes) + self.eps)
                cl_l_2vol_p = ((cl_label* x * loss_mask).sum(axes) + self.eps) / ((cl_label*loss_mask).sum(axes) + self.eps)
            if self.batch_dice:
                if self.ddp:
                    cl_p_2vol_l = AllGatherGrad.apply(cl_p_2vol_l).sum(0)
                    cl_l_2vol_p = AllGatherGrad.apply(cl_l_2vol_p).sum(0)

                cl_p_2vol_l = cl_p_2vol_l.sum(0)
                cl_l_2vol_p = cl_l_2vol_p.sum(0)

            dc = (2 * cl_p_2vol_l * cl_l_2vol_p) / (cl_l_2vol_p + cl_p_2vol_l)

            return -dc.mean()

    # what we want is a pool with a diamond kernel
    # 0 1 0
    # 1 1 1
    # 0 1 0
    # sadly, there is no such thing in pytorch, so we have to do it manually
    def _cl_minpool_2d(self, x, k):
        p1 = -nn.functional.max_pool2d(-x, (k, 1), (1, 1), (1, 0))
        p2 = -nn.functional.max_pool2d(-x, (1, k), (1, 1), (0, 1))
        return torch.min(p1, p2)

    def _cl_minpool_3d(self, x, k):
        p1 = -nn.functional.max_pool3d(-x, (k, 1, 1), (1, 1, 1), (1, 0, 0))
        p2 = -nn.functional.max_pool3d(-x, (1, k, 1), (1, 1, 1), (0, 1, 0))
        p3 = -nn.functional.max_pool3d(-x, (1, 1, k), (1, 1, 1), (0, 0, 1))
        return torch.min(p1, torch.min(p2, p3))

    def _soft_centerline(self, label):
        # make sure label is not a bool tensor
        label = label.to(torch.float)
        # (b, c, x, y(, z)))
        if label.ndim == 5:
            maxpool = lambda x, k: nn.functional.max_pool3d(x, k, stride=1, padding=1)
            minpool = self._cl_minpool_3d
        else:
            maxpool = lambda x, k: nn.functional.max_pool2d(x, k, stride=1, padding=1)
            minpool = self._cl_minpool_2d
        lp = maxpool(minpool(label, 3), 3)
        cl = nn.functional.relu(label-lp)
        # slight optimisation with respect to the paper
        label = minpool(label, 3)
        while torch.sum(label) != 0:
            # uses one more variable but avoids one pooling operation per loop
            next_label = minpool(label, 3)
            lp = maxpool(next_label, 3)
            cl = cl + nn.functional.relu((1-cl) * nn.functional.relu(label-lp))
            label = next_label
        cl = minpool(maxpool(cl, 3), 3)
        return cl