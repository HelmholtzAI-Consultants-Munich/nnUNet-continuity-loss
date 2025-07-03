import torch
from nnunetv2.training.loss.dice import SoftDiceLoss, MemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.cldice import SoftCLDiceLoss
from nnunetv2.training.loss.loss_scalers import graphmatch_torch_adapter
from nnunetv2.training.loss.topo_loss import WassersteinPersistentHomologyLoss
from nnunetv2.training.loss.robust_ce_loss import RobustCrossEntropyLoss, TopKLoss
from nnunetv2.utilities.helpers import softmax_helper_dim1
from torch import nn

class DC_and_CE_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, ignore_label=None,
                 dice_class=SoftDiceLoss, loss_scaler=None, loss_scaler_weight=0.8):
        """
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super(DC_and_CE_loss, self).__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label

        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)
        self.loss_scaler = loss_scaler
        self.loss_scaler_weight = loss_scaler_weight

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """            
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(DC_and_CE_loss)'
            mask = target != self.ignore_label
            # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            target_dice = torch.where(mask, target, 0)
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None

        dc_loss = self.dc(net_output, target_dice, loss_mask=mask) \
            if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target[:, 0]) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0
        if self.loss_scaler:
            with torch.no_grad():
                out_nl = softmax_helper_dim1(net_output)
                out_nl = torch.argmax(out_nl, dim=1, keepdim=True)
                one_hot = torch.zeros_like(net_output)
                one_hot.scatter_(1, out_nl, 1.0)
                loss_mask = self.loss_scaler(one_hot, target, do_bg=False)
            
            if torch.count_nonzero(loss_mask) == 0:
                scaler_loss = 0
            else:
                # move the channel axis to the last position, to prepare it for cross entropy computation
                logits_flat = net_output.movedim(1, -1) # (batch, c, z, y, x) -> (batch, z, y, x, c)
                logits_flat = logits_flat.reshape(-1, logits_flat.shape[-1]) # -> (batch*z*y*x, c)
                # target tensor is just (batch, 1, z, y, x)
                targets_flat = target[:,0].long().view(-1) # -> (batch*z*y*x)

                # take all samples into consideration whenever any channel is set to true
                loss_mask_samples = loss_mask.any(dim=1, keepdims=False) # (batch, c, z, y, x) -> (batch, z, y, x)
                loss_mask_flat = loss_mask_samples.view(-1) # -> (batch*z*y*x)

                logits_masked = logits_flat[loss_mask_flat]
                targets_masked = targets_flat[loss_mask_flat]

                # this is likely to be larger than ce loss, and this is okay!
                # after all, we're computing mean cross entropy for pixels that are likely to be a mismatch!
                scaler_loss = nn.functional.cross_entropy(logits_masked, targets_masked)
        else:
            scaler_loss = 0
        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss + scaler_loss * self.weight_ce * self.loss_scaler_weight
        return result


class DC_and_BCE_loss(nn.Module):
    def __init__(self, bce_kwargs, soft_dice_kwargs, weight_ce=1, weight_dice=1, use_ignore_label: bool = False,
                 dice_class=MemoryEfficientSoftDiceLoss, loss_scaler=None, loss_scaler_weight=0.8):
        """
        DO NOT APPLY NONLINEARITY IN YOUR NETWORK!

        target mut be one hot encoded
        IMPORTANT: We assume use_ignore_label is located in target[:, -1]!!!

        :param soft_dice_kwargs:
        :param bce_kwargs:
        :param aggregate:
        """
        super(DC_and_BCE_loss, self).__init__()
        if use_ignore_label:
            bce_kwargs['reduction'] = 'none'

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.use_ignore_label = use_ignore_label

        self.ce = nn.BCEWithLogitsLoss(**bce_kwargs)
        self.dc = dice_class(apply_nonlin=torch.sigmoid, **soft_dice_kwargs)

        self.loss_scaler = loss_scaler
        self.loss_scaler_weight = loss_scaler_weight

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        if self.use_ignore_label:
            # target is one hot encoded here. invert it so that it is True wherever we can compute the loss
            if target.dtype == torch.bool:
                mask = ~target[:, -1:]
            else:
                mask = (1 - target[:, -1:]).bool()
            # remove ignore channel now that we have the mask
            # why did we use clone in the past? Should have documented that...
            # target_regions = torch.clone(target[:, :-1])
            target_regions = target[:, :-1]
        else:
            target_regions = target
            mask = None

        dc_loss = self.dc(net_output, target_regions, loss_mask=mask)
        target_regions = target_regions.float()
        if mask is not None:
            ce_loss = (self.ce(net_output, target_regions) * mask).sum() / torch.clip(mask.sum(), min=1e-8)
        else:
            ce_loss = self.ce(net_output, target_regions)


        if self.loss_scaler:
            with torch.no_grad():
                out_nl = softmax_helper_dim1(net_output)
                out_nl = torch.argmax(out_nl, dim=1, keepdim=True)
                one_hot = torch.zeros_like(net_output)
                one_hot.scatter_(1, out_nl, 1.0)
                loss_mask = self.loss_scaler(one_hot, target, do_bg=False)
            
            if torch.count_nonzero(loss_mask) == 0:
                scaler_loss = 0
            else:
                # move the channel axis to the last position, to prepare it for cross entropy computation
                logits_flat = net_output.movedim(1, -1) # (batch, c, z, y, x) -> (batch, z, y, x, c)
                logits_flat = logits_flat.reshape(-1, logits_flat.shape[-1]) # -> (batch*z*y*x, c)
                # target tensor is just (batch, 1, z, y, x)
                targets_flat = target[:,0].long().view(-1) # -> (batch*z*y*x)

                # take all samples into consideration whenever any channel is set to true
                loss_mask_samples = loss_mask.any(dim=1, keepdims=False) # (batch, c, z, y, x) -> (batch, z, y, x)
                loss_mask_flat = loss_mask_samples.view(-1) # -> (batch*z*y*x)

                logits_masked = logits_flat[loss_mask_flat]
                targets_masked = targets_flat[loss_mask_flat]

                # this is likely to be larger than ce loss, and this is okay!
                # after all, we're computing mean cross entropy for pixels that are likely to be a mismatch!
                scaler_loss = nn.functional.cross_entropy(logits_masked, targets_masked)
        else:
            scaler_loss = 0
            

        if self.loss_scaler:
            with torch.nograd():
                out_nl = torch.sigmoid(net_output) > 0.5
                loss_mask = self.loss_scaler(out_nl, target_regions, do_bg=True)

            if torch.count_nonzero(loss_mask) == 0:
                scaler_loss = 0
            else:
                # our channel axis is one anyway, we can just get rid of it
                logits_flat = net_output.reshape(-1, logits_flat.shape[-1]) # -> (batch*z*y*x, )
                targets_flat = target[:,0].long().view(-1) # -> (batch*z*y*x)

                loss_mask_flat = loss_mask_samples.view(-1) # channel dim = 1 as well

                logits_masked = logits_flat[loss_mask_flat]
                targets_masked = targets_flat[loss_mask_flat]

                scaler_loss = nn.functional.binary_cross_entropy_with_logits(logits_masked, targets_masked)
        else:
            scaler_loss = 0
            

        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss + scaler_loss * self.weight_ce * self.loss_scaler_weight
        return result


class DC_and_topk_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, ignore_label=None):
        """
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super().__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label

        self.ce = TopKLoss(**ce_kwargs)
        self.dc = SoftDiceLoss(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(DC_and_CE_loss)'
            mask = (target != self.ignore_label).bool()
            # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            target_dice = torch.clone(target)
            target_dice[target == self.ignore_label] = 0
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None

        dc_loss = self.dc(net_output, target_dice, loss_mask=mask) \
            if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0

        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result

    
class DC_and_clDC_and_CE_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, cl_dice_kwargs, ce_kwargs, weight_dice=1, weight_cldice=1, weight_ce=1, ignore_label=None):
        super().__init__()
        self.weight_cldice = weight_cldice
        self.dc_and_ce_loss = DC_and_CE_loss(soft_dice_kwargs, ce_kwargs, weight_ce, weight_dice, ignore_label=ignore_label)
        self.cldc = SoftCLDiceLoss(**cl_dice_kwargs, apply_nonlin=softmax_helper_dim1)
        self.ignore_label = ignore_label

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(DC_and_CE_loss)'
            mask = target != self.ignore_label
            target_dice = torch.where(mask, target, 0)
        else:
            target_dice = target
            mask = None

        cldc_loss = self.cldc(net_output, target_dice, loss_mask=mask) if self.weight_cldice else 0
        dc_and_ce_loss = self.dc_and_ce_loss(net_output, target)

        return dc_and_ce_loss + self.weight_cldice * cldc_loss

class DC_and_clDC_and_BCE_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, cl_dice_kwargs, bce_kwargs, weight_dice=1, weight_cldice=1, weight_ce=1, use_ignore_label: bool = False):
        super().__init__()
        self.weight_cldice = weight_cldice
        self.dc_and_bce_loss = DC_and_BCE_loss(bce_kwargs, soft_dice_kwargs, weight_ce, weight_dice, use_ignore_label)
        self.cldc = SoftCLDiceLoss(**cl_dice_kwargs, apply_nonlin=torch.sigmoid)
    
    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        if self.use_ignore_label:
            # target is one hot encoded here. invert it so that it is True wherever we can compute the loss
            if target.dtype == torch.bool:
                mask = ~target[:, -1:]
            else:
                mask = (1 - target[:, -1:]).bool()
            # remove ignore channel now that we have the mask
            # why did we use clone in the past? Should have documented that...
            # target_regions = torch.clone(target[:, :-1])
            target_regions = target[:, :-1]
        else:
            target_regions = target
            mask = None
        cldc_loss = self.cldc(net_output, target_regions, loss_mask=mask) if self.weight_cldice else 0
        dc_and_bce_loss = self.dc_and_bce_loss(net_output, target)
        return dc_and_bce_loss + self.weight_cldice * cldc_loss

class DC_and_CE_and_topo_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, weight_topo=1, ignore_label=None,
                 dice_class=SoftDiceLoss):
        super().__init__()
        self.weight_topo = weight_topo
        self.dc_and_ce_loss = DC_and_CE_loss(soft_dice_kwargs, ce_kwargs, weight_ce, weight_dice, ignore_label, dice_class)
        self.topo_loss = WassersteinPersistentHomologyLoss(apply_nonlin=softmax_helper_dim1)   

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        topo_loss = self.topo_loss(net_output, target[:, 0])
        dc_and_ce_loss = self.dc_and_ce_loss(net_output, target)
        return dc_and_ce_loss + self.weight_topo * topo_loss
    
class DC_and_BCE_and_topo_loss(nn.Module):
    def __init__(self, bce_kwargs, soft_dice_kwargs, weight_ce=1, weight_dice=1, weight_topo=1, use_ignore_label: bool = False,
                 dice_class=MemoryEfficientSoftDiceLoss):
        super().__init__()
        self.dc_and_bce_loss = DC_and_BCE_loss(bce_kwargs, soft_dice_kwargs, weight_ce, weight_dice, use_ignore_label, dice_class)
        self.topo_loss = WassersteinPersistentHomologyLoss(apply_nonlin=torch.sigmoid)
        self.weight_topo = weight_topo

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        topo_loss = self.topo_loss(net_output, target)
        dc_and_bce_loss = self.dc_and_bce_loss(net_output, target)
        return  dc_and_bce_loss + self.weight_topo * topo_loss



# class DC_and_clDC_loss(nn.Module):
#     def __init__(self, soft_dice_kwargs, cl_dice_kwargs, weight_dice=1, weight_cldice=1, use_ignore_label: bool = False):
#         """
#         Weights for clDice and Dice do not need to sum to one. You can set whatever you want.
#         :param soft_dice_kwargs:
#         :param cl_dice_kwargs:
#         :param weight_dice:
#         :param weight_cldice:
#         """
#         super().__init__()

#         self.weight_dice = weight_dice
#         self.weight_cldice = weight_cldice

#         self.dc = MemoryEfficientSoftDiceLoss(**soft_dice_kwargs, apply_nonlin=softmax_helper_dim1)
#         self.cldc = SoftCLDiceLoss(**cl_dice_kwargs, apply_nonlin=softmax_helper_dim1)

#         self.use_ignore_label = use_ignore_label

#     def forward(self, net_output: torch.Tensor, target: torch.Tensor):
#         """
#         target must be b, c, x, y(, z). Behaviour for c != 1 is not explored.
#         :param net_output:
#         :param target:
#         :return:
#         """

#         if self.use_ignore_label:
#             # target is one hot encoded here. invert it so that it is True wherever we can compute the loss
#             if target.dtype == torch.bool:
#                 mask = ~target[:, -1:]
#             else:
#                 mask = (1 - target[:, -1:]).bool()
#             # remove ignore channel now that we have the mask
#             # why did we use clone in the past? Should have documented that...
#             # target_regions = torch.clone(target[:, :-1])
#             target_regions = target[:, :-1]
#         else:
#             target_regions = target
#             mask = None

#         dc_loss = self.dc(net_output, target_regions, loss_mask=mask) if self.weight_dice else 0
#         cldc_loss = self.cldc(net_output, target_regions, loss_mask=mask) if self.weight_cldice else 0

#         result = dc_loss*self.weight_dice + cldc_loss*self.weight_cldice

#         return result

# class DC_and_clDC_and_CE_loss(nn.Module):
#     def __init__(self, soft_dice_kwargs, cl_dice_kwargs, ce_kwargs, weight_dice=1, weight_cldice=1, weight_ce=1, ignore_label=None):
#         """
#         Weights for clDice and Dice do not need to sum to one. You can set whatever you want.
#         :param soft_dice_kwargs:
#         :param cl_dice_kwargs:
#         :param weight_dice:
#         :param weight_cldice:
#         """
#         super().__init__()

#         self.weight_dice = weight_dice
#         self.weight_cldice = weight_cldice
#         self.weight_ce = weight_ce

#         self.dc = MemoryEfficientSoftDiceLoss(**soft_dice_kwargs, apply_nonlin=softmax_helper_dim1)
#         self.cldc = SoftCLDiceLoss(**cl_dice_kwargs, apply_nonlin=softmax_helper_dim1)
#         self.ce = RobustCrossEntropyLoss(**ce_kwargs)

#         self.ignore_label = ignore_label

#     def forward(self, net_output: torch.Tensor, target: torch.Tensor):
#         """
#         target must be b, c, x, y(, z). Behaviour for c != 1 is not explored.
#         :param net_output:
#         :param target:
#         :return:
#         """

#         if self.ignore_label is not None:
#             assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
#                                          '(DC_and_CE_loss)'
#             mask = target != self.ignore_label
#             # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
#             # ignore gradients in those areas anyway
#             target_dice = torch.where(mask, target, 0)
#             num_fg = mask.sum()
#         else:
#             target_dice = target
#             mask = None

#         dc_loss = self.dc(net_output, target_dice, loss_mask=mask) if self.weight_dice else 0
#         cldc_loss = self.cldc(net_output, target_dice, loss_mask=mask) if self.weight_cldice else 0
#         ce_loss = self.ce(net_output, target[:, 0]) if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0

#         result = dc_loss*self.weight_dice + cldc_loss*self.weight_cldice + ce_loss*self.weight_ce

#         return result

# class DC_and_clDC_and_BCE_loss(nn.Module):
#     def __init__(self, soft_dice_kwargs, cl_dice_kwargs, ce_kwargs, weight_dice=1, weight_cldice=1, weight_ce=1, use_ignore_label: bool = False):
#         """
#         Weights for clDice and Dice do not need to sum to one. You can set whatever you want.
#         :param soft_dice_kwargs:
#         :param cl_dice_kwargs:
#         :param weight_dice:
#         :param weight_cldice:
#         """
#         super().__init__()

#         self.weight_dice = weight_dice
#         self.weight_cldice = weight_cldice
#         self.weight_ce = weight_ce

#         self.dc = MemoryEfficientSoftDiceLoss(**soft_dice_kwargs, apply_nonlin=torch.sigmoid)
#         self.cldc = SoftCLDiceLoss(**cl_dice_kwargs, apply_nonlin=torch.sigmoid)
#         self.ce = nn.BCEWithLogitsLoss(**ce_kwargs)

#         self.use_ignore_label = use_ignore_label


#     def forward(self, net_output: torch.Tensor, target: torch.Tensor):
#         if self.use_ignore_label:
#             # target is one hot encoded here. invert it so that it is True wherever we can compute the loss
#             if target.dtype == torch.bool:
#                 mask = ~target[:, -1:]
#             else:
#                 mask = (1 - target[:, -1:]).bool()
#             # remove ignore channel now that we have the mask
#             # why did we use clone in the past? Should have documented that...
#             # target_regions = torch.clone(target[:, :-1])
#             target_regions = target[:, :-1]
#         else:
#             target_regions = target
#             mask = None

#         dc_loss = self.dc(net_output, target_regions, loss_mask=mask)
#         target_regions = target_regions.float()
#         if mask is not None:
#             ce_loss = (self.ce(net_output, target_regions) * mask).sum() / torch.clip(mask.sum(), min=1e-8)
#         else:
#             ce_loss = self.ce(net_output, target_regions)
#         result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
#         return result

#     def forward(self, net_output: torch.Tensor, target: torch.Tensor):
#         if self.use_ignore_label:
#             # target is one hot encoded here. invert it so that it is True wherever we can compute the loss
#             if target.dtype == torch.bool:
#                 mask = ~target[:, -1:]
#             else:
#                 mask = (1 - target[:, -1:]).bool()
#             # remove ignore channel now that we have the mask
#             # why did we use clone in the past? Should have documented that...
#             # target_regions = torch.clone(target[:, :-1])
#             target_regions = target[:, :-1]
#         else:
#             target_regions = target
#             mask = None

#         dc_loss = self.dc(net_output, target_regions, loss_mask=mask)
#         cldc_loss = self.cldc(net_output, target_regions, loss_mask=mask) if self.weight_cldice else 0
#         target_regions = target_regions.float()
#         if mask is not None:
#             ce_loss = (self.ce(net_output, target_regions) * mask).sum() / torch.clip(mask.sum(), min=1e-8)
#         else:
#             ce_loss = self.ce(net_output, target_regions)

#         result = self.weight_ce * ce_loss + self.weight_dice * dc_loss + self.weight_cldice * cldc_loss
#         return result