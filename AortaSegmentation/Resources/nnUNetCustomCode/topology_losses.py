import torch
import torch.nn as nn
import torch.nn.functional as F
from nnunetv2.utilities.helpers import softmax_helper_dim1


# ---------------------------------------------------------------------------
# clDice loss
# ---------------------------------------------------------------------------

def _soft_erode(x: torch.Tensor) -> torch.Tensor:
    """Single step of soft morphological erosion via 3D min-pooling."""
    return -F.max_pool3d(-x, kernel_size=3, stride=1, padding=1)


def _soft_dilate(x: torch.Tensor) -> torch.Tensor:
    return F.max_pool3d(x, kernel_size=3, stride=1, padding=1)


def _soft_open(x: torch.Tensor) -> torch.Tensor:
    return _soft_dilate(_soft_erode(x))


def _soft_skeleton(x: torch.Tensor, iters: int = 5) -> torch.Tensor:
    """Soft skeleton via iterative erosion-difference."""
    skel = F.relu(x - _soft_open(x))
    for _ in range(iters - 1):
        x = _soft_erode(x)
        skel = skel + F.relu(x - _soft_open(x))
    return skel


class SoftClDiceLoss(nn.Module):
    """
    Connectivity-aware loss based on soft skeletons.
    Reference: Shit et al., clDice (CVPR 2021).
    Operates on raw logits (applies softmax internally).
    """

    def __init__(self, smooth: float = 1e-5, iters: int = 5):
        super().__init__()
        self.smooth = smooth
        self.iters = iters

    def forward(self, net_output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # net_output: (B, C, D, H, W) logits; target: (B, 1, D, H, W) integer labels
        probs = softmax_helper_dim1(net_output)
        pred_fg = probs[:, 1:2]                          # foreground probability
        gt_fg = (target == 1).float()                    # binary ground truth

        skel_pred = _soft_skeleton(pred_fg, self.iters)
        skel_gt = _soft_skeleton(gt_fg, self.iters)

        # Topology precision: how much of the predicted skeleton overlaps GT
        tprec = (skel_pred * gt_fg).sum() / (skel_pred.sum() + self.smooth)
        # Topology sensitivity: how much of the GT skeleton is covered by prediction
        tsens = (skel_gt * pred_fg).sum() / (skel_gt.sum() + self.smooth)

        cldice = 1.0 - 2.0 * tprec * tsens / (tprec + tsens + self.smooth)
        return cldice


# ---------------------------------------------------------------------------
# Combined loss: DC+CE + clDice
# ---------------------------------------------------------------------------

class AortaTopologyLoss(nn.Module):
    """
    Wraps an existing base loss (DC+CE) and adds clDice.
    Only applied to the highest-resolution output when deep supervision is active.
    """

    def __init__(
        self,
        base_loss: nn.Module,
        weight_cldice: float = 0.2,
        cldice_iters: int = 5,
    ):
        super().__init__()
        self.base_loss = base_loss
        self.weight_cldice = weight_cldice
        self.cldice = SoftClDiceLoss(iters=cldice_iters)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = self.base_loss(net_output, target)

        if self.weight_cldice > 0:
            loss = loss + self.weight_cldice * self.cldice(net_output, target)

        return loss
