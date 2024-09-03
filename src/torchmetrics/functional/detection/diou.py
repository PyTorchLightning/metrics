# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Optional

import torch

from torchmetrics.utilities.imports import _TORCHVISION_AVAILABLE

if not _TORCHVISION_AVAILABLE:
    __doctest_skip__ = ["distance_intersection_over_union"]


def _diou_update(
    preds: torch.Tensor, target: torch.Tensor, iou_threshold: Optional[float], replacement_val: float = 0
) -> torch.Tensor:
    if preds.ndim != 2 or preds.shape[-1] != 4:
        raise ValueError(f"Expected preds to be of shape (N, 4) but got {preds.shape}")
    if target.ndim != 2 or target.shape[-1] != 4:
        raise ValueError(f"Expected target to be of shape (N, 4) but got {target.shape}")

    from torchvision.ops import distance_box_iou

    iou = distance_box_iou(preds, target)
    if iou_threshold is not None:
        iou[iou < iou_threshold] = replacement_val
    return iou


def _diou_compute(iou: torch.Tensor, aggregate: bool = True) -> torch.Tensor:
    if not aggregate:
        return iou
    return iou.diag().mean() if iou.numel() > 0 else torch.tensor(0.0, device=iou.device)


def distance_intersection_over_union(
    preds: torch.Tensor,
    target: torch.Tensor,
    iou_threshold: Optional[float] = None,
    replacement_val: float = 0,
    aggregate: bool = True,
) -> torch.Tensor:
    r"""Compute Distance Intersection over Union (`DIOU`_) between two sets of boxes.

    Both sets of boxes are expected to be in (x1, y1, x2, y2) format with 0 <= x1 < x2 and 0 <= y1 < y2.

    Args:
        preds:
            The input tensor containing the predicted bounding boxes.
        target:
            The tensor containing the ground truth.
        iou_threshold:
            Optional IoU thresholds for evaluation. If set to `None` the threshold is ignored.
        replacement_val:
            Value to replace values under the threshold with.
        aggregate:
            Return the average value instead of the full matrix of values

    Example::
        By default diou is aggregated across all box pairs e.g. mean along the diagonal of the dIoU matrix:

        >>> import torch
        >>> from torchmetrics.functional.detection import distance_intersection_over_union
        >>> preds = torch.tensor(
        ...     [
        ...         [296.55, 93.96, 314.97, 152.79],
        ...         [328.94, 97.05, 342.49, 122.98],
        ...         [356.62, 95.47, 372.33, 147.55],
        ...     ]
        ... )
        >>> target = torch.tensor(
        ...     [
        ...         [300.00, 100.00, 315.00, 150.00],
        ...         [330.00, 100.00, 350.00, 125.00],
        ...         [350.00, 100.00, 375.00, 150.00],
        ...     ]
        ... )
        >>> distance_intersection_over_union(preds, target)
        tensor(0.5793)

    Example::
        By setting `aggregate=False` the IoU score per prediction and target boxes is returned:

        >>> import torch
        >>> from torchmetrics.functional.detection import distance_intersection_over_union
        >>> preds = torch.tensor(
        ...     [
        ...         [296.55, 93.96, 314.97, 152.79],
        ...         [328.94, 97.05, 342.49, 122.98],
        ...         [356.62, 95.47, 372.33, 147.55],
        ...     ]
        ... )
        >>> target = torch.tensor(
        ...     [
        ...         [300.00, 100.00, 315.00, 150.00],
        ...         [330.00, 100.00, 350.00, 125.00],
        ...         [350.00, 100.00, 375.00, 150.00],
        ...     ]
        ... )
        >>> distance_intersection_over_union(preds, target, aggregate=False)
        tensor([[ 0.6883, -0.2043, -0.3351],
                [-0.2214,  0.4886, -0.1913],
                [-0.3971, -0.1510,  0.5609]])

    """
    if not _TORCHVISION_AVAILABLE:
        raise ModuleNotFoundError(
            f"`{complete_intersection_over_union.__name__}` requires that `torchvision` is installed."
            " Please install with `pip install torchmetrics[detection]`."
        )
    iou = _diou_update(preds, target, iou_threshold, replacement_val)
    return _diou_compute(iou, aggregate)
