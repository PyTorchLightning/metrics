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
from typing import Optional, Union

import torch
from torch import Tensor
from typing_extensions import Literal

from torchmetrics.functional.segmentation.utils import _ignore_background
from torchmetrics.utilities.checks import _check_same_shape
from torchmetrics.utilities.compute import _safe_divide


def _dice_score_validate_args(
    num_classes: int,
    include_background: bool,
    average: Optional[Literal["micro", "macro", "weighted", "none"]] = "micro",
    input_format: Literal["one-hot", "index"] = "one-hot",
    zero_divide: Union[float, Literal["warn", "nan"]] = 1.0,
) -> None:
    """Validate the arguments of the metric."""
    if not isinstance(num_classes, int) or num_classes <= 0:
        raise ValueError(f"Expected argument `num_classes` must be a positive integer, but got {num_classes}.")
    if not isinstance(include_background, bool):
        raise ValueError(f"Expected argument `include_background` must be a boolean, but got {include_background}.")
    allowed_average = ["micro", "macro", "weighted", "none"]
    if average is not None and average not in allowed_average:
        raise ValueError(f"Expected argument `average` to be one of {allowed_average} or None, but got {average}.")
    if input_format not in ["one-hot", "index"]:
        raise ValueError(f"Expected argument `input_format` to be one of 'one-hot', 'index', but got {input_format}.")
    if zero_divide not in [1.0, 0.0, "warn", "nan"]:
        raise ValueError(
            f"Expected argument `zero_divide` to be one of 1.0, 0.0, 'warn', 'nan', but got {zero_divide}."
        )


def _dice_score_update(
    preds: Tensor,
    target: Tensor,
    num_classes: int,
    include_background: bool,
    input_format: Literal["one-hot", "index"] = "one-hot",
) -> tuple[Tensor, Tensor, Tensor]:
    """Update the state with the current prediction and target."""
    _check_same_shape(preds, target)

    if input_format == "index":
        preds = torch.nn.functional.one_hot(preds, num_classes=num_classes).movedim(-1, 1)
        target = torch.nn.functional.one_hot(target, num_classes=num_classes).movedim(-1, 1)

    if preds.ndim < 3:
        raise ValueError(f"Expected both `preds` and `target` to have at least 3 dimensions, but got {preds.ndim}.")

    if not include_background:
        preds, target = _ignore_background(preds, target)

    reduce_axis = list(range(2, target.ndim))
    intersection = torch.sum(preds * target, dim=reduce_axis)
    target_sum = torch.sum(target, dim=reduce_axis)
    pred_sum = torch.sum(preds, dim=reduce_axis)

    numerator = 2 * intersection
    denominator = pred_sum + target_sum
    support = target_sum
    return numerator, denominator, support


def _dice_score_compute(
    numerator: Tensor,
    denominator: Tensor,
    average: Optional[Literal["micro", "macro", "weighted", "none"]] = "micro",
    support: Optional[Tensor] = None,
    zero_division: Union[float, Literal["warn", "nan"]] = 1.0,
) -> Tensor:
    """Compute the Dice score from the numerator and denominator."""
    # If both numerator and denominator are 0, the dice score is 0
    if torch.all(numerator == 0) and torch.all(denominator == 0):
        return torch.tensor(0.0, device=numerator.device, dtype=torch.float)

    if average == "micro":
        numerator = torch.sum(numerator, dim=-1)
        denominator = torch.sum(denominator, dim=-1)
    dice = _safe_divide(numerator, denominator, zero_division=zero_division)
    if average == "macro":
        dice = torch.mean(dice, dim=-1)
    elif average == "weighted" and support is not None:
        weights = _safe_divide(support, torch.sum(support, dim=-1, keepdim=True), zero_division=zero_division)
        dice = torch.sum(dice * weights, dim=-1)
    return dice


def dice_score(
    preds: Tensor,
    target: Tensor,
    num_classes: int,
    include_background: bool = True,
    average: Optional[Literal["micro", "macro", "weighted", "none"]] = "micro",
    input_format: Literal["one-hot", "index"] = "one-hot",
) -> Tensor:
    """Compute the Dice score for semantic segmentation.

    Args:
        preds: Predictions from model
        target: Ground truth values
        num_classes: Number of classes
        include_background: Whether to include the background class in the computation
        average: The method to average the dice score. Options are ``"micro"``, ``"macro"``, ``"weighted"``, ``"none"``
          or ``None``. This determines how to average the dice score across different classes.
        input_format: What kind of input the function receives. Choose between ``"one-hot"`` for one-hot encoded tensors
          or ``"index"`` for index tensors

    Returns:
        The Dice score.

    Example (with one-hot encoded tensors):
        >>> from torch import randint
        >>> from torchmetrics.functional.segmentation import dice_score
        >>> preds = randint(0, 2, (4, 5, 16, 16))  # 4 samples, 5 classes, 16x16 prediction
        >>> target = randint(0, 2, (4, 5, 16, 16))  # 4 samples, 5 classes, 16x16 target
        >>> # dice score micro averaged over all classes
        >>> dice_score(preds, target, num_classes=5, average="micro")
        tensor([0.4842, 0.4968, 0.5053, 0.4902])
        >>> # dice score per sample and class
        >>> dice_score(preds, target, num_classes=5, average="none")
        tensor([[0.4724, 0.5185, 0.4710, 0.5062, 0.4500],
                [0.4571, 0.4980, 0.5191, 0.4380, 0.5649],
                [0.5428, 0.4904, 0.5358, 0.4830, 0.4724],
                [0.4715, 0.4925, 0.4797, 0.5267, 0.4788]])

    Example (with index tensors):
        >>> from torch import randint
        >>> from torchmetrics.functional.segmentation import dice_score
        >>> preds = randint(0, 5, (4, 16, 16))  # 4 samples, 5 classes, 16x16 prediction
        >>> target = randint(0, 5, (4, 16, 16))  # 4 samples, 5 classes, 16x16 target
        >>> # dice score micro averaged over all classes
        >>> dice_score(preds, target, num_classes=5, average="micro", input_format="index")
        tensor([0.2031, 0.1914, 0.2500, 0.2266])
        >>> # dice score per sample and class
        >>> dice_score(preds, target, num_classes=5, average="none", input_format="index")
        tensor([[0.1714, 0.2500, 0.1304, 0.2524, 0.2069],
                [0.1837, 0.2162, 0.0962, 0.2692, 0.1895],
                [0.3866, 0.1348, 0.2526, 0.2301, 0.2083],
                [0.1978, 0.2804, 0.1714, 0.1915, 0.2783]])

    """
    _dice_score_validate_args(num_classes, include_background, average, input_format)
    numerator, denominator, support = _dice_score_update(preds, target, num_classes, include_background, input_format)
    return _dice_score_compute(numerator, denominator, average, support=support)
