# Copyright The Lightning team.
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
from collections.abc import Sequence
from typing import Any, List, Optional, Union

from torch import Tensor
from typing_extensions import Literal

from torchmetrics.functional.segmentation.dice import (
    _dice_score_compute,
    _dice_score_update,
    _dice_score_validate_args,
)
from torchmetrics.metric import Metric
from torchmetrics.utilities.data import dim_zero_cat
from torchmetrics.utilities.imports import _MATPLOTLIB_AVAILABLE
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE

if not _MATPLOTLIB_AVAILABLE:
    __doctest_skip__ = ["DiceScore.plot"]


class DiceScore(Metric):
    r"""Compute `Dice Score`_.

    The metric can be used to evaluate the performance of image segmentation models. The Dice Score is defined as:

    ..math::
        DS = \frac{2 \sum_{i=1}^{N} t_i p_i}{\sum_{i=1}^{N} t_i + \sum_{i=1}^{N} p_i}

    where :math:`N` is the number of classes, :math:`t_i` is the target tensor, and :math:`p_i` is the prediction
    tensor. In general the Dice Score can be interpreted as the overlap between the prediction and target tensors
    divided by the total number of elements in the tensors.

    As input to ``forward`` and ``update`` the metric accepts the following input:

        - ``preds`` (:class:`~torch.Tensor`): An one-hot boolean tensor of shape ``(N, C, ...)`` with ``N`` being
          the number of samples and ``C`` the number of classes. Alternatively, an integer tensor of shape ``(N, ...)``
          can be provided, where the integer values correspond to the class index. The input type can be controlled
          with the ``input_format`` argument.
        - ``target`` (:class:`~torch.Tensor`): An one-hot boolean tensor of shape ``(N, C, ...)`` with ``N`` being
          the number of samples and ``C`` the number of classes. Alternatively, an integer tensor of shape ``(N, ...)``
          can be provided, where the integer values correspond to the class index. The input type can be controlled
          with the ``input_format`` argument.

    As output to ``forward`` and ``compute`` the metric returns the following output:

        - ``gds`` (:class:`~torch.Tensor`): The dice score. If ``average`` is set to ``None`` or ``"none"`` the output
          will be a tensor of shape ``(C,)`` with the dice score for each class. If ``average`` is set to
          ``"micro"``, ``"macro"``, or ``"weighted"`` the output will be a scalar tensor. The score is an average over
          all samples.

    Args:
        num_classes: The number of classes in the segmentation problem.
        include_background: Whether to include the background class in the computation.
        average: The method to average the dice score. Options are ``"micro"``, ``"macro"``, ``"weighted"``, ``"none"``
            or ``None``. This determines how to average the dice score across different classes.
        input_format: What kind of input the function receives. Choose between ``"one-hot"`` for one-hot encoded tensors
            or ``"index"`` for index tensors
        zero_division: The value to return when there is a division by zero. Options are 1.0, 0.0, "warn" or "nan".
            Setting it to "warn" behaves like 0.0 but will also create a warning.
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Raises:
        ValueError:
            If ``num_classes`` is not a positive integer
        ValueError:
            If ``include_background`` is not a boolean
        ValueError:
            If ``average`` is not one of ``"micro"``, ``"macro"``, ``"weighted"``, ``"none"`` or ``None``
        ValueError:
            If ``input_format`` is not one of ``"one-hot"`` or ``"index"``

    Example:
        >>> from torch import randint
        >>> from torchmetrics.segmentation import DiceScore
        >>> preds = randint(0, 2, (4, 5, 16, 16))  # 4 samples, 5 classes, 16x16 prediction
        >>> target = randint(0, 2, (4, 5, 16, 16))  # 4 samples, 5 classes, 16x16 target
        >>> dice_score = DiceScore(num_classes=5, average="micro")
        >>> dice_score(preds, target)
        tensor(0.4941)
        >>> dice_score = DiceScore(num_classes=5, average="none")
        >>> dice_score(preds, target)
        tensor([0.4860, 0.4999, 0.5014, 0.4885, 0.4915])

    """

    full_state_update: bool = False
    is_differentiable: bool = False
    higher_is_better: bool = True
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0

    numerator: List[Tensor]
    denominator: List[Tensor]
    support: List[Tensor]

    def __init__(
        self,
        num_classes: int,
        include_background: bool = True,
        average: Optional[Literal["micro", "macro", "weighted", "none"]] = "micro",
        input_format: Literal["one-hot", "index"] = "one-hot",
        zero_division: Union[float, Literal["warn", "nan"]] = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        _dice_score_validate_args(num_classes, include_background, average, input_format, zero_division)
        self.num_classes = num_classes
        self.include_background = include_background
        self.average = average
        self.input_format = input_format
        self.zero_division = zero_division

        num_classes = num_classes - 1 if not include_background else num_classes
        self.add_state("numerator", [], dist_reduce_fx="cat")
        self.add_state("denominator", [], dist_reduce_fx="cat")
        self.add_state("support", [], dist_reduce_fx="cat")

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update the state with new data."""
        numerator, denominator, support = _dice_score_update(
            preds, target, self.num_classes, self.include_background, self.input_format
        )
        self.numerator.append(numerator)
        self.denominator.append(denominator)
        self.support.append(support)

    def compute(self) -> Tensor:
        """Computes the Dice Score."""
        return _dice_score_compute(
            dim_zero_cat(self.numerator),
            dim_zero_cat(self.denominator),
            self.average,
            support=dim_zero_cat(self.support) if self.average == "weighted" else None,
            zero_division=self.zero_division,
        ).nanmean(dim=0)

    def plot(self, val: Union[Tensor, Sequence[Tensor], None] = None, ax: Optional[_AX_TYPE] = None) -> _PLOT_OUT_TYPE:
        """Plot a single or multiple values from the metric.

        Args:
            val: Either a single result from calling `metric.forward` or `metric.compute` or a list of these results.
                If no value is provided, will automatically call `metric.compute` and plot that result.
            ax: An matplotlib axis object. If provided will add plot to that axis

        Returns:
            Figure and Axes object

        Raises:
            ModuleNotFoundError:
                If `matplotlib` is not installed

        .. plot::
            :scale: 75

            >>> # Example plotting a single value
            >>> import torch
            >>> from torchmetrics.segmentation import DiceScore
            >>> metric = DiceScore(num_classes=3)
            >>> metric.update(torch.randint(0, 2, (10, 3, 128, 128)), torch.randint(0, 2, (10, 3, 128, 128)))
            >>> fig_, ax_ = metric.plot()

        .. plot::
            :scale: 75

            >>> # Example plotting multiple values
            >>> import torch
            >>> from torchmetrics.segmentation import DiceScore
            >>> metric = DiceScore(num_classes=3)
            >>> values = [ ]
            >>> for _ in range(10):
            ...     values.append(
            ...        metric(torch.randint(0, 2, (10, 3, 128, 128)), torch.randint(0, 2, (10, 3, 128, 128)))
            ...     )
            >>> fig_, ax_ = metric.plot(values)

        """
        return self._plot(val, ax)
