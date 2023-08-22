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
from typing import Any, List, Optional, Sequence, Union

import torch
from torch import Tensor

from torchmetrics.functional.clustering.mutual_info_score import mutual_info_score
from torchmetrics.metric import Metric
from torchmetrics.utilities.data import dim_zero_cat
from torchmetrics.utilities.imports import _MATPLOTLIB_AVAILABLE
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE

if not _MATPLOTLIB_AVAILABLE:
    __doctest_skip__ = ["MutualInfoScore.plot"]


class MutualInfoScore(Metric):
    r"""Compute `Mutual Information Score`_.

    .. math::
        MI(U,V) = \sum_{i=1}^{\abs{U}} \sum_{j=1}^{\abs{V}} \frac{\abs{U_i\cap V_j}}{N} \log\frac{N\abs{U_i\cap V_j}}{\abs{U_i}\abs{V_j}}

    Where :math:`U` is a tensor of target values, :math:`V` is a tensor of predictions,
    :math:`\abs{U_i}` is the number of samples in cluster :math:`U_i`, and
    :math:`\abs{V_i}` is the number of samples in cluster :math:`V_i`.

    The metric is symmetric, therefore swapping :math:`U` and :math:`V` yields
    the same mutual information score.

    Args:
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (:class:`~torch.Tensor`): either single output float tensor with shape ``(N,)``
    - ``target`` (:class:`~torch.Tensor`): either single output tensor with shape ``(N,)``

    As output of ``forward`` and ``compute`` the metric returns the following output:

    - ``mi_score`` (:class:`~torch.Tensor`): A tensor with the Mutual Information Score

    Example:
        >>> from torchmetrics.clustering import MutualInfoScore
        >>> target = torch.tensor([])
        >>> preds = torch.tensor([])
        >>> mi_score = MutualInfoScore()
        >>> mi_score(preds, target)
        tensor()

    """

    is_differentiable = True
    higher_is_better = None
    full_state_update: bool = True
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0  # theoretical upper bound is +inf
    preds: List[Tensor]
    target: List[Tensor]
    contingency: Tensor

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("target", default=[], dist_reduce_fx="cat")

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update state with predictions and targets."""
        self.preds.append(preds)
        self.target.append(target)

    def compute(self) -> Tensor:
        """Compute mutual information over state."""
        return mutual_info_score(dim_zero_cat(self.preds), dim_zero_cat(self.target))

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
            >>> from torchmetrics.clustering import MutualInfoScore
            >>> metric = MutualInfoScore(num_classes=5)
            >>> metric.update(torch.randint(0, 4, (100,)), torch.randint(0, 4, (100,)))
            >>> fig_, ax_ = metric.plot()

        .. plot::
            :scale: 75

            >>> # Example plotting multiple values
            >>> import torch
            >>> from torchmetrics.clustering import MutualInfoScore
            >>> metric = MutualInfoScore(num_classes=5)
            >>> values = [ ]
            >>> for _ in range(10):
            ...     values.append(metric(torch.randint(0, 4, (100,)), torch.randint(0, 4, (100,))))
            >>> fig_, ax_ = metric.plot(values)

        """
        return self._plot(val, ax)
