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
from typing import Any, List, Literal, Optional, Union

from torch import Tensor

from torchmetrics.clustering.mutual_info_score import MutualInfoScore
from torchmetrics.functional.clustering.adjusted_mutual_info_score import (
    _validate_average_method_arg,
    adjusted_mutual_info_score,
)
from torchmetrics.utilities.data import dim_zero_cat
from torchmetrics.utilities.imports import _MATPLOTLIB_AVAILABLE
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE

if not _MATPLOTLIB_AVAILABLE:
    __doctest_skip__ = ["AdjustedMutualInfoScore.plot"]


class AdjustedMutualInfoScore(MutualInfoScore):
    r"""Compute `Adjusted Mutual Information Score`_.

    .. math::
        AMI(U,V) = \frac{MI(U,V) - E(MI(U,V))}{avg(H(U), H(V)) - E(MI(U,V))}

    Where :math:`U` is a tensor of target values, :math:`V` is a tensor of predictions, :math:`M_p(U,V)` is the
    generalized mean of order :math:`p` of :math:`U` and :math:`V`, and :math:`MI(U,V)` is the mutual information score
    between clusters :math:`U` and :math:`V`. The metric is symmetric, therefore swapping :math:`U` and :math:`V` yields
    the same mutual information score.

    This clustering metric is an extrinsic measure, because it requires ground truth clustering labels, which may not
    be available in practice since clustering in generally is used for unsupervised learning.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (:class:`~torch.Tensor`): single integer tensor with shape ``(N,)`` with predicted cluster labels
    - ``target`` (:class:`~torch.Tensor`): single integer tensor with shape ``(N,)`` with ground truth cluster labels

    As output of ``forward`` and ``compute`` the metric returns the following output:

    - ``ami_score`` (:class:`~torch.Tensor`): A tensor with the Adjusted Mutual Information Score

    Args:
        average_method: Method used to calculate generalized mean for normalization. Choose between
            ``'min'``, ``'geometric'``, ``'arithmetic'``, ``'max'``.
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Example::
        >>> import torch
        >>> from torchmetrics.clustering import AdjustedMutualInfoScore
        >>> preds = torch.tensor([2, 1, 0, 1, 0])
        >>> target = torch.tensor([0, 2, 1, 1, 0])
        >>> ami_score = AdjustedMutualInfoScore(average_method="arithmetic")
        >>> ami_score(preds, target)
        tensor(-0.2500)

    """

    is_differentiable: bool = True
    higher_is_better: Optional[bool] = None
    full_state_update: bool = False
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0
    preds: List[Tensor]
    target: List[Tensor]

    def __init__(
        self, average_method: Literal["min", "geometric", "arithmetic", "max"] = "arithmetic", **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        _validate_average_method_arg(average_method)
        self.average_method = average_method

    def compute(self) -> Tensor:
        """Compute normalized mutual information over state."""
        return adjusted_mutual_info_score(dim_zero_cat(self.preds), dim_zero_cat(self.target), self.average_method)

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
            >>> from torchmetrics.clustering import AdjustedMutualInfoScore
            >>> metric = AdjustedMutualInfoScore()
            >>> metric.update(torch.randint(0, 4, (10,)), torch.randint(0, 4, (10,)))
            >>> fig_, ax_ = metric.plot(metric.compute())

        .. plot::
            :scale: 75

            >>> # Example plotting multiple values
            >>> import torch
            >>> from torchmetrics.clustering import AdjustedMutualInfoScore
            >>> metric = AdjustedMutualInfoScore()
            >>> values = []
            >>> for _ in range(10):
            ...     values.append(metric(torch.randint(0, 4, (10,)), torch.randint(0, 4, (10,))))
            >>> fig_, ax_ = metric.plot(values)

        """
        return self._plot(val, ax)
