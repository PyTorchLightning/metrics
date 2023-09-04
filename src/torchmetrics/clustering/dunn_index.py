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

from torch import Tensor

from torchmetrics.functional.clustering.dunn_index import dunn_index
from torchmetrics.metric import Metric
from torchmetrics.utilities.data import dim_zero_cat
from torchmetrics.utilities.imports import _MATPLOTLIB_AVAILABLE
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE

if not _MATPLOTLIB_AVAILABLE:
    __doctest_skip__ = ["DunnIndex.plot"]


class DunnIndex(Metric):
    r"""Compute `Dunn Index`_.

    .. math::
        DI_m = \frac{\min_{1\leq i<j\leq m} \delta(C_i,C_j)}{\max_{1\leq k\leq m} \Delta_k}

    Where :math:`C_i` is a cluster of tensors, :math:`C_j` is a cluster of tensors,
    and :math:`\delta(C_i,C_j)` is the intercluster distance metric for :math:`m` clusters.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``data`` (:class:`~torch.Tensor`): float tensor with shape ``(N,d)`` with the embedded data. ``d`` is the dimensionality of the embedding space.
    - ``labels`` (:class:`~torch.Tensor`): single integer tensor with shape ``(N,)`` with cluster labels

    As output of ``forward`` and ``compute`` the metric returns the following output:

    - ``dunn_index`` (:class:`~torch.Tensor`): A tensor with the Dunn Index

    Args:
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Example:
        >>> import torch
        >>> from torchmetrics.clustering import DunnIndex
        >>> preds = torch.tensor([2, 1, 0, 1, 0])
        >>> target = torch.tensor([0, 2, 1, 1, 0])
        >>> dun_index = DunnIndex()
        >>> dunn_index(preds, target)
        tensor(0.5004)

    """

    is_differentiable: bool = True
    higher_is_better: bool = True
    full_state_update: bool = True
    plot_lower_bound: float = 0.0
    x: List[Tensor]
    labels: List[Tensor]
    contingency: Tensor

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        self.add_state("data", default=[], dist_reduce_fx="cat")
        self.add_state("labels", default=[], dist_reduce_fx="cat")

    def update(self, data: Tensor, labels: Tensor) -> None:
        """Update state with predictions and targets."""
        self.data.append(data)
        self.labels.append(labels)

    def compute(self) -> Tensor:
        """Compute mutual information over state."""
        return dunn_index(dim_zero_cat(self.data), dim_zero_cat(self.labels))

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
            >>> from torchmetrics.clustering import DunnIndex
            >>> metric = DunnIndex()
            >>> metric.update(torch.randint(0, 4, (10,)), torch.randint(0, 4, (10,)))
            >>> fig_, ax_ = metric.plot(metric.compute())

        .. plot::
            :scale: 75

            >>> # Example plotting multiple values
            >>> import torch
            >>> from torchmetrics.clustering import DunnIndex
            >>> metric = DunnIndex()
            >>> for _ in range(10):
            ...     metric.update(torch.randint(0, 4, (10,)), torch.randint(0, 4, (10,)))
            >>> fig_, ax_ = metric.plot(metric.compute())

        """
        return self._plot(val, ax)
