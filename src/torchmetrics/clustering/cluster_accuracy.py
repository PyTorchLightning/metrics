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
from typing import Any, Optional, Sequence, Union

import torch
from torch import Tensor

from torchmetrics.functional.classification import multiclass_confusion_matrix
from torchmetrics.functional.clustering.cluster_accuracy import _cluster_accuracy_compute
from torchmetrics.metric import Metric
from torchmetrics.utilities.imports import (
    _MATPLOTLIB_AVAILABLE,
    _TORCH_LINEAR_ASSIGNMENT_AVAILABLE,
)
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE

if not _MATPLOTLIB_AVAILABLE:
    __doctest_skip__ = ["ClusterAccuracy.plot"]

if not _TORCH_LINEAR_ASSIGNMENT_AVAILABLE:
    __doctest_skip__ = ["ClusterAccuracy", "ClusterAccuracy.plot"]


class ClusterAccuracy(Metric):
    r"""Compute `Cluster Accuracy`_ between predicted and target clusters.

    .. math::

        \text{Cluster Accuracy} = \max_g \frac{1}{N} \sum_{n=1}^N \mathbb{1}_{g(p_n) = t_n}

    Where :math:`g` is a function that maps predicted clusters :math:`p` to target clusters :math:`t`, :math:`N` is the
    number of samples, :math:`p_n` is the predicted cluster for sample :math:`n`, :math:`t_n` is the target cluster for
    sample :math:`n`, and :math:`\mathbb{1}` is the indicator function. The function :math:`g` is determined by solving
    the linear sum assignment problem.

    This clustering metric is an extrinsic measure, because it requires ground truth clustering labels, which may not
    be available in practice since clustering in generally is used for unsupervised learning.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (:class:`~torch.Tensor`): single integer tensor with shape ``(N,)`` with predicted cluster labels
    - ``target`` (:class:`~torch.Tensor`): single integer tensor with shape ``(N,)`` with ground truth cluster labels

    As output of ``forward`` and ``compute`` the metric returns the following output:

    - ``acc_score`` (:class:`~torch.Tensor`): A tensor with the Cluster Accuracy score

    Args:
        num_classes: number of classes
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Raises:
        RuntimeError:
            If ``torch_linear_assignment`` is not installed. To install, run ``pip install torchmetrics[clustering]``.
        ValueError
            If ``num_classes`` is not a positive integer

    Example::
        >>> import torch
        >>> from torchmetrics.clustering import ClusterAccuracy
        >>> preds = torch.tensor([0, 0, 1, 1])
        >>> target = torch.tensor([1, 1, 0, 0])
        >>> metric = ClusterAccuracy(num_classes=2)
        >>> metric(preds, target)
        tensor(1.)

    """

    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = False
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0
    confmat: Tensor

    def __init__(self, num_classes: int, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if not _TORCH_LINEAR_ASSIGNMENT_AVAILABLE:
            raise RuntimeError(
                "Missing `torch_linear_assignment`. Please install it with `pip install torchmetrics[clustering]`."
            )

        if not isinstance(num_classes, int) or num_classes <= 0:
            raise ValueError("Argument `num_classes` should be a positive integer")
        self.add_state(
            "confmat", default=torch.zeros((num_classes, num_classes), dtype=torch.int64), dist_reduce_fx="sum"
        )
        self.num_classes = num_classes

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update the confusion matrix with the new predictions and targets."""
        self.confmat += multiclass_confusion_matrix(preds, target, num_classes=self.num_classes)

    def compute(self) -> Tensor:
        """Computes the clustering accuracy."""
        return _cluster_accuracy_compute(self.confmat)

    def plot(self, val: Union[Tensor, Sequence[Tensor], None] = None, ax: Optional[_AX_TYPE] = None) -> _PLOT_OUT_TYPE:
        """Plot a single or multiple values from the metric.

        Args:
            val: Either a single result from calling ``metric.forward`` or ``metric.compute``
                or a list of these results. If no value is provided, will automatically call `metric.compute`
                and plot that result.
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
            >>> from torchmetrics.clustering import ClusterAccuracy
            >>> metric = ClusterAccuracy(num_classes=4)
            >>> metric.update(torch.randint(0, 4, (10,)), torch.randint(0, 4, (10,)))
            >>> fig_, ax_ = metric.plot(metric.compute())

        .. plot::
            :scale: 75

            >>> # Example plotting multiple values
            >>> import torch
            >>> from torchmetrics.clustering import ClusterAccuracy
            >>> metric = ClusterAccuracy(num_classes=4)
            >>> values = [ ]
            >>> for _ in range(10):
            ...     values.append(metric(torch.randint(0, 4, (10,)), torch.randint(0, 4, (10,))))
            >>> fig_, ax_ = metric.plot(values)

        """
        return self._plot(val, ax)
