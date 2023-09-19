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
from typing import Any, List, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor

from torchmetrics.functional.regression.pearson import _pearson_corrcoef_compute, _pearson_corrcoef_update
from torchmetrics.metric import Metric
from torchmetrics.utilities.imports import _MATPLOTLIB_AVAILABLE
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE

if not _MATPLOTLIB_AVAILABLE:
    __doctest_skip__ = ["PearsonCorrCoef.plot"]


def _final_aggregation(
    means_x: Tensor,
    means_y: Tensor,
    vars_x: Tensor,
    vars_y: Tensor,
    corrs_xy: Tensor,
    nbs: Tensor,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Aggregate the statistics from multiple devices.

    Formula taken from here: `Aggregate the statistics from multiple devices`_

    """
    if len(means_x) == 1:
        return means_x[0], means_y[0], vars_x[0], vars_y[0], corrs_xy[0], nbs[0]
    mx1, my1, vx1, vy1, cxy1, n1 = means_x[0], means_y[0], vars_x[0], vars_y[0], corrs_xy[0], nbs[0]
    for i in range(1, len(means_x)):
        mx2, my2, vx2, vy2, cxy2, n2 = means_x[i], means_y[i], vars_x[i], vars_y[i], corrs_xy[i], nbs[i]
        nb = n1 + n2
        mean_x = (n1 * mx1 + n2 * mx2) / nb
        mean_y = (n1 * my1 + n2 * my2) / nb

        # var_x
        element_x1 = (n1 + 1) * mean_x - n1 * mx1
        vx1 += (element_x1 - mx1) * (element_x1 - mean_x) - (element_x1 - mean_x) ** 2
        element_x2 = (n2 + 1) * mean_x - n2 * mx2
        vx2 += (element_x2 - mx2) * (element_x2 - mean_x) - (element_x2 - mean_x) ** 2
        var_x = vx1 + vx2

        # var_y
        element_y1 = (n1 + 1) * mean_y - n1 * my1
        vy1 += (element_y1 - my1) * (element_y1 - mean_y) - (element_y1 - mean_y) ** 2
        element_y2 = (n2 + 1) * mean_y - n2 * my2
        vy2 += (element_y2 - my2) * (element_y2 - mean_y) - (element_y2 - mean_y) ** 2
        var_y = vy1 + vy2

        # corr
        cxy1 += (element_x1 - mx1) * (element_y1 - mean_y) - (element_x1 - mean_x) * (element_y1 - mean_y)
        cxy2 += (element_x2 - mx2) * (element_y2 - mean_y) - (element_x2 - mean_x) * (element_y2 - mean_y)
        corr_xy = cxy1 + cxy2

        mx1, my1, vx1, vy1, cxy1, n1 = mean_x, mean_y, var_x, var_y, corr_xy, nb
    return mean_x, mean_y, var_x, var_y, corr_xy, nb


class PearsonCorrCoef(Metric):
    r"""Compute `Pearson Correlation Coefficient`_.

    .. math::
        P_{corr}(x,y) = \frac{cov(x,y)}{\sigma_x \sigma_y}

    Where :math:`y` is a tensor of target values, and :math:`x` is a tensor of predictions.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (:class:`~torch.Tensor`): either single output float tensor with shape ``(N,)``
      or multioutput float tensor of shape ``(N,d)``
    - ``target`` (:class:`~torch.Tensor`): either single output tensor with shape ``(N,)``
      or multioutput tensor of shape ``(N,d)``

    As output of ``forward`` and ``compute`` the metric returns the following output:

    - ``pearson`` (:class:`~torch.Tensor`): A tensor with the Pearson Correlation Coefficient

    Args:
        num_outputs: Number of outputs in multioutput setting
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Example (single output regression):
        >>> from torchmetrics.regression import PearsonCorrCoef
        >>> target = torch.tensor([3, -0.5, 2, 7])
        >>> preds = torch.tensor([2.5, 0.0, 2, 8])
        >>> pearson = PearsonCorrCoef()
        >>> pearson(preds, target)
        tensor(0.9849)

    Example (multi output regression):
        >>> from torchmetrics.regression import PearsonCorrCoef
        >>> target = torch.tensor([[3, -0.5], [2, 7]])
        >>> preds = torch.tensor([[2.5, 0.0], [2, 8]])
        >>> pearson = PearsonCorrCoef(num_outputs=2)
        >>> pearson(preds, target)
        tensor([1., 1.])

    """
    is_differentiable: bool = True
    higher_is_better: Optional[bool] = None  # both -1 and 1 are optimal
    full_state_update: bool = True
    plot_lower_bound: float = -1.0
    plot_upper_bound: float = 1.0
    preds: List[Tensor]
    target: List[Tensor]
    mean_x: Tensor
    mean_y: Tensor
    var_x: Tensor
    var_y: Tensor
    corr_xy: Tensor
    n_total: Tensor

    def __init__(
        self,
        num_outputs: int = 1,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if not isinstance(num_outputs, int) and num_outputs < 1:
            raise ValueError("Expected argument `num_outputs` to be an int larger than 0, but got {num_outputs}")
        self.num_outputs = num_outputs

        self.add_state("mean_x", default=torch.zeros(self.num_outputs), dist_reduce_fx=None)
        self.add_state("mean_y", default=torch.zeros(self.num_outputs), dist_reduce_fx=None)
        self.add_state("var_x", default=torch.zeros(self.num_outputs), dist_reduce_fx=None)
        self.add_state("var_y", default=torch.zeros(self.num_outputs), dist_reduce_fx=None)
        self.add_state("corr_xy", default=torch.zeros(self.num_outputs), dist_reduce_fx=None)
        self.add_state("n_total", default=torch.zeros(self.num_outputs), dist_reduce_fx=None)

    def update(self, preds: Tensor, target: Tensor, weights: Optional[Tensor] = None) -> None:
        """Update state with predictions and targets."""
        self.mean_x, self.mean_y, self.var_x, self.var_y, self.corr_xy, self.n_total = _pearson_corrcoef_update(
            preds,
            target,
            self.mean_x,
            self.mean_y,
            self.var_x,
            self.var_y,
            self.corr_xy,
            self.n_total,
            self.num_outputs,
            weights=weights,
        )

    def compute(self) -> Tensor:
        """Compute pearson correlation coefficient over state."""
        if (self.num_outputs == 1 and self.mean_x.numel() > 1) or (self.num_outputs > 1 and self.mean_x.ndim > 1):
            # multiple devices, need further reduction
            _, _, var_x, var_y, corr_xy, n_total = _final_aggregation(
                self.mean_x, self.mean_y, self.var_x, self.var_y, self.corr_xy, self.n_total
            )
        else:
            var_x = self.var_x
            var_y = self.var_y
            corr_xy = self.corr_xy
            n_total = self.n_total
        return _pearson_corrcoef_compute(var_x, var_y, corr_xy, n_total)

    def plot(
        self, val: Optional[Union[Tensor, Sequence[Tensor]]] = None, ax: Optional[_AX_TYPE] = None
    ) -> _PLOT_OUT_TYPE:
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

            >>> from torch import randn
            >>> # Example plotting a single value
            >>> from torchmetrics.regression import PearsonCorrCoef
            >>> metric = PearsonCorrCoef()
            >>> metric.update(randn(10,), randn(10,))
            >>> fig_, ax_ = metric.plot()

        .. plot::
            :scale: 75

            >>> from torch import randn
            >>> # Example plotting multiple values
            >>> from torchmetrics.regression import PearsonCorrCoef
            >>> metric = PearsonCorrCoef()
            >>> values = []
            >>> for _ in range(10):
            ...     values.append(metric(randn(10,), randn(10,)))
            >>> fig, ax = metric.plot(values)

        """
        return self._plot(val, ax)
