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

from torch import Tensor, tensor

from torchmetrics.functional.audio.snr import scale_invariant_signal_noise_ratio, signal_noise_ratio
from torchmetrics.metric import Metric
from torchmetrics.utilities.imports import _MATPLOTLIB_AVAILABLE
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE, plot_single_or_multi_val

if not _MATPLOTLIB_AVAILABLE:
    __doctest_skip__ = ["SignalNoiseRatio.plot", "ScaleInvariantSignalNoiseRatio.plot"]


class SignalNoiseRatio(Metric):
    r"""Calculate `Signal-to-noise ratio`_ (SNR_) meric for evaluating quality of audio.

    .. math::
        \text{SNR} = \frac{P_{signal}}{P_{noise}}

    where  :math:`P` denotes the power of each signal. The SNR metric compares the level of the desired signal to
    the level of background noise. Therefore, a high value of SNR means that the audio is clear.

    As input to `forward` and `update` the metric accepts the following input

    - ``preds`` (:class:`~torch.Tensor`): float tensor with shape ``(...,time)``
    - ``target`` (:class:`~torch.Tensor`): float tensor with shape ``(...,time)``

    As output of `forward` and `compute` the metric returns the following output

    - ``snr`` (:class:`~torch.Tensor`): float scalar tensor with average SNR value over samples

    Args:
        zero_mean: if to zero mean target and preds or not
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Raises:
        TypeError:
            if target and preds have a different shape

    Example:
        >>> from torch import tensor
        >>> from torchmetrics import SignalNoiseRatio
        >>> target = tensor([3.0, -0.5, 2.0, 7.0])
        >>> preds = tensor([2.5, 0.0, 2.0, 8.0])
        >>> snr = SignalNoiseRatio()
        >>> snr(preds, target)
        tensor(16.1805)
    """
    full_state_update: bool = False
    is_differentiable: bool = True
    higher_is_better: bool = True
    sum_snr: Tensor
    total: Tensor
    _plot_lower_bound = -20.0
    _plot_upper_bound = 5.0

    def __init__(
        self,
        zero_mean: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.zero_mean = zero_mean

        self.add_state("sum_snr", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update state with predictions and targets."""
        snr_batch = signal_noise_ratio(preds=preds, target=target, zero_mean=self.zero_mean)

        self.sum_snr += snr_batch.sum()
        self.total += snr_batch.numel()

    def compute(self) -> Tensor:
        """Compute metric."""
        return self.sum_snr / self.total

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

            >>> # Example plotting a single value
            >>> import torch
            >>> from torchmetrics.audio.snr import SignalNoiseRatio
            >>> metric = SignalNoiseRatio()
            >>> metric.update(torch.rand(4), torch.rand(4))
            >>> fig_, ax_ = metric.plot()

        .. plot::
            :scale: 75

            >>> # Example plotting multiple values
            >>> import torch
            >>> from torchmetrics.audio.snr import SignalNoiseRatio
            >>> metric = SignalNoiseRatio()
            >>> values = [ ]
            >>> for _ in range(10):
            ...     values.append(metric(torch.rand(4), torch.rand(4)))
            >>> fig_, ax_ = metric.plot(values)
        """
        return super()._plot(val, ax)


class ScaleInvariantSignalNoiseRatio(Metric):
    """Calculate `Scale-invariant signal-to-noise ratio`_ (SI-SNR) metric for evaluating quality of audio.

    As input to `forward` and `update` the metric accepts the following input

    - ``preds`` (:class:`~torch.Tensor`): float tensor with shape ``(...,time)``
    - ``target`` (: :class:`~torch.Tensor`): float tensor with shape ``(...,time)``

    As output of `forward` and `compute` the metric returns the following output

    - ``si_snr`` (: :class:`~torch.Tensor`): float scalar tensor with average SI-SNR value over samples

    Args:
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Raises:
        TypeError:
            if target and preds have a different shape

    Example:
        >>> from torch import tensor
        >>> from torchmetrics import ScaleInvariantSignalNoiseRatio
        >>> target = tensor([3.0, -0.5, 2.0, 7.0])
        >>> preds = tensor([2.5, 0.0, 2.0, 8.0])
        >>> si_snr = ScaleInvariantSignalNoiseRatio()
        >>> si_snr(preds, target)
        tensor(15.0918)
    """

    is_differentiable = True
    sum_si_snr: Tensor
    total: Tensor
    higher_is_better = True
    _plot_lower_bound = -20.0
    _plot_upper_bound = 10.0

    def __init__(
        self,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self.add_state("sum_si_snr", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update state with predictions and targets."""
        si_snr_batch = scale_invariant_signal_noise_ratio(preds=preds, target=target)

        self.sum_si_snr += si_snr_batch.sum()
        self.total += si_snr_batch.numel()

    def compute(self) -> Tensor:
        """Compute metric."""
        return self.sum_si_snr / self.total

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
            >>> from torchmetrics.audio.snr import ScaleInvariantSignalNoiseRatio
            >>> metric = ScaleInvariantSignalNoiseRatio()
            >>> metric.update(torch.rand(4), torch.rand(4))
            >>> fig_, ax_ = metric.plot()

        .. plot::
            :scale: 75

            >>> # Example plotting multiple values
            >>> import torch
            >>> from torchmetrics.audio.snr import ScaleInvariantSignalNoiseRatio
            >>> metric = ScaleInvariantSignalNoiseRatio()
            >>> values = [ ]
            >>> for _ in range(10):
            ...     values.append(metric(torch.rand(4), torch.rand(4)))
            >>> fig_, ax_ = metric.plot(values)
        """
        return super()._plot(val, ax)
