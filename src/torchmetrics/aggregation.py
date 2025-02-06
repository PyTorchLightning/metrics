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
from typing import Any, Callable, Optional, Union

import torch
from torch import Tensor

from torchmetrics.metric import Metric
from torchmetrics.utilities import rank_zero_warn
from torchmetrics.utilities.data import dim_zero_cat
from torchmetrics.utilities.imports import _MATPLOTLIB_AVAILABLE
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE
from torchmetrics.wrappers.running import Running

if not _MATPLOTLIB_AVAILABLE:
    __doctest_skip__ = ["SumMetric.plot", "MeanMetric.plot", "MaxMetric.plot", "MinMetric.plot"]


class BaseAggregator(Metric):
    """Base class for aggregation metrics.

    Args:
        fn: string specifying the reduction function
        default_value: default tensor value to use for the metric state
        nan_strategy: options:
            - ``'error'``: if any `nan` values are encountered will give a RuntimeError
            - ``'warn'``: if any `nan` values are encountered will give a warning and continue
            - ``'ignore'``: all `nan` values are silently removed
            - a float: if a float is provided will impute any `nan` values with this value

        state_name: name of the metric state
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Raises:
        ValueError:
            If ``nan_strategy`` is not one of ``error``, ``warn``, ``ignore``, ``disable`` or a float

    """

    is_differentiable = None
    higher_is_better = None
    full_state_update: bool = False

    def __init__(
        self,
        fn: Union[Callable, str],
        default_value: Union[Tensor, list],
        nan_strategy: Union[str, float] = "error",
        state_name: str = "value",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        allowed_nan_strategy = ("error", "warn", "ignore", "disable")
        if nan_strategy not in allowed_nan_strategy and not isinstance(nan_strategy, float):
            raise ValueError(
                f"Arg `nan_strategy` should either be a float or one of {allowed_nan_strategy} but got {nan_strategy}."
            )

        self.nan_strategy = nan_strategy
        self.add_state(state_name, default=default_value, dist_reduce_fx=fn)
        self.state_name = state_name

    def _cast_and_nan_check_input(
        self, x: Union[float, Tensor], weight: Optional[Union[float, Tensor]] = None
    ) -> tuple[Tensor, Tensor]:
        """Convert input ``x`` to a tensor and check for Nans."""
        if not isinstance(x, Tensor):
            x = torch.as_tensor(x, dtype=self.dtype, device=self.device)
        if weight is not None and not isinstance(weight, Tensor):
            weight = torch.as_tensor(weight, dtype=self.dtype, device=self.device)

        if self.nan_strategy != "disable":
            nans = torch.isnan(x)
            if weight is not None:
                nans_weight = torch.isnan(weight)
            else:
                nans_weight = torch.zeros_like(nans).bool()
                weight = torch.ones_like(x)
            if nans.any() or nans_weight.any():
                if self.nan_strategy == "error":
                    raise RuntimeError("Encountered `nan` values in tensor")
                if self.nan_strategy in ("ignore", "warn"):
                    if self.nan_strategy == "warn":
                        rank_zero_warn("Encountered `nan` values in tensor. Will be removed.", UserWarning)
                    x = x[~(nans | nans_weight)]
                    weight = weight[~(nans | nans_weight)]
                else:
                    if not isinstance(self.nan_strategy, float):
                        raise ValueError(f"`nan_strategy` shall be float but you pass {self.nan_strategy}")
                    x[nans | nans_weight] = self.nan_strategy
                    weight[nans | nans_weight] = self.nan_strategy

        return x.to(self.dtype), weight.to(self.dtype)

    def update(self, value: Union[float, Tensor]) -> None:
        """Overwrite in child class."""

    def compute(self) -> Tensor:
        """Compute the aggregated value."""
        return getattr(self, self.state_name)


class MaxMetric(BaseAggregator):
    """Aggregate a stream of value into their maximum value.

    As input to ``forward`` and ``update`` the metric accepts the following input

    - ``value`` (:class:`~float` or :class:`~torch.Tensor`): a single float or an tensor of float values with
      arbitrary shape ``(...,)``.

    As output of `forward` and `compute` the metric returns the following output

    - ``agg`` (:class:`~torch.Tensor`): scalar float tensor with aggregated maximum value over all inputs received

    Args:
        nan_strategy: options:
            - ``'error'``: if any `nan` values are encountered will give a RuntimeError
            - ``'warn'``: if any `nan` values are encountered will give a warning and continue
            - ``'ignore'``: all `nan` values are silently removed
            - a float: if a float is provided will impute any `nan` values with this value

        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Raises:
        ValueError:
            If ``nan_strategy`` is not one of ``error``, ``warn``, ``ignore`` or a float

    Example:
        >>> from torch import tensor
        >>> from torchmetrics.aggregation import MaxMetric
        >>> metric = MaxMetric()
        >>> metric.update(1)
        >>> metric.update(tensor([2, 3]))
        >>> metric.compute()
        tensor(3.)

    """

    full_state_update: bool = True
    max_value: Tensor

    def __init__(
        self,
        nan_strategy: Union[str, float] = "warn",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            "max",
            -torch.tensor(float("inf"), dtype=torch.get_default_dtype()),
            nan_strategy,
            state_name="max_value",
            **kwargs,
        )

    def update(self, value: Union[float, Tensor]) -> None:
        """Update state with data.

        Args:
            value: Either a float or tensor containing data. Additional tensor
                dimensions will be flattened

        """
        value, _ = self._cast_and_nan_check_input(value)
        if value.numel():  # make sure tensor not empty
            self.max_value = torch.max(self.max_value, torch.max(value))

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
            >>> from torchmetrics.aggregation import MaxMetric
            >>> metric = MaxMetric()
            >>> metric.update([1, 2, 3])
            >>> fig_, ax_ = metric.plot()

        .. plot::
            :scale: 75

            >>> # Example plotting multiple values
            >>> from torchmetrics.aggregation import MaxMetric
            >>> metric = MaxMetric()
            >>> values = [ ]
            >>> for i in range(10):
            ...     values.append(metric(i))
            >>> fig_, ax_ = metric.plot(values)

        """
        return self._plot(val, ax)


class MinMetric(BaseAggregator):
    """Aggregate a stream of value into their minimum value.

    As input to ``forward`` and ``update`` the metric accepts the following input

    - ``value`` (:class:`~float` or :class:`~torch.Tensor`): a single float or an tensor of float values with
      arbitrary shape ``(...,)``.

    As output of `forward` and `compute` the metric returns the following output

    - ``agg`` (:class:`~torch.Tensor`): scalar float tensor with aggregated minimum value over all inputs received

    Args:
        nan_strategy: options:
            - ``'error'``: if any `nan` values are encountered will give a RuntimeError
            - ``'warn'``: if any `nan` values are encountered will give a warning and continue
            - ``'ignore'``: all `nan` values are silently removed
            - a float: if a float is provided will impute any `nan` values with this value

        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Raises:
        ValueError:
            If ``nan_strategy`` is not one of ``error``, ``warn``, ``ignore`` or a float

    Example:
        >>> from torch import tensor
        >>> from torchmetrics.aggregation import MinMetric
        >>> metric = MinMetric()
        >>> metric.update(1)
        >>> metric.update(tensor([2, 3]))
        >>> metric.compute()
        tensor(1.)

    """

    full_state_update: bool = True
    min_value: Tensor

    def __init__(
        self,
        nan_strategy: Union[str, float] = "warn",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            "min",
            torch.tensor(float("inf"), dtype=torch.get_default_dtype()),
            nan_strategy,
            state_name="min_value",
            **kwargs,
        )

    def update(self, value: Union[float, Tensor]) -> None:
        """Update state with data.

        Args:
            value: Either a float or tensor containing data. Additional tensor
                dimensions will be flattened

        """
        value, _ = self._cast_and_nan_check_input(value)
        if value.numel():  # make sure tensor not empty
            self.min_value = torch.min(self.min_value, torch.min(value))

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
            >>> from torchmetrics.aggregation import MinMetric
            >>> metric = MinMetric()
            >>> metric.update([1, 2, 3])
            >>> fig_, ax_ = metric.plot()

        .. plot::
            :scale: 75

            >>> # Example plotting multiple values
            >>> from torchmetrics.aggregation import MinMetric
            >>> metric = MinMetric()
            >>> values = [ ]
            >>> for i in range(10):
            ...     values.append(metric(i))
            >>> fig_, ax_ = metric.plot(values)

        """
        return self._plot(val, ax)


class SumMetric(BaseAggregator):
    """Aggregate a stream of value into their sum.

    As input to ``forward`` and ``update`` the metric accepts the following input

    - ``value`` (:class:`~float` or :class:`~torch.Tensor`): a single float or an tensor of float values with
      arbitrary shape ``(...,)``.

    As output of `forward` and `compute` the metric returns the following output

    - ``agg`` (:class:`~torch.Tensor`): scalar float tensor with aggregated sum over all inputs received

    Args:
        nan_strategy: options:
            - ``'error'``: if any `nan` values are encountered will give a RuntimeError
            - ``'warn'``: if any `nan` values are encountered will give a warning and continue
            - ``'ignore'``: all `nan` values are silently removed
            - a float: if a float is provided will impute any `nan` values with this value

        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Raises:
        ValueError:
            If ``nan_strategy`` is not one of ``error``, ``warn``, ``ignore`` or a float

    Example:
        >>> from torch import tensor
        >>> from torchmetrics.aggregation import SumMetric
        >>> metric = SumMetric()
        >>> metric.update(1)
        >>> metric.update(tensor([2, 3]))
        >>> metric.compute()
        tensor(6.)

    """

    sum_value: Tensor

    def __init__(
        self,
        nan_strategy: Union[str, float] = "warn",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            "sum",
            torch.tensor(0.0, dtype=torch.get_default_dtype()),
            nan_strategy,
            state_name="sum_value",
            **kwargs,
        )

    def update(self, value: Union[float, Tensor]) -> None:
        """Update state with data.

        Args:
            value: Either a float or tensor containing data. Additional tensor
                dimensions will be flattened

        """
        value, _ = self._cast_and_nan_check_input(value)
        if value.numel():
            self.sum_value += value.sum()

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
            >>> from torchmetrics.aggregation import SumMetric
            >>> metric = SumMetric()
            >>> metric.update([1, 2, 3])
            >>> fig_, ax_ = metric.plot()

        .. plot::
            :scale: 75

            >>> # Example plotting multiple values
            >>> from torch import rand, randint
            >>> from torchmetrics.aggregation import SumMetric
            >>> metric = SumMetric()
            >>> values = [ ]
            >>> for i in range(10):
            ...     values.append(metric([i, i+1]))
            >>> fig_, ax_ = metric.plot(values)

        """
        return self._plot(val, ax)


class CatMetric(BaseAggregator):
    """Concatenate a stream of values.

    As input to ``forward`` and ``update`` the metric accepts the following input

    - ``value`` (:class:`~float` or :class:`~torch.Tensor`): a single float or an tensor of float values with
      arbitrary shape ``(...,)``.

    As output of `forward` and `compute` the metric returns the following output

    - ``agg`` (:class:`~torch.Tensor`): scalar float tensor with concatenated values over all input received

    Args:
        nan_strategy: options:
            - ``'error'``: if any `nan` values are encountered will give a RuntimeError
            - ``'warn'``: if any `nan` values are encountered will give a warning and continue
            - ``'ignore'``: all `nan` values are silently removed
            - a float: if a float is provided will impute any `nan` values with this value

        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Raises:
        ValueError:
            If ``nan_strategy`` is not one of ``error``, ``warn``, ``ignore`` or a float

    Example:
        >>> from torch import tensor
        >>> from torchmetrics.aggregation import CatMetric
        >>> metric = CatMetric()
        >>> metric.update(1)
        >>> metric.update(tensor([2, 3]))
        >>> metric.compute()
        tensor([1., 2., 3.])

    """

    value: Tensor

    def __init__(
        self,
        nan_strategy: Union[str, float] = "warn",
        **kwargs: Any,
    ) -> None:
        super().__init__("cat", [], nan_strategy, **kwargs)

    def update(self, value: Union[float, Tensor]) -> None:
        """Update state with data.

        Args:
            value: Either a float or tensor containing data. Additional tensor
                dimensions will be flattened

        """
        value, _ = self._cast_and_nan_check_input(value)
        if value.numel():
            self.value.append(value)

    def compute(self) -> Tensor:
        """Compute the aggregated value."""
        if isinstance(self.value, list) and self.value:
            return dim_zero_cat(self.value)
        return self.value


class MeanMetric(BaseAggregator):
    """Aggregate a stream of value into their mean value.

    As input to ``forward`` and ``update`` the metric accepts the following input

    - ``value`` (:class:`~float` or :class:`~torch.Tensor`): a single float or an tensor of float values with
      arbitrary shape ``(...,)``.
    - ``weight`` (:class:`~float` or :class:`~torch.Tensor`): a single float or an tensor of float value with
      arbitrary shape ``(...,)``. Needs to be broadcastable with the shape of ``value`` tensor.

    As output of `forward` and `compute` the metric returns the following output

    - ``agg`` (:class:`~torch.Tensor`): scalar float tensor with aggregated (weighted) mean over all inputs received

    Args:
       nan_strategy: options:
            - ``'error'``: if any `nan` values are encountered will give a RuntimeError
            - ``'warn'``: if any `nan` values are encountered will give a warning and continue
            - ``'ignore'``: all `nan` values are silently removed
            - a float: if a float is provided will impute any `nan` values with this value

        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Raises:
        ValueError:
            If ``nan_strategy`` is not one of ``error``, ``warn``, ``ignore`` or a float

    Example:
        >>> from torchmetrics.aggregation import MeanMetric
        >>> metric = MeanMetric()
        >>> metric.update(1)
        >>> metric.update(torch.tensor([2, 3]))
        >>> metric.compute()
        tensor(2.)

    """

    mean_value: Tensor

    def __init__(
        self,
        nan_strategy: Union[str, float] = "warn",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            "sum",
            torch.tensor(0.0, dtype=torch.get_default_dtype()),
            nan_strategy,
            state_name="mean_value",
            **kwargs,
        )
        self.add_state("weight", default=torch.tensor(0.0, dtype=torch.get_default_dtype()), dist_reduce_fx="sum")

    def update(self, value: Union[float, Tensor], weight: Union[float, Tensor] = None) -> None:
        """Update state with data.

        Args:
            value: Either a float or tensor containing data. Additional tensor
                dimensions will be flattened
            weight: Either a float or tensor containing weights for calculating
                the average. Shape of weight should be able to broadcast with
                the shape of `value`. Default to None corresponding to simple
                harmonic average.

        """
        # broadcast weight to value shape
        if not isinstance(value, Tensor):
            value = torch.as_tensor(value, dtype=self.dtype, device=self.device)
        if weight is None:
            weight = torch.ones_like(value)
        elif not isinstance(weight, Tensor):
            weight = torch.as_tensor(weight, dtype=self.dtype, device=self.device)
        weight = torch.broadcast_to(weight, value.shape)
        value, weight = self._cast_and_nan_check_input(value, weight)

        if value.numel() == 0:
            return
        self.mean_value += (value * weight).sum()
        self.weight += weight.sum()

    def compute(self) -> Tensor:
        """Compute the aggregated value."""
        return self.mean_value / self.weight

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
            >>> from torchmetrics.aggregation import MeanMetric
            >>> metric = MeanMetric()
            >>> metric.update([1, 2, 3])
            >>> fig_, ax_ = metric.plot()

        .. plot::
            :scale: 75

            >>> # Example plotting multiple values
            >>> from torchmetrics.aggregation import MeanMetric
            >>> metric = MeanMetric()
            >>> values = [ ]
            >>> for i in range(10):
            ...     values.append(metric([i, i+1]))
            >>> fig_, ax_ = metric.plot(values)

        """
        return self._plot(val, ax)


class RunningMean(Running):
    """Aggregate a stream of value into their mean over a running window.

    Using this metric compared to `MeanMetric` allows for calculating metrics over a running window of values, instead
    of the whole history of values. This is beneficial when you want to get a better estimate of the metric during
    training and don't want to wait for the whole training to finish to get epoch level estimates.

    As input to ``forward`` and ``update`` the metric accepts the following input

    - ``value`` (:class:`~float` or :class:`~torch.Tensor`): a single float or an tensor of float values with
      arbitrary shape ``(...,)``.

    As output of `forward` and `compute` the metric returns the following output

    - ``agg`` (:class:`~torch.Tensor`): scalar float tensor with aggregated sum over all inputs received

    Args:
        window: The size of the running window.
        nan_strategy: options:
            - ``'error'``: if any `nan` values are encountered will give a RuntimeError
            - ``'warn'``: if any `nan` values are encountered will give a warning and continue
            - ``'ignore'``: all `nan` values are silently removed
            - a float: if a float is provided will impute any `nan` values with this value

        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Raises:
        ValueError:
            If ``nan_strategy`` is not one of ``error``, ``warn``, ``ignore`` or a float

    Example:
        >>> from torch import tensor
        >>> from torchmetrics.aggregation import RunningMean
        >>> metric = RunningMean(window=3)
        >>> for i in range(6):
        ...     current_val = metric(tensor([i]))
        ...     running_val = metric.compute()
        ...     total_val = tensor(sum(list(range(i+1)))) / (i+1)  # total mean over all samples
        ...     print(f"{current_val=}, {running_val=}, {total_val=}")
        current_val=tensor(0.), running_val=tensor(0.), total_val=tensor(0.)
        current_val=tensor(1.), running_val=tensor(0.5000), total_val=tensor(0.5000)
        current_val=tensor(2.), running_val=tensor(1.), total_val=tensor(1.)
        current_val=tensor(3.), running_val=tensor(2.), total_val=tensor(1.5000)
        current_val=tensor(4.), running_val=tensor(3.), total_val=tensor(2.)
        current_val=tensor(5.), running_val=tensor(4.), total_val=tensor(2.5000)

    """

    def __init__(
        self,
        window: int = 5,
        nan_strategy: Union[str, float] = "warn",
        **kwargs: Any,
    ) -> None:
        super().__init__(base_metric=MeanMetric(nan_strategy=nan_strategy, **kwargs), window=window)


class RunningSum(Running):
    """Aggregate a stream of value into their sum over a running window.

    Using this metric compared to `SumMetric` allows for calculating metrics over a running window of values, instead
    of the whole history of values. This is beneficial when you want to get a better estimate of the metric during
    training and don't want to wait for the whole training to finish to get epoch level estimates.

    As input to ``forward`` and ``update`` the metric accepts the following input

    - ``value`` (:class:`~float` or :class:`~torch.Tensor`): a single float or an tensor of float values with
      arbitrary shape ``(...,)``.

    As output of `forward` and `compute` the metric returns the following output

    - ``agg`` (:class:`~torch.Tensor`): scalar float tensor with aggregated sum over all inputs received

    Args:
        window: The size of the running window.
        nan_strategy: options:
            - ``'error'``: if any `nan` values are encountered will give a RuntimeError
            - ``'warn'``: if any `nan` values are encountered will give a warning and continue
            - ``'ignore'``: all `nan` values are silently removed
            - a float: if a float is provided will impute any `nan` values with this value

        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Raises:
        ValueError:
            If ``nan_strategy`` is not one of ``error``, ``warn``, ``ignore`` or a float

    Example:
        >>> from torch import tensor
        >>> from torchmetrics.aggregation import RunningSum
        >>> metric = RunningSum(window=3)
        >>> for i in range(6):
        ...     current_val = metric(tensor([i]))
        ...     running_val = metric.compute()
        ...     total_val = tensor(sum(list(range(i+1))))  # total sum over all samples
        ...     print(f"{current_val=}, {running_val=}, {total_val=}")
        current_val=tensor(0.), running_val=tensor(0.), total_val=tensor(0)
        current_val=tensor(1.), running_val=tensor(1.), total_val=tensor(1)
        current_val=tensor(2.), running_val=tensor(3.), total_val=tensor(3)
        current_val=tensor(3.), running_val=tensor(6.), total_val=tensor(6)
        current_val=tensor(4.), running_val=tensor(9.), total_val=tensor(10)
        current_val=tensor(5.), running_val=tensor(12.), total_val=tensor(15)

    """

    def __init__(
        self,
        window: int = 5,
        nan_strategy: Union[str, float] = "warn",
        **kwargs: Any,
    ) -> None:
        super().__init__(base_metric=SumMetric(nan_strategy=nan_strategy, **kwargs), window=window)
