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
# this is just a bypass for this module name collision with built-in one
from collections import OrderedDict
from collections.abc import Hashable, Iterable, Iterator, Mapping, Sequence
from copy import deepcopy
from typing import Any, ClassVar, Dict, List, Optional, Union

import torch
from torch import Tensor
from torch.nn import ModuleDict
from typing_extensions import Literal

from torchmetrics.metric import Metric
from torchmetrics.utilities import rank_zero_warn
from torchmetrics.utilities.data import _flatten_dict, allclose
from torchmetrics.utilities.imports import _MATPLOTLIB_AVAILABLE
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE, plot_single_or_multi_val

if not _MATPLOTLIB_AVAILABLE:
    __doctest_skip__ = ["MetricCollection.plot", "MetricCollection.plot_all"]


def _remove_prefix(string: str, prefix: str) -> str:
    """Patch for older version with missing method `removeprefix`.

    >>> _remove_prefix("prefix_string", "prefix_")
    'string'
    >>> _remove_prefix("not_prefix_string", "prefix_")
    'not_prefix_string'

    """
    return string[len(prefix) :] if string.startswith(prefix) else string


def _remove_suffix(string: str, suffix: str) -> str:
    """Patch for older version with missing method `removesuffix`.

    >>> _remove_suffix("string_suffix", "_suffix")
    'string'
    >>> _remove_suffix("string_suffix_missing", "_suffix")
    'string_suffix_missing'

    """
    return string[: -len(suffix)] if string.endswith(suffix) else string


class MetricCollection(ModuleDict):
    """MetricCollection class can be used to chain metrics that have the same call pattern into one single class.

    Args:
        metrics: One of the following

            * list or tuple (sequence): if metrics are passed in as a list or tuple, will use the metrics class name
              as key for output dict. Therefore, two metrics of the same class cannot be chained this way.

            * arguments: similar to passing in as a list, metrics passed in as arguments will use their metric
              class name as key for the output dict.

            * dict: if metrics are passed in as a dict, will use each key in the dict as key for output dict.
              Use this format if you want to chain together multiple of the same metric with different parameters.
              Note that the keys in the output dict will be sorted alphabetically.

        prefix: a string to append in front of the keys of the output dict

        postfix: a string to append after the keys of the output dict

        compute_groups:
            By default the MetricCollection will try to reduce the computations needed for the metrics in the collection
            by checking if they belong to the same **compute group**. All metrics in a compute group share the same
            metric state and are therefore only different in their compute step e.g. accuracy, precision and recall
            can all be computed from the true positives/negatives and false positives/negatives. By default,
            this argument is ``True`` which enables this feature. Set this argument to `False` for disabling
            this behaviour. Can also be set to a list of lists of metrics for setting the compute groups yourself.

    .. tip::
        The compute groups feature can significantly speedup the calculation of metrics under the right conditions.
        First, the feature is only available when calling the ``update`` method and not when calling ``forward`` method
        due to the internal logic of ``forward`` preventing this. Secondly, since we compute groups share metric
        states by reference, calling ``.items()``, ``.values()`` etc. on the metric collection will break this
        reference and a copy of states are instead returned in this case (reference will be reestablished on the next
        call to ``update``).

    .. important::
        Metric collections can be nested at initialization (see last example) but the output of the collection will
        still be a single flatten dictionary combining the prefix and postfix arguments from the nested collection.

    Raises:
        ValueError:
            If one of the elements of ``metrics`` is not an instance of ``pl.metrics.Metric``.
        ValueError:
            If two elements in ``metrics`` have the same ``name``.
        ValueError:
            If ``metrics`` is not a ``list``, ``tuple`` or a ``dict``.
        ValueError:
            If ``metrics`` is ``dict`` and additional_metrics are passed in.
        ValueError:
            If ``prefix`` is set and it is not a string.
        ValueError:
            If ``postfix`` is set and it is not a string.

    Example::
        In the most basic case, the metrics can be passed in as a list or tuple. The keys of the output dict will be
        the same as the class name of the metric:

        >>> from torch import tensor
        >>> from pprint import pprint
        >>> from torchmetrics import MetricCollection
        >>> from torchmetrics.regression import MeanSquaredError
        >>> from torchmetrics.classification import MulticlassAccuracy, MulticlassPrecision, MulticlassRecall
        >>> target = tensor([0, 2, 0, 2, 0, 1, 0, 2])
        >>> preds = tensor([2, 1, 2, 0, 1, 2, 2, 2])
        >>> metrics = MetricCollection([MulticlassAccuracy(num_classes=3, average='micro'),
        ...                             MulticlassPrecision(num_classes=3, average='macro'),
        ...                             MulticlassRecall(num_classes=3, average='macro')])
        >>> metrics(preds, target)  # doctest: +NORMALIZE_WHITESPACE
        {'MulticlassAccuracy': tensor(0.1250),
         'MulticlassPrecision': tensor(0.0667),
         'MulticlassRecall': tensor(0.1111)}

    Example::
        Alternatively, metrics can be passed in as arguments. The keys of the output dict will be the same as the
        class name of the metric:

        >>> metrics = MetricCollection(MulticlassAccuracy(num_classes=3, average='micro'),
        ...                            MulticlassPrecision(num_classes=3, average='macro'),
        ...                            MulticlassRecall(num_classes=3, average='macro'))
        >>> metrics(preds, target)  # doctest: +NORMALIZE_WHITESPACE
        {'MulticlassAccuracy': tensor(0.1250),
         'MulticlassPrecision': tensor(0.0667),
         'MulticlassRecall': tensor(0.1111)}

    Example::
        If multiple of the same metric class (with different parameters) should be chained together, metrics can be
        passed in as a dict and the output dict will have the same keys as the input dict:

        >>> metrics = MetricCollection({'micro_recall': MulticlassRecall(num_classes=3, average='micro'),
        ...                             'macro_recall': MulticlassRecall(num_classes=3, average='macro')})
        >>> same_metric = metrics.clone()
        >>> pprint(metrics(preds, target))
        {'macro_recall': tensor(0.1111), 'micro_recall': tensor(0.1250)}
        >>> pprint(same_metric(preds, target))
        {'macro_recall': tensor(0.1111), 'micro_recall': tensor(0.1250)}

    Example::
        Metric collections can also be nested up to a single time. The output of the collection will still be a single
        dict with the prefix and postfix arguments from the nested collection:

        >>> metrics = MetricCollection([
        ...     MetricCollection([
        ...         MulticlassAccuracy(num_classes=3, average='macro'),
        ...         MulticlassPrecision(num_classes=3, average='macro')
        ...     ], postfix='_macro'),
        ...     MetricCollection([
        ...         MulticlassAccuracy(num_classes=3, average='micro'),
        ...         MulticlassPrecision(num_classes=3, average='micro')
        ...     ], postfix='_micro'),
        ... ], prefix='valmetrics/')
        >>> pprint(metrics(preds, target))  # doctest: +NORMALIZE_WHITESPACE
        {'valmetrics/MulticlassAccuracy_macro': tensor(0.1111),
         'valmetrics/MulticlassAccuracy_micro': tensor(0.1250),
         'valmetrics/MulticlassPrecision_macro': tensor(0.0667),
         'valmetrics/MulticlassPrecision_micro': tensor(0.1250)}

    Example::
        The `compute_groups` argument allow you to specify which metrics should share metric state. By default, this
        will automatically be derived but can also be set manually.

        >>> metrics = MetricCollection(
        ...     MulticlassRecall(num_classes=3, average='macro'),
        ...     MulticlassPrecision(num_classes=3, average='macro'),
        ...     MeanSquaredError(),
        ...     compute_groups=[['MulticlassRecall', 'MulticlassPrecision'], ['MeanSquaredError']]
        ... )
        >>> metrics.update(preds, target)
        >>> pprint(metrics.compute())
        {'MeanSquaredError': tensor(2.3750), 'MulticlassPrecision': tensor(0.0667), 'MulticlassRecall': tensor(0.1111)}
        >>> pprint(metrics.compute_groups)
        {0: ['MulticlassRecall', 'MulticlassPrecision'], 1: ['MeanSquaredError']}

    """

    _modules: dict[str, Metric]  # type: ignore[assignment]
    _groups: Dict[int, List[str]]
    __jit_unused_properties__: ClassVar[list[str]] = ["metric_state"]

    def __init__(
        self,
        metrics: Union[Metric, Sequence[Metric], dict[str, Metric]],
        *additional_metrics: Metric,
        prefix: Optional[str] = None,
        postfix: Optional[str] = None,
        compute_groups: Union[bool, list[list[str]]] = True,
    ) -> None:
        super().__init__()

        self.prefix = self._check_arg(prefix, "prefix")
        self.postfix = self._check_arg(postfix, "postfix")
        self._enable_compute_groups = compute_groups
        self._groups_checked: bool = False
        self._state_is_copy: bool = False

        self.add_metrics(metrics, *additional_metrics)

    @property
    def metric_state(self) -> dict[str, dict[str, Any]]:
        """Get the current state of the metric."""
        return {k: m.metric_state for k, m in self.items(keep_base=False, copy_state=False)}

    @torch.jit.unused
    def forward(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Call forward for each metric sequentially.

        Positional arguments (args) will be passed to every metric in the collection, while keyword arguments (kwargs)
        will be filtered based on the signature of the individual metric.

        """
        return self._compute_and_reduce("forward", *args, **kwargs)

    def update(self, *args: Any, **kwargs: Any) -> None:
        """Call update for each metric sequentially.

        Positional arguments (args) will be passed to every metric in the collection, while keyword arguments (kwargs)
        will be filtered based on the signature of the individual metric.

        """
        # Use compute groups if already initialized and checked
        if self._groups_checked:
            # Delete the cache of all metrics to invalidate the cache and therefore recent compute calls, forcing new
            # compute calls to recompute
            for k in self.keys(keep_base=True):
                mi = getattr(self, str(k))
                mi._computed = None
            for cg in self._groups.values():
                # only update the first member
                m0 = getattr(self, cg[0])
                m0.update(*args, **m0._filter_kwargs(**kwargs))
            if self._state_is_copy:
                # If we have deep copied state in between updates, reestablish link
                self._compute_groups_create_state_ref()
                self._state_is_copy = False
        else:  # the first update always do per metric to form compute groups
            for m in self.values(copy_state=False):
                m_kwargs = m._filter_kwargs(**kwargs)
                m.update(*args, **m_kwargs)

            if self._enable_compute_groups:
                self._merge_compute_groups()
                # create reference between states
                self._compute_groups_create_state_ref()
                self._groups_checked = True

    def _merge_compute_groups(self) -> None:
        """Iterate over the collection of metrics, checking if the state of each metric matches another.

        If so, their compute groups will be merged into one. The complexity of the method is approximately
        ``O(number_of_metrics_in_collection ** 2)``, as all metrics need to be compared to all other metrics.

        """
        num_groups = len(self._groups)
        while True:
            for cg_idx1, cg_members1 in deepcopy(self._groups).items():
                for cg_idx2, cg_members2 in deepcopy(self._groups).items():
                    if cg_idx1 == cg_idx2:
                        continue

                    metric1 = getattr(self, cg_members1[0])
                    metric2 = getattr(self, cg_members2[0])

                    if self._equal_metric_states(metric1, metric2):
                        self._groups[cg_idx1].extend(self._groups.pop(cg_idx2))
                        break

                # Start over if we merged groups
                if len(self._groups) != num_groups:
                    break

            # Stop when we iterate over everything and do not merge any groups
            if len(self._groups) == num_groups:
                break
            num_groups = len(self._groups)

        # Re-index groups
        temp = deepcopy(self._groups)
        self._groups = {}
        for idx, values in enumerate(temp.values()):
            self._groups[idx] = values

    @staticmethod
    def _equal_metric_states(metric1: Metric, metric2: Metric) -> bool:
        """Check if the metric state of two metrics are the same."""
        # empty state
        if len(metric1._defaults) == 0 or len(metric2._defaults) == 0:
            return False

        if metric1._defaults.keys() != metric2._defaults.keys():
            return False

        for key in metric1._defaults:
            state1 = getattr(metric1, key)
            state2 = getattr(metric2, key)

            if type(state1) != type(state2):  # noqa: E721
                return False

            if isinstance(state1, Tensor) and isinstance(state2, Tensor):
                if not (state1.shape == state2.shape and allclose(state1, state2)):
                    return False

            if isinstance(state1, list) and isinstance(state2, list):
                if not (all(s1.shape == s2.shape and allclose(s1, s2) for s1, s2 in zip(state1, state2))):
                    return False

        return True

    def _compute_groups_create_state_ref(self, copy: bool = False) -> None:
        """Create reference between metrics in the same compute group.

        Args:
            copy: If `True` the metric state will between members will be copied instead
                of just passed by reference

        """
        if not self._state_is_copy:
            for cg in self._groups.values():
                m0 = getattr(self, cg[0])
                for i in range(1, len(cg)):
                    mi = getattr(self, cg[i])
                    for state in m0._defaults:
                        m0_state = getattr(m0, state)
                        # Determine if we just should set a reference or a full copy
                        setattr(mi, state, deepcopy(m0_state) if copy else m0_state)
                    mi._update_count = deepcopy(m0._update_count) if copy else m0._update_count
        self._state_is_copy = copy

    def compute(self) -> dict[str, Any]:
        """Compute the result for each metric in the collection."""
        return self._compute_and_reduce("compute")

    def _compute_and_reduce(
        self, method_name: Literal["compute", "forward"], *args: Any, **kwargs: Any
    ) -> dict[str, Any]:
        """Compute result from collection and reduce into a single dictionary.

        Args:
            method_name: The method to call on each metric in the collection.
                Should be either `compute` or `forward`.
            args: Positional arguments to pass to each metric (if method_name is `forward`)
            kwargs: Keyword arguments to pass to each metric (if method_name is `forward`)

        Raises:
            ValueError:
                If method_name is not `compute` or `forward`.

        """
        result = {}
        for k, m in self.items(keep_base=True, copy_state=False):
            if method_name == "compute":
                res = m.compute()
            elif method_name == "forward":
                res = m(*args, **m._filter_kwargs(**kwargs))
            else:
                raise ValueError(f"method_name should be either 'compute' or 'forward', but got {method_name}")
            result[k] = res

        _, duplicates = _flatten_dict(result)

        flattened_results = {}
        for k, m in self.items(keep_base=True, copy_state=False):
            res = result[k]
            if isinstance(res, dict):
                for key, v in res.items():
                    # if duplicates of keys we need to add unique prefix to each key
                    if duplicates:
                        stripped_k = k.replace(getattr(m, "prefix", ""), "")
                        stripped_k = stripped_k.replace(getattr(m, "postfix", ""), "")
                        key = f"{stripped_k}_{key}"
                    if getattr(m, "_from_collection", None) and m.prefix is not None:
                        key = f"{m.prefix}{key}"
                    if getattr(m, "_from_collection", None) and m.postfix is not None:
                        key = f"{key}{m.postfix}"
                    flattened_results[key] = v
            else:
                flattened_results[k] = res
        return {self._set_name(k): v for k, v in flattened_results.items()}

    def reset(self) -> None:
        """Call reset for each metric sequentially."""
        for m in self.values(copy_state=False):
            m.reset()
        if self._enable_compute_groups and self._groups_checked:
            # reset state reference
            self._compute_groups_create_state_ref()

    def clone(self, prefix: Optional[str] = None, postfix: Optional[str] = None) -> "MetricCollection":
        """Make a copy of the metric collection.

        Args:
            prefix: a string to append in front of the metric keys
            postfix: a string to append after the keys of the output dict.

        """
        mc = deepcopy(self)
        if prefix:
            mc.prefix = self._check_arg(prefix, "prefix")
        if postfix:
            mc.postfix = self._check_arg(postfix, "postfix")
        return mc

    def persistent(self, mode: bool = True) -> None:
        """Change if metric states should be saved to its state_dict after initialization."""
        for m in self.values(copy_state=False):
            m.persistent(mode)

    def add_metrics(
        self, metrics: Union[Metric, Sequence[Metric], dict[str, Metric]], *additional_metrics: Metric
    ) -> None:
        """Add new metrics to Metric Collection."""
        if isinstance(metrics, Metric):
            # set compatible with original type expectations
            metrics = [metrics]
        if isinstance(metrics, Sequence):
            # prepare for optional additions
            metrics = list(metrics)
            remain: list = []
            for m in additional_metrics:
                sel = metrics if isinstance(m, Metric) else remain
                sel.append(m)

            if remain:
                rank_zero_warn(
                    f"You have passes extra arguments {remain} which are not `Metric` so they will be ignored."
                )
        elif additional_metrics:
            raise ValueError(
                f"You have passes extra arguments {additional_metrics} which are not compatible"
                f" with first passed dictionary {metrics} so they will be ignored."
            )

        if isinstance(metrics, dict):
            # Check all values are metrics
            # Make sure that metrics are added in deterministic order
            for name in sorted(metrics.keys()):
                metric = metrics[name]
                if not isinstance(metric, (Metric, MetricCollection)):
                    raise ValueError(
                        f"Value {metric} belonging to key {name} is not an instance of"
                        " `torchmetrics.Metric` or `torchmetrics.MetricCollection`"
                    )
                if isinstance(metric, Metric):
                    self[name] = metric
                else:
                    for k, v in metric.items(keep_base=False):
                        v.postfix = metric.postfix
                        v.prefix = metric.prefix
                        v._from_collection = True
                        self[f"{name}_{k}"] = v
        elif isinstance(metrics, Sequence):
            for metric in metrics:
                if not isinstance(metric, (Metric, MetricCollection)):
                    raise ValueError(
                        f"Input {metric} to `MetricCollection` is not a instance of"
                        " `torchmetrics.Metric` or `torchmetrics.MetricCollection`"
                    )
                if isinstance(metric, Metric):
                    name = metric.__class__.__name__
                    if name in self:
                        raise ValueError(f"Encountered two metrics both named {name}")
                    self[name] = metric
                else:
                    for k, v in metric.items(keep_base=False):
                        v.postfix = metric.postfix
                        v.prefix = metric.prefix
                        v._from_collection = True
                        self[k] = v
        else:
            raise ValueError(
                "Unknown input to MetricCollection. Expected, `Metric`, `MetricCollection` or `dict`/`sequence` of the"
                f" previous, but got {metrics}"
            )

        self._groups_checked = False
        if self._enable_compute_groups:
            self._init_compute_groups()
        else:
            self._groups = {}

    def _init_compute_groups(self) -> None:
        """Initialize compute groups.

        If user provided a list, we check that all metrics in the list are also in the collection. If set to `True` we
        simply initialize each metric in the collection as its own group

        """
        if isinstance(self._enable_compute_groups, list):
            self._groups = dict(enumerate(self._enable_compute_groups))
            for v in self._groups.values():
                for metric in v:
                    if metric not in self:
                        raise ValueError(
                            f"Input {metric} in `compute_groups` argument does not match a metric in the collection."
                            f" Please make sure that {self._enable_compute_groups} matches {self.keys(keep_base=True)}"
                        )
            self._groups_checked = True
        else:
            # Initialize all metrics as their own compute group
            self._groups = {i: [str(k)] for i, k in enumerate(self.keys(keep_base=True))}

    @property
    def compute_groups(self) -> Dict[int, List[str]]:
        """Return a dict with the current compute groups in the collection."""
        return self._groups

    def _set_name(self, base: str) -> str:
        """Adjust name of metric with both prefix and postfix."""
        name = base if self.prefix is None else self.prefix + base
        return name if self.postfix is None else name + self.postfix

    def _to_renamed_dict(self) -> Mapping[str, Metric]:
        # self._modules changed from OrderedDict to dict as of PyTorch 2.5.0
        dict_modules = OrderedDict() if isinstance(self._modules, OrderedDict) else {}
        for k, v in self._modules.items():
            dict_modules[self._set_name(k)] = v
        return dict_modules

    def __iter__(self) -> Iterator[Hashable]:
        """Return an iterator over the keys of the MetricDict."""
        return iter(self.keys())

    # TODO: redefine this as native python dict
    def keys(self, keep_base: bool = False) -> Iterable[Hashable]:
        r"""Return an iterable of the ModuleDict key.

        Args:
            keep_base: Whether to add prefix/postfix on the items collection.

        """
        if keep_base:
            return self._modules.keys()
        return self._to_renamed_dict().keys()

    def items(self, keep_base: bool = False, copy_state: bool = True) -> Iterable[tuple[str, Metric]]:
        r"""Return an iterable of the ModuleDict key/value pairs.

        Args:
            keep_base: Whether to add prefix/postfix on the collection.
            copy_state:
                If metric states should be copied between metrics in the same compute group or just passed by reference

        """
        self._compute_groups_create_state_ref(copy_state)
        if keep_base:
            return self._modules.items()
        return self._to_renamed_dict().items()

    def values(self, copy_state: bool = True) -> Iterable[Metric]:
        """Return an iterable of the ModuleDict values.

        Args:
            copy_state:
                If metric states should be copied between metrics in the same compute group or just passed by reference

        """
        self._compute_groups_create_state_ref(copy_state)
        return self._modules.values()

    def __getitem__(self, key: str, copy_state: bool = True) -> Metric:
        """Retrieve a single metric from the collection.

        Args:
            key: name of metric to retrieve
            copy_state:
                If metric states should be copied between metrics in the same compute group or just passed by reference

        """
        self._compute_groups_create_state_ref(copy_state)
        if self.prefix:
            key = _remove_prefix(key, self.prefix)
        if self.postfix:
            key = _remove_suffix(key, self.postfix)
        return self._modules[key]

    @staticmethod
    def _check_arg(arg: Optional[str], name: str) -> Optional[str]:
        if arg is None or isinstance(arg, str):
            return arg
        raise ValueError(f"Expected input `{name}` to be a string, but got {type(arg)}")

    def __repr__(self) -> str:
        """Return the representation of the metric collection including all metrics in the collection."""
        repr_str = super().__repr__()[:-2]
        if self.prefix:
            repr_str += f",\n  prefix={self.prefix}{',' if self.postfix else ''}"
        if self.postfix:
            repr_str += f"{',' if not self.prefix else ''}\n  postfix={self.postfix}"
        return repr_str + "\n)"

    def set_dtype(self, dst_type: Union[str, torch.dtype]) -> "MetricCollection":
        """Transfer all metric state to specific dtype. Special version of standard `type` method.

        Arguments:
            dst_type: the desired type as ``torch.dtype`` or string.

        """
        for m in self.values(copy_state=False):
            m.set_dtype(dst_type)
        return self

    def plot(
        self,
        val: Optional[Union[dict, Sequence[dict]]] = None,
        ax: Optional[Union[_AX_TYPE, Sequence[_AX_TYPE]]] = None,
        together: bool = False,
    ) -> Sequence[_PLOT_OUT_TYPE]:
        """Plot a single or multiple values from the metric.

        The plot method has two modes of operation. If argument `together` is set to `False` (default), the `.plot`
        method of each metric will be called individually and the result will be list of figures. If `together` is set
        to `True`, the values of all metrics will instead be plotted in the same figure.

        Args:
            val: Either a single result from calling `metric.forward` or `metric.compute` or a list of these results.
                If no value is provided, will automatically call `metric.compute` and plot that result.
            ax: Either a single instance of matplotlib axis object or an sequence of matplotlib axis objects. If
                provided, will add the plots to the provided axis objects. If not provided, will create a new. If
                argument `together` is set to `True`, a single object is expected. If `together` is set to `False`,
                the number of axis objects needs to be the same length as the number of metrics in the collection.
            together: If `True`, will plot all metrics in the same axis. If `False`, will plot each metric in a separate

        Returns:
            Either install tuple of Figure and Axes object or an sequence of tuples with Figure and Axes object for each
            metric in the collection.

        Raises:
            ModuleNotFoundError:
                If `matplotlib` is not installed
            ValueError:
                If `together` is not an bool
            ValueError:
                If `ax` is not an instance of matplotlib axis object or a sequence of matplotlib axis objects

        .. plot::
            :scale: 75

            >>> # Example plotting a single value
            >>> import torch
            >>> from torchmetrics import MetricCollection
            >>> from torchmetrics.classification import BinaryAccuracy, BinaryPrecision, BinaryRecall
            >>> metrics = MetricCollection([BinaryAccuracy(), BinaryPrecision(), BinaryRecall()])
            >>> metrics.update(torch.rand(10), torch.randint(2, (10,)))
            >>> fig_ax_ = metrics.plot()

        .. plot::
            :scale: 75

            >>> # Example plotting multiple values
            >>> import torch
            >>> from torchmetrics import MetricCollection
            >>> from torchmetrics.classification import BinaryAccuracy, BinaryPrecision, BinaryRecall
            >>> metrics = MetricCollection([BinaryAccuracy(), BinaryPrecision(), BinaryRecall()])
            >>> values = []
            >>> for _ in range(10):
            ...     values.append(metrics(torch.rand(10), torch.randint(2, (10,))))
            >>> fig_, ax_ = metrics.plot(values, together=True)

        """
        if not isinstance(together, bool):
            raise ValueError(f"Expected argument `together` to be a boolean, but got {type(together)}")
        if ax is not None:
            if together and not isinstance(ax, _AX_TYPE):
                raise ValueError(
                    f"Expected argument `ax` to be a matplotlib axis object, but got {type(ax)} when `together=True`"
                )
            if not together and not (
                isinstance(ax, Sequence) and all(isinstance(a, _AX_TYPE) for a in ax) and len(ax) == len(self)
            ):
                raise ValueError(
                    f"Expected argument `ax` to be a sequence of matplotlib axis objects with the same length as the "
                    f"number of metrics in the collection, but got {type(ax)} with len {len(ax)} when `together=False`"
                )
        val = val or self.compute()
        if together:
            return plot_single_or_multi_val(val, ax=ax)
        fig_axs = []
        for i, (k, m) in enumerate(self.items(keep_base=False, copy_state=False)):
            if isinstance(val, dict):
                f, a = m.plot(val[k], ax=ax[i] if ax is not None else ax)
            elif isinstance(val, Sequence):
                f, a = m.plot([v[k] for v in val], ax=ax[i] if ax is not None else ax)
            fig_axs.append((f, a))
        return fig_axs
