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
import warnings
from enum import Enum
from typing import Any, List, Optional, Tuple, Union

import humanfriendly
import torch
from torch import Tensor
from typing_extensions import Literal

from torchmetrics.functional.classification.precision_recall_curve import (
    _adjust_threshold_arg,
    _binary_clf_curve,
    _binary_precision_recall_curve_arg_validation,
    _binary_precision_recall_curve_compute,
    _binary_precision_recall_curve_format,
    _binary_precision_recall_curve_tensor_validation,
    _binary_precision_recall_curve_update,
    _multiclass_precision_recall_curve_arg_validation,
    _multiclass_precision_recall_curve_compute,
    _multiclass_precision_recall_curve_format,
    _multiclass_precision_recall_curve_tensor_validation,
    _multiclass_precision_recall_curve_update,
    _multilabel_precision_recall_curve_arg_validation,
    _multilabel_precision_recall_curve_compute,
    _multilabel_precision_recall_curve_format,
    _multilabel_precision_recall_curve_tensor_validation,
    _multilabel_precision_recall_curve_update,
)
from torchmetrics.metric import Metric
from torchmetrics.utilities.data import dim_zero_cat
from torchmetrics.utilities.enums import ClassificationTask

_DYNAMIC_THRESHOLDS_MIN_NSAMPLES = 1024**2  # 1MiB,TODO: find a better way to estimate a reasonable minimum


def _budget_bytes_to_nsamples(budget_bytes: int):
    # assume that both preds and target ("* 2") will be of size (N, 1) and of type float32 (4 bytes)
    return budget_bytes / (2 * 4)


def _validate_memory_budget(budget: int):
    if budget <= 0:
        raise ValueError("Budget must be larger than 0.")

    if _budget_bytes_to_nsamples(budget) <= _DYNAMIC_THRESHOLDS_MIN_NSAMPLES:
        warnings.warn(
            f"Budget is relatively small ({humanfriendly.format_size(budget, binary=True)}). "
            "The dynamic mode is recommended for bigger samples."
        )

    return budget


class BinaryPrecisionRecallCurve(Metric):
    r"""Compute the precision-recall curve for binary tasks. The curve consist of multiple pairs of precision and
    recall values evaluated at different thresholds, such that the tradeoff between the two values can been seen.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (:class:`~torch.Tensor`): A float tensor of shape ``(N, ...)``. Preds should be a tensor containing
      probabilities or logits for each observation. If preds has values outside [0,1] range we consider the input
      to be logits and will auto apply sigmoid per element.
    - ``target`` (:class:`~torch.Tensor`): An int tensor of shape ``(N, ...)``. Target should be a tensor containing
      ground truth labels, and therefore only contain {0,1} values (except if `ignore_index` is specified). The value
      1 always encodes the positive class.

    .. note::
       Additional dimension ``...`` will be flattened into the batch dimension.

    As output to ``forward`` and ``compute`` the metric returns the following output:

    - ``precision`` (:class:`~torch.Tensor`): if `thresholds=None` a list for each class is returned with an 1d
      tensor of size ``(n_thresholds+1, )`` with precision values (length may differ between classes). If `thresholds`
      is set to something else, then a single 2d tensor of size ``(n_classes, n_thresholds+1)`` with precision values
      is returned.
    - ``recall`` (:class:`~torch.Tensor`): if `thresholds=None` a list for each class is returned with an 1d tensor
      of size ``(n_thresholds+1, )`` with recall values (length may differ between classes). If `thresholds` is set to
      something else, then a single 2d tensor of size ``(n_classes, n_thresholds+1)`` with recall values is returned.
    - ``thresholds`` (:class:`~torch.Tensor`): if `thresholds=None` a list for each class is returned with an 1d
      tensor of size ``(n_thresholds, )`` with increasing threshold values (length may differ between classes). If
      `threshold` is set to something else, then a single 1d tensor of size ``(n_thresholds, )`` is returned with
      shared threshold values for all classes.

    .. note::
       The implementation both supports calculating the metric in a non-binned but accurate version and a binned version
       that is less accurate but more memory efficient. Setting the `thresholds` argument to `None` will activate the
       non-binned  version that uses memory of size :math:`\mathcal{O}(n_{samples})` whereas setting the `thresholds`
       argument to either an integer, list or a 1d tensor will use a binned version that uses memory of
       size :math:`\mathcal{O}(n_{thresholds})` (constant memory).

    Args:
        thresholds:
            Can be one of:

            - If set to `None`, will use a non-binned approach where thresholds are dynamically calculated from
              all the data. Most accurate but also most memory consuming approach.
            - If set to an `int` (larger than 1), will use that number of thresholds linearly spaced from
              0 to 1 as bins for the calculation.
            - If set to an `list` of floats, will use the indicated thresholds in the list as bins for the calculation
            - If set to an 1d `tensor` of floats, will use the indicated thresholds in the tensor as
              bins for the calculation.
            - If set to a `str`, the value is interpreted as a memory budget and the dynamic mode approach is used.

        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Example:
        >>> from torchmetrics.classification import BinaryPrecisionRecallCurve
        >>> preds = torch.tensor([0, 0.5, 0.7, 0.8])
        >>> target = torch.tensor([0, 1, 1, 0])
        >>> bprc = BinaryPrecisionRecallCurve(thresholds=None)
        >>> bprc(preds, target)  # doctest: +NORMALIZE_WHITESPACE
        (tensor([0.6667, 0.5000, 0.0000, 1.0000]),
         tensor([1.0000, 0.5000, 0.0000, 0.0000]),
         tensor([0.5000, 0.7000, 0.8000]))
        >>> bprc = BinaryPrecisionRecallCurve(thresholds=5)
        >>> bprc(preds, target)  # doctest: +NORMALIZE_WHITESPACE
        (tensor([0.5000, 0.6667, 0.6667, 0.0000, 0.0000, 1.0000]),
         tensor([1., 1., 1., 0., 0., 0.]),
         tensor([0.0000, 0.2500, 0.5000, 0.7500, 1.0000]))
    """
    is_differentiable: bool = False
    higher_is_better: Optional[bool] = None
    full_state_update: bool = False

    class _ComputationMode(Enum):
        """Internal state of the dynamic mode."""

        BINNED = "binned"
        NON_BINNED = "non-binned"
        NON_BINNED_DYNAMIC = "non-binned-dynamic"

    @staticmethod
    def _deduce_computation_mode(thresholds: Optional[Union[int, List[float], Tensor, str]]) -> _ComputationMode:
        if isinstance(thresholds, str):
            return BinaryPrecisionRecallCurve._ComputationMode.NON_BINNED_DYNAMIC
        elif thresholds is None:
            return BinaryPrecisionRecallCurve._ComputationMode.NON_BINNED
        else:
            return BinaryPrecisionRecallCurve._ComputationMode.BINNED

    def __init__(
        self,
        thresholds: Optional[Union[int, List[float], Tensor, str]] = None,
        ignore_index: Optional[int] = None,
        validate_args: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if validate_args:
            _binary_precision_recall_curve_arg_validation(thresholds, ignore_index)

        self.ignore_index = ignore_index
        self.validate_args = validate_args

        self._computation_mode = self._deduce_computation_mode(thresholds)
        thresholds = _adjust_threshold_arg(thresholds)

        if self._computation_mode == self._ComputationMode.NON_BINNED_DYNAMIC:
            self._memory_budget_bytes = _validate_memory_budget(thresholds)
            # used after the switch to binned mode
            self.register_buffer("thresholds", None)
            self.add_state(
                "confmat",
                default=torch.zeros(_budget_bytes_to_nsamples(self._memory_budget_bytes), 2, 2, dtype=torch.long),
                dist_reduce_fx="sum",
            )
            # they are deleted after the switch to binned mode
            self.preds = []
            self.target = []

        elif thresholds is None:
            self.thresholds = thresholds
            self.add_state("preds", default=[], dist_reduce_fx="cat")
            self.add_state("target", default=[], dist_reduce_fx="cat")
        else:
            self.register_buffer("thresholds", thresholds)
            self.add_state(
                "confmat", default=torch.zeros(len(thresholds), 2, 2, dtype=torch.long), dist_reduce_fx="sum"
            )

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update metric states."""
        if self.validate_args:
            _binary_precision_recall_curve_tensor_validation(preds, target, self.ignore_index)
        preds, target, _ = _binary_precision_recall_curve_format(preds, target, self.thresholds, self.ignore_index)
        state = _binary_precision_recall_curve_update(preds, target, self.thresholds)
        if isinstance(state, Tensor):
            self.confmat += state
        else:
            self.preds.append(state[0])
            self.target.append(state[1])

        if self._computation_mode != self._ComputationMode.NON_BINNED_DYNAMIC:
            return

        all_preds = dim_zero_cat(self.preds)
        mem_used = all_preds.element_size() * all_preds.nelement() * 2  # 2 accounts for the target

        if mem_used < self._memory_budget_bytes:
            return

        # switch to binned mode
        self.preds, self.target = all_preds, dim_zero_cat(self.target)
        _, _, self.thresholds = _binary_clf_curve(self.preds, self.target)
        # if the number of thr
        self.confmat = _binary_precision_recall_curve_update(self.preds, self.target, self.thresholds)
        del self.preds, self.target
        self._computation_mode = self._ComputationMode.BINNED

    def compute(self) -> Tuple[Tensor, Tensor, Tensor]:
        """Compute metric."""
        state = [dim_zero_cat(self.preds), dim_zero_cat(self.target)] if self.thresholds is None else self.confmat
        return _binary_precision_recall_curve_compute(state, self.thresholds)


class MulticlassPrecisionRecallCurve(Metric):
    r"""Compute the precision-recall curve for multiclass tasks. The curve consist of multiple pairs of precision
    and recall values evaluated at different thresholds, such that the tradeoff between the two values can been
    seen.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (:class:`~torch.Tensor`): A float tensor of shape ``(N, C, ...)``. Preds should be a tensor containing
      probabilities or logits for each observation. If preds has values outside [0,1] range we consider the input to
      be logits and will auto apply softmax per sample.
    - ``target`` (:class:`~torch.Tensor`): An int tensor of shape ``(N, ...)``. Target should be a tensor containing
      ground truth labels, and therefore only contain values in the [0, n_classes-1] range (except if `ignore_index`
      is specified).

    .. note::
       Additional dimension ``...`` will be flattened into the batch dimension.

    As output to ``forward`` and ``compute`` the metric returns the following output:

    - ``precision`` (:class:`~torch.Tensor`): A 1d tensor of size ``(n_thresholds+1, )`` with precision values
    - ``recall`` (:class:`~torch.Tensor`): A 1d tensor of size ``(n_thresholds+1, )`` with recall values
    - ``thresholds`` (:class:`~torch.Tensor`): A 1d tensor of size ``(n_thresholds, )`` with increasing threshold values

    .. note::
       The implementation both supports calculating the metric in a non-binned but accurate version and a binned version
       that is less accurate but more memory efficient. Setting the `thresholds` argument to `None` will activate the
       non-binned  version that uses memory of size :math:`\mathcal{O}(n_{samples})` whereas setting the `thresholds`
       argument to either an integer, list or a 1d tensor will use a binned version that uses memory of
       size :math:`\mathcal{O}(n_{thresholds} \times n_{classes})` (constant memory).

    Args:
        num_classes: Integer specifing the number of classes
        thresholds:
            Can be one of:

            - If set to `None`, will use a non-binned approach where thresholds are dynamically calculated from
              all the data. Most accurate but also most memory consuming approach.
            - If set to an `int` (larger than 1), will use that number of thresholds linearly spaced from
              0 to 1 as bins for the calculation.
            - If set to an `list` of floats, will use the indicated thresholds in the list as bins for the calculation
            - If set to an 1d `tensor` of floats, will use the indicated thresholds in the tensor as
              bins for the calculation.

        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Example:
        >>> from torchmetrics.classification import MulticlassPrecisionRecallCurve
        >>> preds = torch.tensor([[0.75, 0.05, 0.05, 0.05, 0.05],
        ...                       [0.05, 0.75, 0.05, 0.05, 0.05],
        ...                       [0.05, 0.05, 0.75, 0.05, 0.05],
        ...                       [0.05, 0.05, 0.05, 0.75, 0.05]])
        >>> target = torch.tensor([0, 1, 3, 2])
        >>> mcprc = MulticlassPrecisionRecallCurve(num_classes=5, thresholds=None)
        >>> precision, recall, thresholds = mcprc(preds, target)
        >>> precision  # doctest: +NORMALIZE_WHITESPACE
        [tensor([1., 1.]), tensor([1., 1.]), tensor([0.2500, 0.0000, 1.0000]),
         tensor([0.2500, 0.0000, 1.0000]), tensor([0., 1.])]
        >>> recall
        [tensor([1., 0.]), tensor([1., 0.]), tensor([1., 0., 0.]), tensor([1., 0., 0.]), tensor([nan, 0.])]
        >>> thresholds
        [tensor(0.7500), tensor(0.7500), tensor([0.0500, 0.7500]), tensor([0.0500, 0.7500]), tensor(0.0500)]
        >>> mcprc = MulticlassPrecisionRecallCurve(num_classes=5, thresholds=5)
        >>> mcprc(preds, target)  # doctest: +NORMALIZE_WHITESPACE
        (tensor([[0.2500, 1.0000, 1.0000, 1.0000, 0.0000, 1.0000],
                 [0.2500, 1.0000, 1.0000, 1.0000, 0.0000, 1.0000],
                 [0.2500, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000],
                 [0.2500, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000],
                 [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000]]),
         tensor([[1., 1., 1., 1., 0., 0.],
                 [1., 1., 1., 1., 0., 0.],
                 [1., 0., 0., 0., 0., 0.],
                 [1., 0., 0., 0., 0., 0.],
                 [0., 0., 0., 0., 0., 0.]]),
         tensor([0.0000, 0.2500, 0.5000, 0.7500, 1.0000]))
    """
    is_differentiable: bool = False
    higher_is_better: Optional[bool] = None
    full_state_update: bool = False

    def __init__(
        self,
        num_classes: int,
        thresholds: Optional[Union[int, List[float], Tensor]] = None,
        ignore_index: Optional[int] = None,
        validate_args: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if validate_args:
            _multiclass_precision_recall_curve_arg_validation(num_classes, thresholds, ignore_index)

        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.validate_args = validate_args

        thresholds = _adjust_threshold_arg(thresholds)
        if thresholds is None:
            self.thresholds = thresholds
            self.add_state("preds", default=[], dist_reduce_fx="cat")
            self.add_state("target", default=[], dist_reduce_fx="cat")
        else:
            self.register_buffer("thresholds", thresholds)
            self.add_state(
                "confmat",
                default=torch.zeros(len(thresholds), num_classes, 2, 2, dtype=torch.long),
                dist_reduce_fx="sum",
            )

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update metric states."""
        if self.validate_args:
            _multiclass_precision_recall_curve_tensor_validation(preds, target, self.num_classes, self.ignore_index)
        preds, target, _ = _multiclass_precision_recall_curve_format(
            preds, target, self.num_classes, self.thresholds, self.ignore_index
        )
        state = _multiclass_precision_recall_curve_update(preds, target, self.num_classes, self.thresholds)
        if isinstance(state, Tensor):
            self.confmat += state
        else:
            self.preds.append(state[0])
            self.target.append(state[1])

    def compute(self) -> Union[Tuple[Tensor, Tensor, Tensor], Tuple[List[Tensor], List[Tensor], List[Tensor]]]:
        """Compute metric."""
        state = [dim_zero_cat(self.preds), dim_zero_cat(self.target)] if self.thresholds is None else self.confmat
        return _multiclass_precision_recall_curve_compute(state, self.num_classes, self.thresholds)


class MultilabelPrecisionRecallCurve(Metric):
    r"""Compute the precision-recall curve for multilabel tasks. The curve consist of multiple pairs of precision
    and recall values evaluated at different thresholds, such that the tradeoff between the two values can been
    seen.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (:class:`~torch.Tensor`): A float tensor of shape ``(N, C, ...)``. Preds should be a tensor containing
      probabilities or logits for each observation. If preds has values outside [0,1] range we consider the input to
      be logits and will auto apply sigmoid per element.
    - ``target`` (:class:`~torch.Tensor`): An int tensor of shape ``(N, C, ...)``. Target should be a tensor containing
      ground truth labels, and therefore only contain {0,1} values (except if `ignore_index` is specified).

    .. note::
       Additional dimension ``...`` will be flattened into the batch dimension.

    As output to ``forward`` and ``compute`` the metric returns the following a tuple of either 3 tensors or
    3 lists containing:

    - ``precision`` (:class:`~torch.Tensor` or :class:`~List`): if `thresholds=None` a list for each label is returned
      with an 1d tensor of size ``(n_thresholds+1, )`` with precision values (length may differ between labels). If
      `thresholds` is set to something else, then a single 2d tensor of size ``(n_labels, n_thresholds+1)`` with
      precision values is returned.
    - ``recall`` (:class:`~torch.Tensor` or :class:`~List`): if `thresholds=None` a list for each label is returned
      with an 1d tensor of size ``(n_thresholds+1, )`` with recall values (length may differ between labels). If
      `thresholds` is set to something else, then a single 2d tensor of size ``(n_labels, n_thresholds+1)`` with recall
      values is returned.
    - ``thresholds`` (:class:`~torch.Tensor` or :class:`~List`): if `thresholds=None` a list for each label is
      returned with an 1d tensor of size ``(n_thresholds, )`` with increasing threshold values (length may differ
      between labels). If `threshold` is set to something else, then a single 1d tensor of size ``(n_thresholds, )``
      is returned with shared threshold values for all labels.

    .. note::
       The implementation both supports calculating the metric in a non-binned but accurate version and a binned version
       that is less accurate but more memory efficient. Setting the `thresholds` argument to `None` will activate the
       non-binned  version that uses memory of size :math:`\mathcal{O}(n_{samples})` whereas setting the `thresholds`
       argument to either an integer, list or a 1d tensor will use a binned version that uses memory of
       size :math:`\mathcal{O}(n_{thresholds} \times n_{labels})` (constant memory).

    Args:
        preds: Tensor with predictions
        target: Tensor with true labels
        num_labels: Integer specifing the number of labels
        thresholds:
            Can be one of:

            - If set to `None`, will use a non-binned approach where thresholds are dynamically calculated from
              all the data. Most accurate but also most memory consuming approach.
            - If set to an `int` (larger than 1), will use that number of thresholds linearly spaced from
              0 to 1 as bins for the calculation.
            - If set to an `list` of floats, will use the indicated thresholds in the list as bins for the calculation
            - If set to an 1d `tensor` of floats, will use the indicated thresholds in the tensor as
              bins for the calculation.

        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.

    Example:
        >>> from torchmetrics.classification import MultilabelPrecisionRecallCurve
        >>> preds = torch.tensor([[0.75, 0.05, 0.35],
        ...                       [0.45, 0.75, 0.05],
        ...                       [0.05, 0.55, 0.75],
        ...                       [0.05, 0.65, 0.05]])
        >>> target = torch.tensor([[1, 0, 1],
        ...                        [0, 0, 0],
        ...                        [0, 1, 1],
        ...                        [1, 1, 1]])
        >>> mlprc = MultilabelPrecisionRecallCurve(num_labels=3, thresholds=None)
        >>> precision, recall, thresholds = mlprc(preds, target)
        >>> precision  # doctest: +NORMALIZE_WHITESPACE
        [tensor([0.5000, 0.5000, 1.0000, 1.0000]), tensor([0.6667, 0.5000, 0.0000, 1.0000]),
         tensor([0.7500, 1.0000, 1.0000, 1.0000])]
        >>> recall  # doctest: +NORMALIZE_WHITESPACE
        [tensor([1.0000, 0.5000, 0.5000, 0.0000]), tensor([1.0000, 0.5000, 0.0000, 0.0000]),
         tensor([1.0000, 0.6667, 0.3333, 0.0000])]
        >>> thresholds  # doctest: +NORMALIZE_WHITESPACE
        [tensor([0.0500, 0.4500, 0.7500]), tensor([0.5500, 0.6500, 0.7500]),
         tensor([0.0500, 0.3500, 0.7500])]
        >>> mlprc = MultilabelPrecisionRecallCurve(num_labels=3, thresholds=5)
        >>> mlprc(preds, target)  # doctest: +NORMALIZE_WHITESPACE
        (tensor([[0.5000, 0.5000, 1.0000, 1.0000, 0.0000, 1.0000],
                 [0.5000, 0.6667, 0.6667, 0.0000, 0.0000, 1.0000],
                 [0.7500, 1.0000, 1.0000, 1.0000, 0.0000, 1.0000]]),
         tensor([[1.0000, 0.5000, 0.5000, 0.5000, 0.0000, 0.0000],
                 [1.0000, 1.0000, 1.0000, 0.0000, 0.0000, 0.0000],
                 [1.0000, 0.6667, 0.3333, 0.3333, 0.0000, 0.0000]]),
         tensor([0.0000, 0.2500, 0.5000, 0.7500, 1.0000]))
    """
    is_differentiable: bool = False
    higher_is_better: Optional[bool] = None
    full_state_update: bool = False

    def __init__(
        self,
        num_labels: int,
        thresholds: Optional[Union[int, List[float], Tensor]] = None,
        ignore_index: Optional[int] = None,
        validate_args: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if validate_args:
            _multilabel_precision_recall_curve_arg_validation(num_labels, thresholds, ignore_index)

        self.num_labels = num_labels
        self.ignore_index = ignore_index
        self.validate_args = validate_args

        thresholds = _adjust_threshold_arg(thresholds)
        if thresholds is None:
            self.thresholds = thresholds
            self.add_state("preds", default=[], dist_reduce_fx="cat")
            self.add_state("target", default=[], dist_reduce_fx="cat")
        else:
            self.register_buffer("thresholds", thresholds)
            self.add_state(
                "confmat",
                default=torch.zeros(len(thresholds), num_labels, 2, 2, dtype=torch.long),
                dist_reduce_fx="sum",
            )

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update metric states."""
        if self.validate_args:
            _multilabel_precision_recall_curve_tensor_validation(preds, target, self.num_labels, self.ignore_index)
        preds, target, _ = _multilabel_precision_recall_curve_format(
            preds, target, self.num_labels, self.thresholds, self.ignore_index
        )
        state = _multilabel_precision_recall_curve_update(preds, target, self.num_labels, self.thresholds)
        if isinstance(state, Tensor):
            self.confmat += state
        else:
            self.preds.append(state[0])
            self.target.append(state[1])

    def compute(self) -> Union[Tuple[Tensor, Tensor, Tensor], Tuple[List[Tensor], List[Tensor], List[Tensor]]]:
        """Compute metric."""
        state = [dim_zero_cat(self.preds), dim_zero_cat(self.target)] if self.thresholds is None else self.confmat
        return _multilabel_precision_recall_curve_compute(state, self.num_labels, self.thresholds, self.ignore_index)


class PrecisionRecallCurve:
    r"""Compute the precision-recall curve. The curve consist of multiple pairs of precision and recall values
    evaluated at different thresholds, such that the tradeoff between the two values can been seen.

    This function is a simple wrapper to get the task specific versions of this metric, which is done by setting the
    ``task`` argument to either ``'binary'``, ``'multiclass'`` or ``multilabel``. See the documentation of
    :mod:`BinaryPrecisionRecallCurve`, :mod:`MulticlassPrecisionRecallCurve` and
    :mod:`MultilabelPrecisionRecallCurve` for the specific details of each argument influence and examples.

    Legacy Example:
        >>> pred = torch.tensor([0, 0.1, 0.8, 0.4])
        >>> target = torch.tensor([0, 1, 1, 0])
        >>> pr_curve = PrecisionRecallCurve(task="binary")
        >>> precision, recall, thresholds = pr_curve(pred, target)
        >>> precision
        tensor([0.6667, 0.5000, 1.0000, 1.0000])
        >>> recall
        tensor([1.0000, 0.5000, 0.5000, 0.0000])
        >>> thresholds
        tensor([0.1000, 0.4000, 0.8000])

        >>> pred = torch.tensor([[0.75, 0.05, 0.05, 0.05, 0.05],
        ...                      [0.05, 0.75, 0.05, 0.05, 0.05],
        ...                      [0.05, 0.05, 0.75, 0.05, 0.05],
        ...                      [0.05, 0.05, 0.05, 0.75, 0.05]])
        >>> target = torch.tensor([0, 1, 3, 2])
        >>> pr_curve = PrecisionRecallCurve(task="multiclass", num_classes=5)
        >>> precision, recall, thresholds = pr_curve(pred, target)
        >>> precision
        [tensor([1., 1.]), tensor([1., 1.]), tensor([0.2500, 0.0000, 1.0000]),
         tensor([0.2500, 0.0000, 1.0000]), tensor([0., 1.])]
        >>> recall
        [tensor([1., 0.]), tensor([1., 0.]), tensor([1., 0., 0.]), tensor([1., 0., 0.]), tensor([nan, 0.])]
        >>> thresholds
        [tensor(0.7500), tensor(0.7500), tensor([0.0500, 0.7500]), tensor([0.0500, 0.7500]), tensor(0.0500)]
    """

    def __new__(
        cls,
        task: Literal["binary", "multiclass", "multilabel"],
        thresholds: Optional[Union[int, List[float], Tensor]] = None,
        num_classes: Optional[int] = None,
        num_labels: Optional[int] = None,
        ignore_index: Optional[int] = None,
        validate_args: bool = True,
        **kwargs: Any,
    ) -> Metric:
        """Initialize task metric."""
        task = ClassificationTask.from_str(task)
        kwargs.update({"thresholds": thresholds, "ignore_index": ignore_index, "validate_args": validate_args})
        if task == ClassificationTask.BINARY:
            return BinaryPrecisionRecallCurve(**kwargs)
        if task == ClassificationTask.MULTICLASS:
            assert isinstance(num_classes, int)
            return MulticlassPrecisionRecallCurve(num_classes, **kwargs)
        if task == ClassificationTask.MULTILABEL:
            assert isinstance(num_labels, int)
            return MultilabelPrecisionRecallCurve(num_labels, **kwargs)
