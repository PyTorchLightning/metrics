# Copyright The PyTorch Lightning team.
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
from typing import Optional, Tuple

from torch import Tensor, tensor

from torchmetrics.classification.stat_scores import _reduce_stat_scores
from torchmetrics.functional.classification.stat_scores import _stat_scores_update
from torchmetrics.utilities.checks import _check_classification_inputs, _input_format_classification, _input_squeeze
from torchmetrics.utilities.enums import DataType


def _check_subset_validity(mode):
    return mode in (DataType.MULTILABEL, DataType.MULTIDIM_MULTICLASS)


def _mode(
    preds: Tensor,
    target: Tensor,
    threshold: float,
    top_k: Optional[int],
    num_classes: Optional[int],
    multiclass: Optional[bool],
) -> DataType:
    mode = _check_classification_inputs(
        preds, target, threshold=threshold, top_k=top_k, num_classes=num_classes, multiclass=multiclass
    )
    return mode


def _accuracy_update(
    preds: Tensor,
    target: Tensor,
    reduce: str,
    mdmc_reduce: str,
    threshold: float,
    num_classes: Optional[int],
    top_k: Optional[int],
    multiclass: Optional[bool],
    ignore_index: Optional[int],
    mode: DataType,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    if mode == DataType.MULTILABEL and top_k:
        raise ValueError("You can not use the `top_k` parameter to calculate accuracy for multi-label inputs.")

    preds, target = _input_squeeze(preds, target)
    tp, fp, tn, fn = _stat_scores_update(
        preds,
        target,
        reduce=reduce,
        mdmc_reduce=mdmc_reduce,
        threshold=threshold,
        num_classes=num_classes,
        top_k=top_k,
        multiclass=multiclass,
        ignore_index=ignore_index,
    )
    return tp, fp, tn, fn


def _accuracy_compute(
    tp: Tensor, fp: Tensor, tn: Tensor, fn: Tensor, average: str, mdmc_average: str, mode: DataType
) -> Tensor:
    simple_average = ["micro", "samples"]
    if (mode == DataType.BINARY and average in simple_average) or mode == DataType.MULTILABEL:
        numerator = tp + tn
        denominator = tp + tn + fp + fn
    else:
        numerator = tp
        denominator = tp + fn
    return _reduce_stat_scores(
        numerator=numerator,
        denominator=denominator,
        weights=None if average != "weighted" else tp + fn,
        average=average,
        mdmc_average=mdmc_average,
    )


def _subset_accuracy_update(
    preds: Tensor,
    target: Tensor,
    threshold: float,
    top_k: Optional[int],
) -> Tuple[Tensor, Tensor]:

    preds, target = _input_squeeze(preds, target)
    preds, target, mode = _input_format_classification(preds, target, threshold=threshold, top_k=top_k)

    if mode == DataType.MULTILABEL and top_k:
        raise ValueError("You can not use the `top_k` parameter to calculate accuracy for multi-label inputs.")

    if mode == DataType.MULTILABEL:
        correct = (preds == target).all(dim=1).sum()
        total = tensor(target.shape[0], device=target.device)
    elif mode == DataType.MULTICLASS:
        correct = (preds * target).sum()
        total = target.sum()
    elif mode == DataType.MULTIDIM_MULTICLASS:
        sample_correct = (preds * target).sum(dim=(1, 2))
        correct = (sample_correct == target.shape[2]).sum()
        total = tensor(target.shape[0], device=target.device)

    return correct, total


def _subset_accuracy_compute(correct: Tensor, total: Tensor) -> Tensor:
    return correct.float() / total


def accuracy(
    preds: Tensor,
    target: Tensor,
    average: str = "micro",
    mdmc_average: Optional[str] = "global",
    threshold: float = 0.5,
    top_k: Optional[int] = None,
    subset_accuracy: bool = False,
    num_classes: Optional[int] = None,
    multiclass: Optional[bool] = None,
    ignore_index: Optional[int] = None,
) -> Tensor:
    r"""Computes `Accuracy <https://en.wikipedia.org/wiki/Accuracy_and_precision>`_:

    .. math::
        \text{Accuracy} = \frac{1}{N}\sum_i^N 1(y_i = \hat{y}_i)

    Where :math:`y` is a tensor of target values, and :math:`\hat{y}` is a
    tensor of predictions.

    For multi-class and multi-dimensional multi-class data with probability or logits predictions, the
    parameter ``top_k`` generalizes this metric to a Top-K accuracy metric: for each sample the
    top-K highest probability or logits items are considered to find the correct label.

    For multi-label and multi-dimensional multi-class inputs, this metric computes the "global"
    accuracy by default, which counts all labels or sub-samples separately. This can be
    changed to subset accuracy (which requires all labels or sub-samples in the sample to
    be correctly predicted) by setting ``subset_accuracy=True``.

    Accepts all input types listed in :ref:`references/modules:input types`.

    Args:
        preds: Predictions from model (probabilities, logits or labels)
        target: Ground truth labels
        average:
            Defines the reduction that is applied. Should be one of the following:

            - ``'micro'`` [default]: Calculate the metric globally, across all samples and classes.
            - ``'macro'``: Calculate the metric for each class separately, and average the
              metrics across classes (with equal weights for each class).
            - ``'weighted'``: Calculate the metric for each class separately, and average the
              metrics across classes, weighting each class by its support (``tp + fn``).
            - ``'none'`` or ``None``: Calculate the metric for each class separately, and return
              the metric for every class.
            - ``'samples'``: Calculate the metric for each sample, and average the metrics
              across samples (with equal weights for each sample).

            .. note:: What is considered a sample in the multi-dimensional multi-class case
                depends on the value of ``mdmc_average``.

        mdmc_average:
            Defines how averaging is done for multi-dimensional multi-class inputs (on top of the
            ``average`` parameter). Should be one of the following:

            - ``None`` [default]: Should be left unchanged if your data is not multi-dimensional
              multi-class.

            - ``'samplewise'``: In this case, the statistics are computed separately for each
              sample on the ``N`` axis, and then averaged over samples.
              The computation for each sample is done by treating the flattened extra axes ``...``
              (see :ref:`references/modules:input types`) as the ``N`` dimension within the sample,
              and computing the metric for the sample based on that.

            - ``'global'``: In this case the ``N`` and ``...`` dimensions of the inputs
              (see :ref:`references/modules:input types`)
              are flattened into a new ``N_X`` sample axis, i.e. the inputs are treated as if they
              were ``(N_X, C)``. From here on the ``average`` parameter applies as usual.
        num_classes:
            Number of classes. Necessary for ``'macro'``, ``'weighted'`` and ``None`` average methods.

        threshold:
            Threshold for transforming probability or logit predictions to binary (0,1) predictions, in the case
            of binary or multi-label inputs. Default value of 0.5 corresponds to input being probabilities.
        top_k:
            Number of highest probability or logit score predictions considered to find the correct label,
            relevant only for (multi-dimensional) multi-class inputs. The
            default value (``None``) will be interpreted as 1 for these inputs.

            Should be left at default (``None``) for all other types of inputs.
        multiclass:
            Used only in certain special cases, where you want to treat inputs as a different type
            than what they appear to be. See the parameter's
            :ref:`documentation section <references/modules:using the multiclass parameter>`
            for a more detailed explanation and examples.
        ignore_index:
            Integer specifying a target class to ignore. If given, this class index does not contribute
            to the returned score, regardless of reduction method. If an index is ignored, and ``average=None``
            or ``'none'``, the score for the ignored class will be returned as ``nan``.
        subset_accuracy:
            Whether to compute subset accuracy for multi-label and multi-dimensional
            multi-class inputs (has no effect for other input types).

            - For multi-label inputs, if the parameter is set to ``True``, then all labels for
              each sample must be correctly predicted for the sample to count as correct. If it
              is set to ``False``, then all labels are counted separately - this is equivalent to
              flattening inputs beforehand (i.e. ``preds = preds.flatten()`` and same for ``target``).

            - For multi-dimensional multi-class inputs, if the parameter is set to ``True``, then all
              sub-sample (on the extra axis) must be correct for the sample to be counted as correct.
              If it is set to ``False``, then all sub-samples are counter separately - this is equivalent,
              in the case of label predictions, to flattening the inputs beforehand (i.e.
              ``preds = preds.flatten()`` and same for ``target``). Note that the ``top_k`` parameter
              still applies in both cases, if set.

    Raises:
        ValueError:
            If ``threshold`` is not a ``float`` between ``0`` and ``1``.
        ValueError:
            If ``top_k`` parameter is set for ``multi-label`` inputs.
        ValueError:
            If ``average`` is none of ``"micro"``, ``"macro"``, ``"weighted"``, ``"samples"``, ``"none"``, ``None``.
        ValueError:
            If ``mdmc_average`` is not one of ``None``, ``"samplewise"``, ``"global"``.
        ValueError:
            If ``average`` is set but ``num_classes`` is not provided.
        ValueError:
            If ``num_classes`` is set
            and ``ignore_index`` is not in the range ``[0, num_classes)``.
        ValueError:
            If ``top_k`` is not an ``integer`` larger than ``0``.

    Example:
        >>> import torch
        >>> from torchmetrics.functional import accuracy
        >>> target = torch.tensor([0, 1, 2, 3])
        >>> preds = torch.tensor([0, 2, 1, 3])
        >>> accuracy(preds, target)
        tensor(0.5000)

        >>> target = torch.tensor([0, 1, 2])
        >>> preds = torch.tensor([[0.1, 0.9, 0], [0.3, 0.1, 0.6], [0.2, 0.5, 0.3]])
        >>> accuracy(preds, target, top_k=2)
        tensor(0.6667)
    """

    if not 0 < threshold < 1:
        raise ValueError(f"The `threshold` should be a float in the (0,1) interval, got {threshold}")

    allowed_average = ["micro", "macro", "weighted", "samples", "none", None]
    if average not in allowed_average:
        raise ValueError(f"The `average` has to be one of {allowed_average}, got {average}.")

    if average in ["macro", "weighted", "none", None] and (not num_classes or num_classes < 1):
        raise ValueError(f"When you set `average` as {average}, you have to provide the number of classes.")

    allowed_mdmc_average = [None, "samplewise", "global"]
    if mdmc_average not in allowed_mdmc_average:
        raise ValueError(f"The `mdmc_average` has to be one of {allowed_mdmc_average}, got {mdmc_average}.")

    if num_classes and ignore_index is not None and (not 0 <= ignore_index < num_classes or num_classes == 1):
        raise ValueError(f"The `ignore_index` {ignore_index} is not valid for inputs with {num_classes} classes")

    if top_k is not None and (not isinstance(top_k, int) or top_k <= 0):
        raise ValueError(f"The `top_k` should be an integer larger than 0, got {top_k}")

    preds, target = _input_squeeze(preds, target)
    mode = _mode(preds, target, threshold, top_k, num_classes, multiclass)
    reduce = "macro" if average in ["weighted", "none", None] else average

    if subset_accuracy and _check_subset_validity(mode):
        correct, total = _subset_accuracy_update(preds, target, threshold, top_k)
        return _subset_accuracy_compute(correct, total)
    tp, fp, tn, fn = _accuracy_update(
        preds, target, reduce, mdmc_average, threshold, num_classes, top_k, multiclass, ignore_index, mode
    )
    return _accuracy_compute(tp, fp, tn, fn, average, mdmc_average, mode)
