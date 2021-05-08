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
from typing import Optional

import numpy as np
import torch
from torch import Tensor

from torchmetrics.classification.stat_scores import _reduce_stat_scores
from torchmetrics.functional.classification.stat_scores import _stat_scores_update
from torchmetrics.utilities import _deprecation_warn_arg_multilabel
from torchmetrics.utilities.enums import AverageMethod, MDMCAverageMethod


def _safe_divide(num: Tensor, denom: Tensor):
    """ prevent zero division """
    denom[denom == 0.] = 1
    return num / denom


def _fbeta_compute(
    tp: Tensor,
    fp: Tensor,
    tn: Tensor,
    fn: Tensor,
    beta: float,
    ignore_index: Optional[int],
    average: str,
    mdmc_average: Optional[str],
) -> Tensor:

    if average == AverageMethod.MICRO and mdmc_average != MDMCAverageMethod.SAMPLEWISE:
        mask = tp >= 0
        precision = _safe_divide(tp[mask].sum().float(), (tp[mask] + fp[mask]).sum())
        recall = _safe_divide(tp[mask].sum().float(), (tp[mask] + fn[mask]).sum())
    else:
        precision = _safe_divide(tp.float(), tp + fp)
        recall = _safe_divide(tp.float(), tp + fn)

    num = (1 + beta**2) * precision * recall
    denom = beta**2 * precision + recall
    denom[denom == 0.] = 1  # avoid division by 0
    # if classes matter and a given class is not present in both the preds and the target,
    # computing the score for this class is meaningless, thus they should be ignored
    if average == AverageMethod.NONE and mdmc_average is None:
        # a class is not present if there exists no TPs, no FPs, and no FNs
        meaningless_indeces = torch.nonzero(np.logical_not(np.logical_or(tp, np.logical_or(fp, fn))))
        if ignore_index is None:
            ignore_index = meaningless_indeces
        else:
            ignore_index = torch.unique(torch.cat((meaningless_indeces, torch.tensor([[ignore_index]]))))

    if ignore_index is not None:
        if (
            average not in (AverageMethod.MICRO, AverageMethod.SAMPLES)
            and mdmc_average == MDMCAverageMethod.SAMPLEWISE  # noqa: W503
        ):
            num[..., ignore_index] = -1
            denom[..., ignore_index] = -1
        elif average not in (AverageMethod.MICRO, AverageMethod.SAMPLES):
            num[ignore_index, ...] = -1
            denom[ignore_index, ...] = -1

    return _reduce_stat_scores(
        numerator=num,
        denominator=denom,
        weights=None if average != AverageMethod.WEIGHTED else tp + fn,
        average=average,
        mdmc_average=mdmc_average,
    )


def fbeta(
    preds: Tensor,
    target: Tensor,
    beta: float = 1.0,
    average: str = "micro",
    mdmc_average: Optional[str] = None,
    ignore_index: Optional[int] = None,
    num_classes: Optional[int] = None,
    threshold: float = 0.5,
    top_k: Optional[int] = None,
    multiclass: Optional[bool] = None,
    multilabel: Optional[bool] = None,  # todo: deprecated, remove in v0.4
) -> Tensor:
    r"""
    Computes f_beta metric.

    .. math::
        F_{\beta} = (1 + \beta^2) * \frac{\text{precision} * \text{recall}}
        {(\beta^2 * \text{precision}) + \text{recall}}

    Works with binary, multiclass, and multilabel data.
    Accepts probabilities from a model output or integer class values in prediction.
    Works with multi-dimensional preds and target.

    If preds and target are the same shape and preds is a float tensor, we use the ``self.threshold`` argument
    to convert into integer labels. This is the case for binary and multi-label probabilities.

    If preds has an extra dimension as in the case of multi-class scores we perform an argmax on ``dim=1``.

    The reduction method (how the precision scores are aggregated) is controlled by the
    ``average`` parameter, and additionally by the ``mdmc_average`` parameter in the
    multi-dimensional multi-class case. Accepts all inputs listed in :ref:`references/modules:input types`.

    Args:
        preds: Predictions from model (probabilities or labels)
        target: Ground truth values
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

        ignore_index:
            Integer specifying a target class to ignore. If given, this class index does not contribute
            to the returned score, regardless of reduction method. If an index is ignored, and ``average=None``
            or ``'none'``, the score for the ignored class will be returned as ``nan``.
        num_classes:
            Number of classes. Necessary for ``'macro'``, ``'weighted'`` and ``None`` average methods.
        threshold:
            Threshold probability value for transforming probability predictions to binary
            (0,1) predictions, in the case of binary or multi-label inputs.
        top_k:
            Number of highest probability entries for each sample to convert to 1s - relevant
            only for inputs with probability predictions. If this parameter is set for multi-label
            inputs, it will take precedence over ``threshold``. For (multi-dim) multi-class inputs,
            this parameter defaults to 1. Should be left unset (``None``) for inputs with label predictions.
        multiclass:
            Used only in certain special cases, where you want to treat inputs as a different type
            than what they appear to be. See the parameter's
            :ref:`documentation section <references/modules:using the multiclass parameter>`
            for a more detailed explanation and examples.
        multilabel:
            .. deprecated:: 0.3
                Argument will not have any effect and will be removed in v0.4, please use ``multiclass`` intead.

    Return:
        The shape of the returned tensor depends on the ``average`` parameter

        - If ``average in ['micro', 'macro', 'weighted', 'samples']``, a one-element tensor will be returned
        - If ``average in ['none', None]``, the shape will be ``(C,)``, where ``C`` stands  for the number
          of classes

    Example:
        >>> from torchmetrics.functional import fbeta
        >>> target = torch.tensor([0, 1, 2, 0, 1, 2])
        >>> preds = torch.tensor([0, 2, 1, 0, 0, 1])
        >>> fbeta(preds, target, num_classes=3, beta=0.5)
        tensor(0.3333)

    """
    _deprecation_warn_arg_multilabel(multilabel)

    allowed_average = list(AverageMethod)
    if average not in allowed_average:
        raise ValueError(f"The `average` has to be one of {allowed_average}, got {average}.")

    allowed_mdmc_average = list(MDMCAverageMethod) + [None]
    if mdmc_average not in allowed_mdmc_average:
        raise ValueError(f"The `mdmc_average` has to be one of {allowed_mdmc_average}, got {mdmc_average}.")

    if average in [AverageMethod.MACRO, AverageMethod.WEIGHTED, AverageMethod.NONE] and (
        not num_classes or num_classes < 1
    ):
        raise ValueError(f"When you set `average` as {average}, you have to provide the number of classes.")

    if num_classes and ignore_index is not None and (not 0 <= ignore_index < num_classes or num_classes == 1):
        raise ValueError(f"The `ignore_index` {ignore_index} is not valid for inputs with {num_classes} classes")

    reduce = AverageMethod.MACRO if average in [AverageMethod.WEIGHTED, AverageMethod.NONE] else average
    tp, fp, tn, fn = _stat_scores_update(
        preds,
        target,
        reduce=reduce,
        mdmc_reduce=mdmc_average,
        threshold=threshold,
        num_classes=num_classes,
        top_k=top_k,
        multiclass=multiclass,
        ignore_index=ignore_index,
    )

    return _fbeta_compute(tp, fp, tn, fn, beta, ignore_index, average, mdmc_average)


def f1(
    preds: Tensor,
    target: Tensor,
    beta: float = 1.0,
    average: str = "micro",
    mdmc_average: Optional[str] = None,
    ignore_index: Optional[int] = None,
    num_classes: Optional[int] = None,
    threshold: float = 0.5,
    top_k: Optional[int] = None,
    multiclass: Optional[bool] = None,
    multilabel: Optional[bool] = None,  # todo: deprecated, remove in v0.4
) -> Tensor:
    """
    Computes F1 metric. F1 metrics correspond to a equally weighted average of the
    precision and recall scores.

    Works with binary, multiclass, and multilabel data.
    Accepts probabilities from a model output or integer class values in prediction.
    Works with multi-dimensional preds and target.

    If preds and target are the same shape and preds is a float tensor, we use the ``self.threshold`` argument
    to convert into integer labels. This is the case for binary and multi-label probabilities.

    If preds has an extra dimension as in the case of multi-class scores we perform an argmax on ``dim=1``.

    The reduction method (how the precision scores are aggregated) is controlled by the
    ``average`` parameter, and additionally by the ``mdmc_average`` parameter in the
    multi-dimensional multi-class case. Accepts all inputs listed in :ref:`references/modules:input types`.

    Args:
        preds: Predictions from model (probabilities or labels)
        target: Ground truth values
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

            .. note:: If ``'none'`` and a given class doesn't occur in the `preds` or `target`,
                the value for the class will be ``nan``.

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

        ignore_index:
            Integer specifying a target class to ignore. If given, this class index does not contribute
            to the returned score, regardless of reduction method. If an index is ignored, and ``average=None``
            or ``'none'``, the score for the ignored class will be returned as ``nan``.

        num_classes:
            Number of classes. Necessary for ``'macro'``, ``'weighted'`` and ``None`` average methods.

        threshold:
            Threshold probability value for transforming probability predictions to binary
            (0,1) predictions, in the case of binary or multi-label inputs.
        top_k:
            Number of highest probability entries for each sample to convert to 1s - relevant
            only for inputs with probability predictions. If this parameter is set for multi-label
            inputs, it will take precedence over ``threshold``. For (multi-dim) multi-class inputs,
            this parameter defaults to 1.

            Should be left unset (``None``) for inputs with label predictions.
        multiclass:
            Used only in certain special cases, where you want to treat inputs as a different type
            than what they appear to be. See the parameter's
            :ref:`documentation section <references/modules:using the multiclass parameter>`
            for a more detailed explanation and examples.
        multilabel:
            .. deprecated:: 0.3
                Argument will not have any effect and will be removed in v0.4, please use ``multiclass`` intead.

    Return:
        The shape of the returned tensor depends on the ``average`` parameter

        - If ``average in ['micro', 'macro', 'weighted', 'samples']``, a one-element tensor will be returned
        - If ``average in ['none', None]``, the shape will be ``(C,)``, where ``C`` stands  for the number
          of classes

    Example:
        >>> from torchmetrics.functional import f1
        >>> target = torch.tensor([0, 1, 2, 0, 1, 2])
        >>> preds = torch.tensor([0, 2, 1, 0, 0, 1])
        >>> f1(preds, target, num_classes=3)
        tensor(0.3333)
    """
    _deprecation_warn_arg_multilabel(multilabel)
    return fbeta(preds, target, 1.0, average, mdmc_average, ignore_index, num_classes, threshold, top_k, multiclass)
