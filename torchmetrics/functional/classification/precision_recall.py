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

import torch
from torch import Tensor

from torchmetrics.classification.stat_scores import _reduce_stat_scores
from torchmetrics.functional.classification.stat_scores import _stat_scores_update
from torchmetrics.utilities import _deprecation_warn_arg_is_multiclass, _deprecation_warn_arg_multilabel


def _precision_compute(
    tp: Tensor,
    fp: Tensor,
    tn: Tensor,
    fn: Tensor,
    average: str,
    mdmc_average: Optional[str],
) -> Tensor:
    # todo: `tn` is unused
    return _reduce_stat_scores(
        numerator=tp,
        denominator=tp + fp,
        weights=None if average != "weighted" else tp + fn,
        average=average,
        mdmc_average=mdmc_average,
    )


def precision(
    preds: Tensor,
    target: Tensor,
    average: str = "micro",
    mdmc_average: Optional[str] = None,
    ignore_index: Optional[int] = None,
    num_classes: Optional[int] = None,
    threshold: float = 0.5,
    top_k: Optional[int] = None,
    multiclass: Optional[bool] = None,
    multilabel: Optional[bool] = None,  # todo: deprecated, remove in v0.4
    is_multiclass: Optional[bool] = None,  # todo: deprecated, remove in v0.4
) -> Tensor:
    r"""
    Computes `Precision <https://en.wikipedia.org/wiki/Precision_and_recall>`_:

    .. math:: \text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}

    Where :math:`\text{TP}` and :math:`\text{FP}` represent the number of true positives and
    false positives respecitively. With the use of ``top_k`` parameter, this metric can
    generalize to Precision@K.

    The reduction method (how the precision scores are aggregated) is controlled by the
    ``average`` parameter, and additionally by the ``mdmc_average`` parameter in the
    multi-dimensional multi-class case. Accepts all inputs listed in :ref:`references/modules:input types`.

    Args:
        preds: Predictions from model (probabilities, logits or labels)
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
        multilabel:
            .. deprecated:: 0.3
                Argument will not have any effect and will be removed in v0.4, please use ``multiclass`` intead.
        is_multiclass:
            .. deprecated:: 0.3
                Argument will not have any effect and will be removed in v0.4, please use ``multiclass`` intead.

    Return:
        The shape of the returned tensor depends on the ``average`` parameter

        - If ``average in ['micro', 'macro', 'weighted', 'samples']``, a one-element tensor will be returned
        - If ``average in ['none', None]``, the shape will be ``(C,)``, where ``C`` stands  for the number
          of classes

    Raises:
        ValueError:
            If ``average`` is not one of ``"micro"``, ``"macro"``, ``"weighted"``,
            ``"samples"``, ``"none"`` or ``None``.
        ValueError:
            If ``mdmc_average`` is not one of ``None``, ``"samplewise"``, ``"global"``.
        ValueError:
            If ``average`` is set but ``num_classes`` is not provided.
        ValueError:
            If ``num_classes`` is set
            and ``ignore_index`` is not in the range ``[0, num_classes)``.

    Example:
        >>> from torchmetrics.functional import precision
        >>> preds  = torch.tensor([2, 0, 2, 1])
        >>> target = torch.tensor([1, 1, 2, 0])
        >>> precision(preds, target, average='macro', num_classes=3)
        tensor(0.1667)
        >>> precision(preds, target, average='micro')
        tensor(0.2500)

    """
    _deprecation_warn_arg_multilabel(multilabel)
    multiclass = _deprecation_warn_arg_is_multiclass(is_multiclass, multiclass)

    allowed_average = ["micro", "macro", "weighted", "samples", "none", None]
    if average not in allowed_average:
        raise ValueError(f"The `average` has to be one of {allowed_average}, got {average}.")

    allowed_mdmc_average = [None, "samplewise", "global"]
    if mdmc_average not in allowed_mdmc_average:
        raise ValueError(f"The `mdmc_average` has to be one of {allowed_mdmc_average}, got {mdmc_average}.")

    if average in ["macro", "weighted", "none", None] and (not num_classes or num_classes < 1):
        raise ValueError(f"When you set `average` as {average}, you have to provide the number of classes.")

    if num_classes and ignore_index is not None and (not 0 <= ignore_index < num_classes or num_classes == 1):
        raise ValueError(f"The `ignore_index` {ignore_index} is not valid for inputs with {num_classes} classes")

    reduce = "macro" if average in ["weighted", "none", None] else average
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

    return _precision_compute(tp, fp, tn, fn, average, mdmc_average)


def _recall_compute(
    tp: Tensor,
    fp: Tensor,
    tn: Tensor,
    fn: Tensor,
    average: str,
    mdmc_average: Optional[str],
) -> Tensor:
    # todo: `tp` is unused
    # todo: `tn` is unused
    return _reduce_stat_scores(
        numerator=tp,
        denominator=tp + fn,
        weights=None if average != "weighted" else tp + fn,
        average=average,
        mdmc_average=mdmc_average,
    )


def recall(
    preds: Tensor,
    target: Tensor,
    average: str = "micro",
    mdmc_average: Optional[str] = None,
    ignore_index: Optional[int] = None,
    num_classes: Optional[int] = None,
    threshold: float = 0.5,
    top_k: Optional[int] = None,
    multiclass: Optional[bool] = None,
    multilabel: Optional[bool] = None,  # todo: deprecated, remove in v0.4
    is_multiclass: Optional[bool] = None,  # todo: deprecated, remove in v0.4
) -> Tensor:
    r"""
    Computes `Recall <https://en.wikipedia.org/wiki/Precision_and_recall>`_:

    .. math:: \text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}

    Where :math:`\text{TP}` and :math:`\text{FN}` represent the number of true positives and
    false negatives respecitively. With the use of ``top_k`` parameter, this metric can
    generalize to Recall@K.

    The reduction method (how the recall scores are aggregated) is controlled by the
    ``average`` parameter, and additionally by the ``mdmc_average`` parameter in the
    multi-dimensional multi-class case. Accepts all inputs listed in :ref:`references/modules:input types`.

    Args:
        preds: Predictions from model (probabilities, logits or labels)
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
        multilabel:
            .. deprecated:: 0.3
                Argument will not have any effect and will be removed in v0.4, please use ``multiclass`` intead.
        is_multiclass:
            .. deprecated:: 0.3
                Argument will not have any effect and will be removed in v0.4, please use ``multiclass`` intead.

    Return:
        The shape of the returned tensor depends on the ``average`` parameter

        - If ``average in ['micro', 'macro', 'weighted', 'samples']``, a one-element tensor will be returned
        - If ``average in ['none', None]``, the shape will be ``(C,)``, where ``C`` stands  for the number
          of classes

    Raises:
        ValueError:
            If ``average`` is not one of ``"micro"``, ``"macro"``, ``"weighted"``,
            ``"samples"``, ``"none"`` or ``None``.
        ValueError:
            If ``mdmc_average`` is not one of ``None``, ``"samplewise"``, ``"global"``.
        ValueError:
            If ``average`` is set but ``num_classes`` is not provided.
        ValueError:
            If ``num_classes`` is set
            and ``ignore_index`` is not in the range ``[0, num_classes)``.

    Example:
        >>> from torchmetrics.functional import recall
        >>> preds  = torch.tensor([2, 0, 2, 1])
        >>> target = torch.tensor([1, 1, 2, 0])
        >>> recall(preds, target, average='macro', num_classes=3)
        tensor(0.3333)
        >>> recall(preds, target, average='micro')
        tensor(0.2500)

    """
    _deprecation_warn_arg_multilabel(multilabel)
    multiclass = _deprecation_warn_arg_is_multiclass(is_multiclass, multiclass)

    allowed_average = ["micro", "macro", "weighted", "samples", "none", None]
    if average not in allowed_average:
        raise ValueError(f"The `average` has to be one of {allowed_average}, got {average}.")

    allowed_mdmc_average = [None, "samplewise", "global"]
    if mdmc_average not in allowed_mdmc_average:
        raise ValueError("The `mdmc_average` has to be one of {allowed_mdmc_average}, got {mdmc_average}.")

    if average in ["macro", "weighted", "none", None] and (not num_classes or num_classes < 1):
        raise ValueError(f"When you set `average` as {average}, you have to provide the number of classes.")

    if num_classes and ignore_index is not None and (not 0 <= ignore_index < num_classes or num_classes == 1):
        raise ValueError(f"The `ignore_index` {ignore_index} is not valid for inputs with {num_classes} classes")

    reduce = "macro" if average in ["weighted", "none", None] else average
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

    return _recall_compute(tp, fp, tn, fn, average, mdmc_average)


def precision_recall(
    preds: Tensor,
    target: Tensor,
    average: str = "micro",
    mdmc_average: Optional[str] = None,
    ignore_index: Optional[int] = None,
    num_classes: Optional[int] = None,
    threshold: float = 0.5,
    top_k: Optional[int] = None,
    multiclass: Optional[bool] = None,
    multilabel: Optional[bool] = None,  # todo: deprecated, remove in v0.4
    is_multiclass: Optional[bool] = None,  # todo: deprecated, remove in v0.4
) -> Tuple[Tensor, Tensor]:
    r"""
    Computes `Precision and Recall <https://en.wikipedia.org/wiki/Precision_and_recall>`_:

    .. math:: \text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}


    .. math:: \text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}

    Where :math:`\text{TP}`m :math:`\text{FN}` and :math:`\text{FP}` represent the number
    of true positives, false negatives and false positives respecitively. With the use of
    ``top_k`` parameter, this metric can generalize to Recall@K and Precision@K.

    The reduction method (how the recall scores are aggregated) is controlled by the
    ``average`` parameter, and additionally by the ``mdmc_average`` parameter in the
    multi-dimensional multi-class case. Accepts all inputs listed in :ref:`references/modules:input types`.

    Args:
        preds: Predictions from model (probabilities, logits or labels)
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
        multilabel:
            .. deprecated:: 0.3
                Argument will not have any effect and will be removed in v0.4, please use ``multiclass`` intead.
        is_multiclass:
            .. deprecated:: 0.3
                Argument will not have any effect and will be removed in v0.4, please use ``multiclass`` intead.

    Return:
        The function returns a tuple with two elements: precision and recall. Their shape
        depends on the ``average`` parameter

        - If ``average in ['micro', 'macro', 'weighted', 'samples']``, they are a single element tensor
        - If ``average in ['none', None]``, they are a tensor of shape ``(C, )``, where ``C`` stands for
          the number of classes

    Raises:
        ValueError:
            If ``average`` is not one of ``"micro"``, ``"macro"``, ``"weighted"``,
            ``"samples"``, ``"none"`` or ``None``.
        ValueError:
            If ``mdmc_average`` is not one of ``None``, ``"samplewise"``, ``"global"``.
        ValueError:
            If ``average`` is set but ``num_classes`` is not provided.
        ValueError:
            If ``num_classes`` is set
            and ``ignore_index`` is not in the range ``[0, num_classes)``.

    Example:
        >>> from torchmetrics.functional import precision_recall
        >>> preds  = torch.tensor([2, 0, 2, 1])
        >>> target = torch.tensor([1, 1, 2, 0])
        >>> precision_recall(preds, target, average='macro', num_classes=3)
        (tensor(0.1667), tensor(0.3333))
        >>> precision_recall(preds, target, average='micro')
        (tensor(0.2500), tensor(0.2500))

    """
    _deprecation_warn_arg_multilabel(multilabel)
    multiclass = _deprecation_warn_arg_is_multiclass(is_multiclass, multiclass)

    allowed_average = ["micro", "macro", "weighted", "samples", "none", None]
    if average not in allowed_average:
        raise ValueError(f"The `average` has to be one of {allowed_average}, got {average}.")

    allowed_mdmc_average = [None, "samplewise", "global"]
    if mdmc_average not in allowed_mdmc_average:
        raise ValueError("The `mdmc_average` has to be one of {allowed_mdmc_average}, got {mdmc_average}.")

    if average in ["macro", "weighted", "none", None] and (not num_classes or num_classes < 1):
        raise ValueError(f"When you set `average` as {average}, you have to provide the number of classes.")

    if num_classes and ignore_index is not None and (not 0 <= ignore_index < num_classes or num_classes == 1):
        raise ValueError(f"The `ignore_index` {ignore_index} is not valid for inputs with {num_classes} classes")

    reduce = "macro" if average in ["weighted", "none", None] else average
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

    precision = _precision_compute(tp, fp, tn, fn, average, mdmc_average)
    recall = _recall_compute(tp, fp, tn, fn, average, mdmc_average)

    return precision, recall
