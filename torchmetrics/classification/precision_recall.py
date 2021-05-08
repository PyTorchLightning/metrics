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
from typing import Any, Callable, Optional

import torch
from torch import Tensor

from torchmetrics.classification.stat_scores import StatScores
from torchmetrics.functional.classification.precision_recall import _precision_compute, _recall_compute
from torchmetrics.utilities import _deprecation_warn_arg_is_multiclass, _deprecation_warn_arg_multilabel


class Precision(StatScores):
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
        num_classes:
            Number of classes. Necessary for ``'macro'``, ``'weighted'`` and ``None`` average methods.
        threshold:
            Threshold for transforming probability or logit predictions to binary (0,1) predictions, in the case
            of binary or multi-label inputs. Default value of 0.5 corresponds to input being probabilities.
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

        compute_on_step:
            Forward only calls ``update()`` and return ``None`` if this is set to ``False``.
        dist_sync_on_step:
            Synchronize metric state across processes at each ``forward()``
            before returning the value at the step
        process_group:
            Specify the process group on which synchronization is called.
            default: ``None`` (which selects the entire world)
        dist_sync_fn:
            Callback that performs the allgather operation on the metric state. When ``None``, DDP
            will be used to perform the allgather.
        multilabel:
            .. deprecated:: 0.3
                Argument will not have any effect and will be removed in v0.4, please use ``multiclass`` intead.
        is_multiclass:
            .. deprecated:: 0.3
                Argument will not have any effect and will be removed in v0.4, please use ``multiclass`` intead.

    Raises:
        ValueError:
            If ``average`` is none of ``"micro"``, ``"macro"``, ``"weighted"``, ``"samples"``, ``"none"``, ``None``.

    Example:
        >>> from torchmetrics import Precision
        >>> preds  = torch.tensor([2, 0, 2, 1])
        >>> target = torch.tensor([1, 1, 2, 0])
        >>> precision = Precision(average='macro', num_classes=3)
        >>> precision(preds, target)
        tensor(0.1667)
        >>> precision = Precision(average='micro')
        >>> precision(preds, target)
        tensor(0.2500)

    """

    def __init__(
        self,
        num_classes: Optional[int] = None,
        threshold: float = 0.5,
        average: str = "micro",
        mdmc_average: Optional[str] = None,
        ignore_index: Optional[int] = None,
        top_k: Optional[int] = None,
        multiclass: Optional[bool] = None,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Callable = None,
        multilabel: Optional[bool] = None,  # todo: deprecated, remove in v0.4
        is_multiclass: Optional[bool] = None,  # todo: deprecated, remove in v0.4
    ):
        _deprecation_warn_arg_multilabel(multilabel)
        multiclass = _deprecation_warn_arg_is_multiclass(is_multiclass, multiclass)

        allowed_average = ["micro", "macro", "weighted", "samples", "none", None]
        if average not in allowed_average:
            raise ValueError(f"The `average` has to be one of {allowed_average}, got {average}.")

        super().__init__(
            reduce="macro" if average in ["weighted", "none", None] else average,
            mdmc_reduce=mdmc_average,
            threshold=threshold,
            top_k=top_k,
            num_classes=num_classes,
            multiclass=multiclass,
            ignore_index=ignore_index,
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )

        self.average = average

    def compute(self) -> Tensor:
        """
        Computes the precision score based on inputs passed in to ``update`` previously.

        Return:
            The shape of the returned tensor depends on the ``average`` parameter

            - If ``average in ['micro', 'macro', 'weighted', 'samples']``, a one-element tensor will be returned
            - If ``average in ['none', None]``, the shape will be ``(C,)``, where ``C`` stands  for the number
              of classes
        """
        tp, fp, tn, fn = self._get_final_stats()
        return _precision_compute(tp, fp, tn, fn, self.average, self.mdmc_reduce)


class Recall(StatScores):
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
        num_classes:
            Number of classes. Necessary for ``'macro'``, ``'weighted'`` and ``None`` average methods.
        threshold:
            Threshold for transforming probability or logit predictions to binary (0,1) predictions, in the case
            of binary or multi-label inputs. Default value of 0.5 corresponds to input being probabilities.
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

        top_k:
            Number of highest probability or logit score predictions considered to find the correct label,
            relevant only for (multi-dimensional) multi-class. The
            default value (``None``) will be interpreted as 1 for these inputs.

            Should be left at default (``None``) for all other types of inputs.

        multiclass:
            Used only in certain special cases, where you want to treat inputs as a different type
            than what they appear to be. See the parameter's
            :ref:`documentation section <references/modules:using the multiclass parameter>`
            for a more detailed explanation and examples.

        compute_on_step:
            Forward only calls ``update()`` and return ``None`` if this is set to ``False``.
        dist_sync_on_step:
            Synchronize metric state across processes at each ``forward()``
            before returning the value at the step
        process_group:
            Specify the process group on which synchronization is called.
            default: ``None`` (which selects the entire world)
        dist_sync_fn:
            Callback that performs the allgather operation on the metric state. When ``None``, DDP
            will be used to perform the allgather.
        multilabel:
            .. deprecated:: 0.3
                Argument will not have any effect and will be removed in v0.4, please use ``multiclass`` intead.
        is_multiclass:
            .. deprecated:: 0.3
                Argument will not have any effect and will be removed in v0.4, please use ``multiclass`` intead.

    Raises:
        ValueError:
            If ``average`` is none of ``"micro"``, ``"macro"``, ``"weighted"``, ``"samples"``, ``"none"``, ``None``.

    Example:
        >>> from torchmetrics import Recall
        >>> preds  = torch.tensor([2, 0, 2, 1])
        >>> target = torch.tensor([1, 1, 2, 0])
        >>> recall = Recall(average='macro', num_classes=3)
        >>> recall(preds, target)
        tensor(0.3333)
        >>> recall = Recall(average='micro')
        >>> recall(preds, target)
        tensor(0.2500)

    """

    def __init__(
        self,
        num_classes: Optional[int] = None,
        threshold: float = 0.5,
        average: str = "micro",
        mdmc_average: Optional[str] = None,
        ignore_index: Optional[int] = None,
        top_k: Optional[int] = None,
        multiclass: Optional[bool] = None,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Callable = None,
        multilabel: Optional[bool] = None,  # todo: deprecated, remove in v0.4
        is_multiclass: Optional[bool] = None,  # todo: deprecated, remove in v0.4
    ):
        _deprecation_warn_arg_multilabel(multilabel)
        multiclass = _deprecation_warn_arg_is_multiclass(is_multiclass, multiclass)

        allowed_average = ["micro", "macro", "weighted", "samples", "none", None]
        if average not in allowed_average:
            raise ValueError(f"The `average` has to be one of {allowed_average}, got {average}.")

        super().__init__(
            reduce="macro" if average in ["weighted", "none", None] else average,
            mdmc_reduce=mdmc_average,
            threshold=threshold,
            top_k=top_k,
            num_classes=num_classes,
            multiclass=multiclass,
            ignore_index=ignore_index,
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )

        self.average = average

    def compute(self) -> Tensor:
        """
        Computes the recall score based on inputs passed in to ``update`` previously.

        Return:
            The shape of the returned tensor depends on the ``average`` parameter

            - If ``average in ['micro', 'macro', 'weighted', 'samples']``, a one-element tensor will be returned
            - If ``average in ['none', None]``, the shape will be ``(C,)``, where ``C`` stands  for the number
              of classes
        """
        tp, fp, tn, fn = self._get_final_stats()
        return _recall_compute(tp, fp, tn, fn, self.average, self.mdmc_reduce)
