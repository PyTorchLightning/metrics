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

from torch import Tensor, tensor

from torchmetrics.functional.classification.accuracy import (
    _accuracy_compute,
    _accuracy_update,
    _check_subset_validity,
    _mode,
    _subset_accuracy_compute,
    _subset_accuracy_update,
)

from torchmetrics.classification.stat_scores import StatScores  # isort:skip


class Accuracy(StatScores):
    r"""
    Computes `Accuracy <https://en.wikipedia.org/wiki/Accuracy_and_precision>`__:

    .. math::
        \text{Accuracy} = \frac{1}{N}\sum_i^N 1(y_i = \hat{y}_i)

    Where :math:`y` is a tensor of target values, and :math:`\hat{y}` is a
    tensor of predictions.

    For multi-class and multi-dimensional multi-class data with probability or logits predictions, the
    parameter ``top_k`` generalizes this metric to a Top-K accuracy metric: for each sample the
    top-K highest probability or logit score items are considered to find the correct label.

    For multi-label and multi-dimensional multi-class inputs, this metric computes the "global"
    accuracy by default, which counts all labels or sub-samples separately. This can be
    changed to subset accuracy (which requires all labels or sub-samples in the sample to
    be correctly predicted) by setting ``subset_accuracy=True``.

    Accepts all input types listed in :ref:`references/modules:input types`.

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
            will be used to perform the allgather

    Raises:
        ValueError:
            If ``threshold`` is not between ``0`` and ``1``.
        ValueError:
            If ``top_k`` is not an ``integer`` larger than ``0``.
        ValueError:
            If ``average`` is none of ``"micro"``, ``"macro"``, ``"weighted"``, ``"samples"``, ``"none"``, ``None``.
        ValueError:
            If two different input modes are provided, eg. using ``mult-label`` with ``multi-class``.
        ValueError:
            If ``top_k`` parameter is set for ``multi-label`` inputs.

    Example:
        >>> import torch
        >>> from torchmetrics import Accuracy
        >>> target = torch.tensor([0, 1, 2, 3])
        >>> preds = torch.tensor([0, 2, 1, 3])
        >>> accuracy = Accuracy()
        >>> accuracy(preds, target)
        tensor(0.5000)

        >>> target = torch.tensor([0, 1, 2])
        >>> preds = torch.tensor([[0.1, 0.9, 0], [0.3, 0.1, 0.6], [0.2, 0.5, 0.3]])
        >>> accuracy = Accuracy(top_k=2)
        >>> accuracy(preds, target)
        tensor(0.6667)

    """

    def __init__(
        self,
        threshold: float = 0.5,
        num_classes: Optional[int] = None,
        average: str = "micro",
        mdmc_average: Optional[str] = "global",
        ignore_index: Optional[int] = None,
        top_k: Optional[int] = None,
        multiclass: Optional[bool] = None,
        subset_accuracy: bool = False,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Callable = None,
    ):
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

        self.add_state("correct", default=tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=tensor(0), dist_reduce_fx="sum")

        if not 0 < threshold < 1:
            raise ValueError(f"The `threshold` should be a float in the (0,1) interval, got {threshold}")

        if top_k is not None and (not isinstance(top_k, int) or top_k <= 0):
            raise ValueError(f"The `top_k` should be an integer larger than 0, got {top_k}")

        self.average = average
        self.threshold = threshold
        self.top_k = top_k
        self.subset_accuracy = subset_accuracy
        self.mode = None
        self.multiclass = multiclass

    def update(self, preds: Tensor, target: Tensor):
        """
        Update state with predictions and targets. See :ref:`references/modules:input types` for more information
        on input types.

        Args:
            preds: Predictions from model (logits, probabilities, or labels)
            target: Ground truth labels
        """
        """ returns the mode of the data (binary, multi label, multi class, multi-dim multi class) """
        mode = _mode(preds, target, self.threshold, self.top_k, self.num_classes, self.multiclass)

        if self.mode is None:
            self.mode = mode
        elif self.mode != mode:
            raise ValueError("You can not use {} inputs with {} inputs.".format(mode, self.mode))

        if self.subset_accuracy and not _check_subset_validity(self.mode):
            self.subset_accuracy = False

        if self.subset_accuracy:
            correct, total = _subset_accuracy_update(preds, target, threshold=self.threshold, top_k=self.top_k)
            self.correct += correct
            self.total += total
        else:
            tp, fp, tn, fn = _accuracy_update(
                preds,
                target,
                reduce=self.reduce,
                mdmc_reduce=self.mdmc_reduce,
                threshold=self.threshold,
                num_classes=self.num_classes,
                top_k=self.top_k,
                multiclass=self.multiclass,
                ignore_index=self.ignore_index,
                mode=self.mode,
            )

            # Update states
            if self.reduce != "samples" and self.mdmc_reduce != "samplewise":
                self.tp += tp
                self.fp += fp
                self.tn += tn
                self.fn += fn
            else:
                self.tp.append(tp)
                self.fp.append(fp)
                self.tn.append(tn)
                self.fn.append(fn)

    def compute(self) -> Tensor:
        """
        Computes accuracy based on inputs passed in to ``update`` previously.
        """
        if self.subset_accuracy:
            return _subset_accuracy_compute(self.correct, self.total)
        else:
            tp, fp, tn, fn = self._get_final_stats()
            return _accuracy_compute(tp, fp, tn, fn, self.average, self.mdmc_reduce, self.mode)

    @property
    def is_differentiable(self):
        return False
