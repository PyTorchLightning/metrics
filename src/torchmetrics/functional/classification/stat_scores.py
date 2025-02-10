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
from typing import Optional, Union

import torch
from torch import Tensor, tensor
from typing_extensions import Literal

from torchmetrics.utilities.checks import _check_same_shape, _input_format_classification
from torchmetrics.utilities.compute import normalize_logits_if_needed
from torchmetrics.utilities.data import _bincount, select_topk
from torchmetrics.utilities.enums import AverageMethod, ClassificationTask, DataType, MDMCAverageMethod


def _binary_stat_scores_arg_validation(
    threshold: float = 0.5,
    multidim_average: Literal["global", "samplewise"] = "global",
    ignore_index: Optional[int] = None,
    zero_division: float = 0,
) -> None:
    """Validate non tensor input.

    - ``threshold`` has to be a float in the [0,1] range
    - ``multidim_average`` has to be either "global" or "samplewise"
    - ``ignore_index`` has to be None or int
    - ``zero_division`` has to be 0 or 1

    """
    if not (isinstance(threshold, float) and (0 <= threshold <= 1)):
        raise ValueError(f"Expected argument `threshold` to be a float in the [0,1] range, but got {threshold}.")
    allowed_multidim_average = ("global", "samplewise")
    if multidim_average not in allowed_multidim_average:
        raise ValueError(
            f"Expected argument `multidim_average` to be one of {allowed_multidim_average}, but got {multidim_average}"
        )
    if ignore_index is not None and not isinstance(ignore_index, int):
        raise ValueError(f"Expected argument `ignore_index` to either be `None` or an integer, but got {ignore_index}")
    if zero_division not in [0, 1]:
        raise ValueError(f"Expected argument `zero_division` to be 0 or 1, but got {zero_division}.")


def _binary_stat_scores_tensor_validation(
    preds: Tensor,
    target: Tensor,
    multidim_average: Literal["global", "samplewise"] = "global",
    ignore_index: Optional[int] = None,
) -> None:
    """Validate tensor input.

    - tensors have to be of same shape
    - all values in target tensor that are not ignored have to be in {0, 1}
    - if pred tensor is not floating point, then all values also have to be in {0, 1}
    - if ``multidim_average`` is set to ``samplewise`` preds tensor needs to be at least 2 dimensional

    """
    # Check that they have same shape
    _check_same_shape(preds, target)

    # Check that target only contains [0,1] values or value in ignore_index
    unique_values = torch.unique(target, dim=None)
    if ignore_index is None:
        check = torch.any((unique_values != 0) & (unique_values != 1))
    else:
        check = torch.any((unique_values != 0) & (unique_values != 1) & (unique_values != ignore_index))
    if check:
        raise RuntimeError(
            f"Detected the following values in `target`: {unique_values} but expected only"
            f" the following values {[0, 1] if ignore_index is None else [ignore_index]}."
        )

    # If preds is label tensor, also check that it only contains [0,1] values
    if not preds.is_floating_point():
        unique_values = torch.unique(preds, dim=None)
        if torch.any((unique_values != 0) & (unique_values != 1)):
            raise RuntimeError(
                f"Detected the following values in `preds`: {unique_values} but expected only"
                " the following values [0,1] since `preds` is a label tensor."
            )

    if multidim_average != "global" and preds.ndim < 2:
        raise ValueError("Expected input to be at least 2D when multidim_average is set to `samplewise`")


def _binary_stat_scores_format(
    preds: Tensor,
    target: Tensor,
    threshold: float = 0.5,
    ignore_index: Optional[int] = None,
) -> tuple[Tensor, Tensor]:
    """Convert all input to label format.

    - If preds tensor is floating point, applies sigmoid if pred tensor not in [0,1] range
    - If preds tensor is floating point, thresholds afterwards
    - Mask all datapoints that should be ignored with negative values

    """
    if preds.is_floating_point():
        preds = normalize_logits_if_needed(preds, "sigmoid")
        preds = preds > threshold

    preds = preds.reshape(preds.shape[0], -1)
    target = target.reshape(target.shape[0], -1)

    if ignore_index is not None:
        idx = target == ignore_index
        target = target.clone()
        target[idx] = -1

    return preds, target


def _binary_stat_scores_update(
    preds: Tensor,
    target: Tensor,
    multidim_average: Literal["global", "samplewise"] = "global",
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Compute the statistics."""
    sum_dim = [0, 1] if multidim_average == "global" else [1]
    tp = ((target == preds) & (target == 1)).sum(sum_dim).squeeze()
    fn = ((target != preds) & (target == 1)).sum(sum_dim).squeeze()
    fp = ((target != preds) & (target == 0)).sum(sum_dim).squeeze()
    tn = ((target == preds) & (target == 0)).sum(sum_dim).squeeze()
    return tp, fp, tn, fn


def _binary_stat_scores_compute(
    tp: Tensor, fp: Tensor, tn: Tensor, fn: Tensor, multidim_average: Literal["global", "samplewise"] = "global"
) -> Tensor:
    """Stack statistics and compute support also."""
    return torch.stack([tp, fp, tn, fn, tp + fn], dim=0 if multidim_average == "global" else 1).squeeze()


def binary_stat_scores(
    preds: Tensor,
    target: Tensor,
    threshold: float = 0.5,
    multidim_average: Literal["global", "samplewise"] = "global",
    ignore_index: Optional[int] = None,
    validate_args: bool = True,
) -> Tensor:
    r"""Compute the true positives, false positives, true negatives, false negatives, support for binary tasks.

    Related to `Type I and Type II errors`_.

    Accepts the following input tensors:

    - ``preds`` (int or float tensor): ``(N, ...)``. If preds is a floating point tensor with values outside
      [0,1] range we consider the input to be logits and will auto apply sigmoid per element. Additionally,
      we convert to int tensor with thresholding using the value in ``threshold``.
    - ``target`` (int tensor): ``(N, ...)``

    Args:
        preds: Tensor with predictions
        target: Tensor with true labels
        threshold: Threshold for transforming probability to binary {0,1} predictions
        multidim_average:
            Defines how additionally dimensions ``...`` should be handled. Should be one of the following:

            - ``global``: Additional dimensions are flatted along the batch dimension
            - ``samplewise``: Statistic will be calculated independently for each sample on the ``N`` axis.
              The statistics in this case are calculated over the additional dimensions.

        ignore_index:
            Specifies a target value that is ignored and does not contribute to the metric calculation
        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.

    Returns:
        The metric returns a tensor of shape ``(..., 5)``, where the last dimension corresponds
        to ``[tp, fp, tn, fn, sup]`` (``sup`` stands for support and equals ``tp + fn``). The shape
        depends on the ``multidim_average`` parameter:

        - If ``multidim_average`` is set to ``global``, the shape will be ``(5,)``
        - If ``multidim_average`` is set to ``samplewise``, the shape will be ``(N, 5)``

    Example (preds is int tensor):
        >>> from torch import tensor
        >>> from torchmetrics.functional.classification import binary_stat_scores
        >>> target = tensor([0, 1, 0, 1, 0, 1])
        >>> preds = tensor([0, 0, 1, 1, 0, 1])
        >>> binary_stat_scores(preds, target)
        tensor([2, 1, 2, 1, 3])

    Example (preds is float tensor):
        >>> from torchmetrics.functional.classification import binary_stat_scores
        >>> target = tensor([0, 1, 0, 1, 0, 1])
        >>> preds = tensor([0.11, 0.22, 0.84, 0.73, 0.33, 0.92])
        >>> binary_stat_scores(preds, target)
        tensor([2, 1, 2, 1, 3])

    Example (multidim tensors):
        >>> from torchmetrics.functional.classification import binary_stat_scores
        >>> target = tensor([[[0, 1], [1, 0], [0, 1]], [[1, 1], [0, 0], [1, 0]]])
        >>> preds = tensor([[[0.59, 0.91], [0.91, 0.99], [0.63, 0.04]],
        ...                 [[0.38, 0.04], [0.86, 0.780], [0.45, 0.37]]])
        >>> binary_stat_scores(preds, target, multidim_average='samplewise')
        tensor([[2, 3, 0, 1, 3],
                [0, 2, 1, 3, 3]])

    """
    if validate_args:
        _binary_stat_scores_arg_validation(threshold, multidim_average, ignore_index)
        _binary_stat_scores_tensor_validation(preds, target, multidim_average, ignore_index)
    preds, target = _binary_stat_scores_format(preds, target, threshold, ignore_index)
    tp, fp, tn, fn = _binary_stat_scores_update(preds, target, multidim_average)
    return _binary_stat_scores_compute(tp, fp, tn, fn, multidim_average)


def _multiclass_stat_scores_arg_validation(
    num_classes: Optional[int],
    top_k: int = 1,
    average: Optional[Literal["micro", "macro", "weighted", "none"]] = "micro",
    multidim_average: Literal["global", "samplewise"] = "global",
    ignore_index: Optional[int] = None,
    zero_division: float = 0,
) -> None:
    """Validate non tensor input.

    - ``num_classes`` has to be a int larger than 1
    - ``top_k`` has to be an int larger than 0 but no larger than number of classes
    - ``average`` has to be "micro" | "macro" | "weighted" | "none"
    - ``multidim_average`` has to be either "global" or "samplewise"
    - ``ignore_index`` has to be None or int
    - ``zero_division`` has to be 0 or 1

    """
    if num_classes is None and average != "micro":
        raise ValueError(
            f"Argument `num_classes` can only be `None` for `average='micro'`, but got `average={average}`."
        )
    if num_classes is not None and (not isinstance(num_classes, int) or num_classes < 2):
        raise ValueError(f"Expected argument `num_classes` to be an integer larger than 1, but got {num_classes}")
    if not isinstance(top_k, int) and top_k < 1:
        raise ValueError(f"Expected argument `top_k` to be an integer larger than or equal to 1, but got {top_k}")
    if top_k > (num_classes if num_classes is not None else 1):
        raise ValueError(
            f"Expected argument `top_k` to be smaller or equal to `num_classes` but got {top_k} and {num_classes}"
        )
    allowed_average = ("micro", "macro", "weighted", "none", None)
    if average not in allowed_average:
        raise ValueError(f"Expected argument `average` to be one of {allowed_average}, but got {average}")
    allowed_multidim_average = ("global", "samplewise")
    if multidim_average not in allowed_multidim_average:
        raise ValueError(
            f"Expected argument `multidim_average` to be one of {allowed_multidim_average}, but got {multidim_average}"
        )
    if ignore_index is not None and not isinstance(ignore_index, int):
        raise ValueError(f"Expected argument `ignore_index` to either be `None` or an integer, but got {ignore_index}")
    if zero_division not in [0, 1]:
        raise ValueError(f"Expected argument `zero_division` to be 0 or 1, but got {zero_division}.")


def _multiclass_stat_scores_tensor_validation(
    preds: Tensor,
    target: Tensor,
    num_classes: Optional[int],
    multidim_average: Literal["global", "samplewise"] = "global",
    ignore_index: Optional[int] = None,
) -> None:
    """Validate tensor input.

    - if preds has one more dimension than target, then all dimensions except for preds.shape[1] should match
    exactly. preds.shape[1] should have size equal to number of classes
    - if preds and target have same number of dims, then all dimensions should match
    - if ``multidim_average`` is set to ``samplewise`` preds tensor needs to be at least 2 dimensional in the
    int case and 3 dimensional in the float case
    - all values in target tensor that are not ignored have to be {0, ..., num_classes - 1}
    - if pred tensor is not floating point, then all values also have to be in {0, ..., num_classes - 1}

    """
    if preds.ndim == target.ndim + 1:
        if not preds.is_floating_point():
            raise ValueError("If `preds` have one dimension more than `target`, `preds` should be a float tensor.")
        if num_classes is not None and preds.shape[1] != num_classes:
            raise ValueError(
                "If `preds` have one dimension more than `target`, `preds.shape[1]` should be"
                " equal to number of classes."
            )
        if preds.shape[2:] != target.shape[1:]:
            raise ValueError(
                "If `preds` have one dimension more than `target`, the shape of `preds` should be"
                " (N, C, ...), and the shape of `target` should be (N, ...)."
            )
        if multidim_average != "global" and preds.ndim < 3:
            raise ValueError(
                "If `preds` have one dimension more than `target`, the shape of `preds` should be"
                " at least 3D when multidim_average is set to `samplewise`"
            )

    elif preds.ndim == target.ndim:
        if preds.shape != target.shape:
            raise ValueError(
                "The `preds` and `target` should have the same shape,"
                f" got `preds` with shape={preds.shape} and `target` with shape={target.shape}."
            )
        if multidim_average != "global" and preds.ndim < 2:
            raise ValueError(
                "When `preds` and `target` have the same shape, the shape of `preds` should be"
                " at least 2D when multidim_average is set to `samplewise`"
            )
    else:
        raise ValueError(
            "Either `preds` and `target` both should have the (same) shape (N, ...), or `target` should be (N, ...)"
            " and `preds` should be (N, C, ...)."
        )
    if num_classes is not None:
        check_value = num_classes if ignore_index is None else num_classes + 1
        for t, name in ((target, "target"),) + ((preds, "preds"),) if not preds.is_floating_point() else ():  # noqa: RUF005
            num_unique_values = len(torch.unique(t, dim=None))
            if num_unique_values > check_value:
                raise RuntimeError(
                    f"Detected more unique values in `{name}` than expected. Expected only {check_value} but found"
                    f" {num_unique_values} in `{name}`. Found values: {torch.unique(t, dim=None)}."
                )


def _multiclass_stat_scores_format(
    preds: Tensor,
    target: Tensor,
    top_k: int = 1,
) -> tuple[Tensor, Tensor]:
    """Convert all input to label format except if ``top_k`` is not 1.

    - Applies argmax if preds have one more dimension than target
    - Flattens additional dimensions

    """
    # Apply argmax if we have one more dimension
    if preds.ndim == target.ndim + 1 and top_k == 1:
        preds = preds.argmax(dim=1)
    preds = preds.reshape(*preds.shape[:2], -1) if top_k != 1 else preds.reshape(preds.shape[0], -1)
    target = target.reshape(target.shape[0], -1)
    return preds, target


def _refine_preds_oh(preds: Tensor, preds_oh: Tensor, target: Tensor, top_k: int) -> Tensor:
    """Refines prediction one-hot encodings by replacing entries with target one-hot when there's an intersection.

    When no intersection is found between the top-k predictions and target, uses the top-1 prediction.

    Args:
        preds: Original prediction tensor with probabilities/logits
        preds_oh: Current one-hot encoded predictions from top-k selection
        target: Target tensor with class indices
        top_k: Number of top predictions to consider

    Returns:
        Refined one-hot encoded predictions tensor

    """
    preds = preds.squeeze()
    target = target.squeeze()
    top_k_indices = torch.topk(preds, k=top_k, dim=1).indices
    top_1_indices = top_k_indices[:, 0]
    target_in_topk = torch.any(top_k_indices == target.unsqueeze(1), dim=1)
    result = torch.where(target_in_topk, target, top_1_indices)
    return torch.zeros_like(preds_oh, dtype=torch.int32).scatter_(-1, result.unsqueeze(1).unsqueeze(1), 1)


def _multiclass_stat_scores_update(
    preds: Tensor,
    target: Tensor,
    num_classes: int,
    top_k: int = 1,
    average: Optional[Literal["micro", "macro", "weighted", "none"]] = "micro",
    multidim_average: Literal["global", "samplewise"] = "global",
    ignore_index: Optional[int] = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Compute the statistics.

    - If ``multidim_average`` is equal to samplewise or ``top_k`` is not 1, we transform both preds and
    target into one hot format.
    - Else we calculate statistics by first calculating the confusion matrix and afterwards deriving the
    statistics from that
    - Remove all datapoints that should be ignored. Depending on if ``ignore_index`` is in the set of labels
    or outside we have do use different augmentation strategies when one hot encoding.

    """
    if multidim_average == "samplewise" or top_k != 1:
        ignore_in = 0 <= ignore_index <= num_classes - 1 if ignore_index is not None else None
        if ignore_index is not None and not ignore_in:
            preds = preds.clone()
            target = target.clone()
            idx = target == ignore_index
            target[idx] = num_classes
            idx = idx.unsqueeze(1).repeat(1, num_classes, 1) if preds.ndim > target.ndim else idx
            preds[idx] = num_classes

        if top_k > 1:
            preds_oh = torch.movedim(select_topk(preds, topk=top_k, dim=1), 1, -1)
            preds_oh = _refine_preds_oh(preds, preds_oh, target, top_k)
        else:
            preds_oh = torch.nn.functional.one_hot(
                preds.long(), num_classes + 1 if ignore_index is not None and not ignore_in else num_classes
            )

        target_oh = torch.nn.functional.one_hot(
            target.long(), num_classes + 1 if ignore_index is not None and not ignore_in else num_classes
        )

        if ignore_index is not None:
            if 0 <= ignore_index <= num_classes - 1:
                target_oh[target == ignore_index, :] = -1
            else:
                preds_oh = preds_oh[..., :-1] if top_k == 1 else preds_oh
                target_oh = target_oh[..., :-1]
                target_oh[target == num_classes, :] = -1
        sum_dim = [0, 1] if multidim_average == "global" else [1]
        tp = ((target_oh == preds_oh) & (target_oh == 1)).sum(sum_dim)
        fn = ((target_oh != preds_oh) & (target_oh == 1)).sum(sum_dim)
        fp = ((target_oh != preds_oh) & (target_oh == 0)).sum(sum_dim)
        tn = ((target_oh == preds_oh) & (target_oh == 0)).sum(sum_dim)
    elif average == "micro":
        preds = preds.flatten()
        target = target.flatten()
        if ignore_index is not None:
            idx = target != ignore_index
            preds = preds[idx]
            target = target[idx]
        tp = (preds == target).sum()
        fp = (preds != target).sum()
        fn = (preds != target).sum()
        tn = num_classes * preds.numel() - (fp + fn + tp)
    else:
        preds = preds.flatten()
        target = target.flatten()
        if ignore_index is not None:
            idx = target != ignore_index
            preds = preds[idx]
            target = target[idx]
        unique_mapping = target.to(torch.long) * num_classes + preds.to(torch.long)
        bins = _bincount(unique_mapping, minlength=num_classes**2)
        confmat = bins.reshape(num_classes, num_classes)
        tp = confmat.diag()
        fp = confmat.sum(0) - tp
        fn = confmat.sum(1) - tp
        tn = confmat.sum() - (fp + fn + tp)
    return tp, fp, tn, fn


def _multiclass_stat_scores_compute(
    tp: Tensor,
    fp: Tensor,
    tn: Tensor,
    fn: Tensor,
    average: Optional[Literal["micro", "macro", "weighted", "none"]] = "micro",
    multidim_average: Literal["global", "samplewise"] = "global",
) -> Tensor:
    """Stack statistics and compute support also.

    Applies average strategy afterwards.

    """
    res = torch.stack([tp, fp, tn, fn, tp + fn], dim=-1)
    sum_dim = 0 if multidim_average == "global" else 1
    if average == "micro":
        return res.sum(sum_dim) if res.ndim > 1 else res
    if average == "macro":
        return res.float().mean(sum_dim)
    if average == "weighted":
        weight = tp + fn
        if multidim_average == "global":
            return (res * (weight / weight.sum()).reshape(*weight.shape, 1)).sum(sum_dim)
        return (res * (weight / weight.sum(-1, keepdim=True)).reshape(*weight.shape, 1)).sum(sum_dim)
    if average is None or average == "none":
        return res
    return None


def multiclass_stat_scores(
    preds: Tensor,
    target: Tensor,
    num_classes: int,
    average: Optional[Literal["micro", "macro", "weighted", "none"]] = "micro",
    top_k: int = 1,
    multidim_average: Literal["global", "samplewise"] = "global",
    ignore_index: Optional[int] = None,
    validate_args: bool = True,
) -> Tensor:
    r"""Compute the true positives, false positives, true negatives, false negatives and support for multiclass tasks.

    Related to `Type I and Type II errors`_.

    Accepts the following input tensors:

    - ``preds``: ``(N, ...)`` (int tensor) or ``(N, C, ..)`` (float tensor). If preds is a floating point
      we apply ``torch.argmax`` along the ``C`` dimension to automatically convert probabilities/logits into
      an int tensor.
    - ``target`` (int tensor): ``(N, ...)``

    Args:
        preds: Tensor with predictions
        target: Tensor with true labels
        num_classes: Integer specifying the number of classes
        average:
            Defines the reduction that is applied over labels. Should be one of the following:

            - ``micro``: Sum statistics over all labels
            - ``macro``: Calculate statistics for each label and average them
            - ``weighted``: calculates statistics for each label and computes weighted average using their support
            - ``"none"`` or ``None``: calculates statistic for each label and applies no reduction
        top_k:
            Number of highest probability or logit score predictions considered to find the correct label.
            Only works when ``preds`` contain probabilities/logits.
        multidim_average:
            Defines how additionally dimensions ``...`` should be handled. Should be one of the following:

            - ``global``: Additional dimensions are flatted along the batch dimension
            - ``samplewise``: Statistic will be calculated independently for each sample on the ``N`` axis.
              The statistics in this case are calculated over the additional dimensions.

        ignore_index:
            Specifies a target value that is ignored and does not contribute to the metric calculation
        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.

    Returns:
        The metric returns a tensor of shape ``(..., 5)``, where the last dimension corresponds
        to ``[tp, fp, tn, fn, sup]`` (``sup`` stands for support and equals ``tp + fn``). The shape
        depends on ``average`` and ``multidim_average`` parameters:

        - If ``multidim_average`` is set to ``global``:

          - If ``average='micro'/'macro'/'weighted'``, the shape will be ``(5,)``
          - If ``average=None/'none'``, the shape will be ``(C, 5)``

        - If ``multidim_average`` is set to ``samplewise``:

          - If ``average='micro'/'macro'/'weighted'``, the shape will be ``(N, 5)``
          - If ``average=None/'none'``, the shape will be ``(N, C, 5)``

    Example (preds is int tensor):
        >>> from torch import tensor
        >>> from torchmetrics.functional.classification import multiclass_stat_scores
        >>> target = tensor([2, 1, 0, 0])
        >>> preds = tensor([2, 1, 0, 1])
        >>> multiclass_stat_scores(preds, target, num_classes=3, average='micro')
        tensor([3, 1, 7, 1, 4])
        >>> multiclass_stat_scores(preds, target, num_classes=3, average=None)
        tensor([[1, 0, 2, 1, 2],
                [1, 1, 2, 0, 1],
                [1, 0, 3, 0, 1]])

    Example (preds is float tensor):
        >>> from torchmetrics.functional.classification import multiclass_stat_scores
        >>> target = tensor([2, 1, 0, 0])
        >>> preds = tensor([[0.16, 0.26, 0.58],
        ...                 [0.22, 0.61, 0.17],
        ...                 [0.71, 0.09, 0.20],
        ...                 [0.05, 0.82, 0.13]])
        >>> multiclass_stat_scores(preds, target, num_classes=3, average='micro')
        tensor([3, 1, 7, 1, 4])
        >>> multiclass_stat_scores(preds, target, num_classes=3, average=None)
        tensor([[1, 0, 2, 1, 2],
                [1, 1, 2, 0, 1],
                [1, 0, 3, 0, 1]])

    Example (multidim tensors):
        >>> from torchmetrics.functional.classification import multiclass_stat_scores
        >>> target = tensor([[[0, 1], [2, 1], [0, 2]], [[1, 1], [2, 0], [1, 2]]])
        >>> preds = tensor([[[0, 2], [2, 0], [0, 1]], [[2, 2], [2, 1], [1, 0]]])
        >>> multiclass_stat_scores(preds, target, num_classes=3, multidim_average='samplewise', average='micro')
        tensor([[3, 3, 9, 3, 6],
                [2, 4, 8, 4, 6]])
        >>> multiclass_stat_scores(preds, target, num_classes=3, multidim_average='samplewise', average=None)
        tensor([[[2, 1, 3, 0, 2],
                 [0, 1, 3, 2, 2],
                 [1, 1, 3, 1, 2]],
                [[0, 1, 4, 1, 1],
                 [1, 1, 2, 2, 3],
                 [1, 2, 2, 1, 2]]])

    """
    if validate_args:
        _multiclass_stat_scores_arg_validation(num_classes, top_k, average, multidim_average, ignore_index)
        _multiclass_stat_scores_tensor_validation(preds, target, num_classes, multidim_average, ignore_index)
    preds, target = _multiclass_stat_scores_format(preds, target, top_k)
    tp, fp, tn, fn = _multiclass_stat_scores_update(
        preds, target, num_classes, top_k, average, multidim_average, ignore_index
    )
    return _multiclass_stat_scores_compute(tp, fp, tn, fn, average, multidim_average)


def _multilabel_stat_scores_arg_validation(
    num_labels: int,
    threshold: float = 0.5,
    average: Optional[Literal["micro", "macro", "weighted", "none"]] = "micro",
    multidim_average: Literal["global", "samplewise"] = "global",
    ignore_index: Optional[int] = None,
    zero_division: float = 0,
) -> None:
    """Validate non tensor input.

    - ``num_labels`` should be an int larger than 1
    - ``threshold`` has to be a float in the [0,1] range
    - ``average`` has to be "micro" | "macro" | "weighted" | "none"
    - ``multidim_average`` has to be either "global" or "samplewise"
    - ``ignore_index`` has to be None or int
    - ``zero_division`` has to be 0 or 1

    """
    if not isinstance(num_labels, int) or num_labels < 2:
        raise ValueError(f"Expected argument `num_labels` to be an integer larger than 1, but got {num_labels}")
    if not (isinstance(threshold, float) and (0 <= threshold <= 1)):
        raise ValueError(f"Expected argument `threshold` to be a float, but got {threshold}.")
    allowed_average = ("micro", "macro", "weighted", "none", None)
    if average not in allowed_average:
        raise ValueError(f"Expected argument `average` to be one of {allowed_average}, but got {average}")
    allowed_multidim_average = ("global", "samplewise")
    if multidim_average not in allowed_multidim_average:
        raise ValueError(
            f"Expected argument `multidim_average` to be one of {allowed_multidim_average}, but got {multidim_average}"
        )
    if ignore_index is not None and not isinstance(ignore_index, int):
        raise ValueError(f"Expected argument `ignore_index` to either be `None` or an integer, but got {ignore_index}")
    if zero_division not in [0, 1]:
        raise ValueError(f"Expected argument `zero_division` to be 0 or 1, but got {zero_division}.")


def _multilabel_stat_scores_tensor_validation(
    preds: Tensor,
    target: Tensor,
    num_labels: int,
    multidim_average: str,
    ignore_index: Optional[int] = None,
) -> None:
    """Validate tensor input.

    - tensors have to be of same shape
    - the second dimension of both tensors need to be equal to the number of labels
    - all values in target tensor that are not ignored have to be in {0, 1}
    - if pred tensor is not floating point, then all values also have to be in {0, 1}
    - if ``multidim_average`` is set to ``samplewise`` preds tensor needs to be at least 3 dimensional

    """
    # Check that they have same shape
    _check_same_shape(preds, target)

    if preds.shape[1] != num_labels:
        raise ValueError(
            "Expected both `target.shape[1]` and `preds.shape[1]` to be equal to the number of labels"
            f" but got {preds.shape[1]} and expected {num_labels}"
        )

    # Check that target only contains [0,1] values or value in ignore_index
    unique_values = torch.unique(target, dim=None)
    if ignore_index is None:
        check = torch.any((unique_values != 0) & (unique_values != 1))
    else:
        check = torch.any((unique_values != 0) & (unique_values != 1) & (unique_values != ignore_index))
    if check:
        raise RuntimeError(
            f"Detected the following values in `target`: {unique_values} but expected only"
            f" the following values {[0, 1] if ignore_index is None else [ignore_index]}."
        )

    # If preds is label tensor, also check that it only contains [0,1] values
    if not preds.is_floating_point():
        unique_values = torch.unique(preds, dim=None)
        if torch.any((unique_values != 0) & (unique_values != 1)):
            raise RuntimeError(
                f"Detected the following values in `preds`: {unique_values} but expected only"
                " the following values [0,1] since preds is a label tensor."
            )

    if multidim_average != "global" and preds.ndim < 3:
        raise ValueError("Expected input to be at least 3D when multidim_average is set to `samplewise`")


def _multilabel_stat_scores_format(
    preds: Tensor, target: Tensor, num_labels: int, threshold: float = 0.5, ignore_index: Optional[int] = None
) -> tuple[Tensor, Tensor]:
    """Convert all input to label format.

    - If preds tensor is floating point, applies sigmoid if pred tensor not in [0,1] range
    - If preds tensor is floating point, thresholds afterwards
    - Mask all elements that should be ignored with negative numbers for later filtration

    """
    if preds.is_floating_point():
        preds = normalize_logits_if_needed(preds, "sigmoid")
        preds = preds > threshold
    preds = preds.reshape(*preds.shape[:2], -1)
    target = target.reshape(*target.shape[:2], -1)

    if ignore_index is not None:
        idx = target == ignore_index
        target = target.clone()
        target[idx] = -1

    return preds, target


def _multilabel_stat_scores_update(
    preds: Tensor, target: Tensor, multidim_average: Literal["global", "samplewise"] = "global"
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Compute the statistics."""
    sum_dim = [0, -1] if multidim_average == "global" else [-1]
    tp = ((target == preds) & (target == 1)).sum(sum_dim).squeeze()
    fn = ((target != preds) & (target == 1)).sum(sum_dim).squeeze()
    fp = ((target != preds) & (target == 0)).sum(sum_dim).squeeze()
    tn = ((target == preds) & (target == 0)).sum(sum_dim).squeeze()
    return tp, fp, tn, fn


def _multilabel_stat_scores_compute(
    tp: Tensor,
    fp: Tensor,
    tn: Tensor,
    fn: Tensor,
    average: Optional[Literal["micro", "macro", "weighted", "none"]] = "micro",
    multidim_average: Literal["global", "samplewise"] = "global",
) -> Tensor:
    """Stack statistics and compute support also.

    Applies average strategy afterwards.

    """
    res = torch.stack([tp, fp, tn, fn, tp + fn], dim=-1)
    sum_dim = 0 if multidim_average == "global" else 1
    if average == "micro":
        return res.sum(sum_dim)
    if average == "macro":
        return res.float().mean(sum_dim)
    if average == "weighted":
        w = tp + fn
        return (res * (w / w.sum()).reshape(*w.shape, 1)).sum(sum_dim)
    if average is None or average == "none":
        return res
    return None


def multilabel_stat_scores(
    preds: Tensor,
    target: Tensor,
    num_labels: int,
    threshold: float = 0.5,
    average: Optional[Literal["micro", "macro", "weighted", "none"]] = "micro",
    multidim_average: Literal["global", "samplewise"] = "global",
    ignore_index: Optional[int] = None,
    validate_args: bool = True,
) -> Tensor:
    r"""Compute the true positives, false positives, true negatives, false negatives and support for multilabel tasks.

    Related to `Type I and Type II errors`_.

    Accepts the following input tensors:

    - ``preds`` (int or float tensor): ``(N, C, ...)``. If preds is a floating point tensor with values outside
      [0,1] range we consider the input to be logits and will auto apply sigmoid per element. Additionally,
      we convert to int tensor with thresholding using the value in ``threshold``.
    - ``target`` (int tensor): ``(N, C, ...)``

    Args:
        preds: Tensor with predictions
        target: Tensor with true labels
        num_labels: Integer specifying the number of labels
        threshold: Threshold for transforming probability to binary (0,1) predictions
        average:
            Defines the reduction that is applied over labels. Should be one of the following:

            - ``micro``: Sum statistics over all labels
            - ``macro``: Calculate statistics for each label and average them
            - ``weighted``: calculates statistics for each label and computes weighted average using their support
            - ``"none"`` or ``None``: calculates statistic for each label and applies no reduction

        multidim_average:
            Defines how additionally dimensions ``...`` should be handled. Should be one of the following:

            - ``global``: Additional dimensions are flatted along the batch dimension
            - ``samplewise``: Statistic will be calculated independently for each sample on the ``N`` axis.
              The statistics in this case are calculated over the additional dimensions.

        ignore_index:
            Specifies a target value that is ignored and does not contribute to the metric calculation
        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.

    Returns:
        The metric returns a tensor of shape ``(..., 5)``, where the last dimension corresponds
        to ``[tp, fp, tn, fn, sup]`` (``sup`` stands for support and equals ``tp + fn``). The shape
        depends on ``average`` and ``multidim_average`` parameters:

        - If ``multidim_average`` is set to ``global``:

          - If ``average='micro'/'macro'/'weighted'``, the shape will be ``(5,)``
          - If ``average=None/'none'``, the shape will be ``(C, 5)``

        - If ``multidim_average`` is set to ``samplewise``:

          - If ``average='micro'/'macro'/'weighted'``, the shape will be ``(N, 5)``
          - If ``average=None/'none'``, the shape will be ``(N, C, 5)``

    Example (preds is int tensor):
        >>> from torch import tensor
        >>> from torchmetrics.functional.classification import multilabel_stat_scores
        >>> target = tensor([[0, 1, 0], [1, 0, 1]])
        >>> preds = tensor([[0, 0, 1], [1, 0, 1]])
        >>> multilabel_stat_scores(preds, target, num_labels=3, average='micro')
        tensor([2, 1, 2, 1, 3])
        >>> multilabel_stat_scores(preds, target, num_labels=3, average=None)
        tensor([[1, 0, 1, 0, 1],
                [0, 0, 1, 1, 1],
                [1, 1, 0, 0, 1]])

    Example (preds is float tensor):
        >>> from torchmetrics.functional.classification import multilabel_stat_scores
        >>> target = tensor([[0, 1, 0], [1, 0, 1]])
        >>> preds = tensor([[0.11, 0.22, 0.84], [0.73, 0.33, 0.92]])
        >>> multilabel_stat_scores(preds, target, num_labels=3, average='micro')
        tensor([2, 1, 2, 1, 3])
        >>> multilabel_stat_scores(preds, target, num_labels=3, average=None)
        tensor([[1, 0, 1, 0, 1],
                [0, 0, 1, 1, 1],
                [1, 1, 0, 0, 1]])

    Example (multidim tensors):
        >>> from torchmetrics.functional.classification import multilabel_stat_scores
        >>> target = tensor([[[0, 1], [1, 0], [0, 1]], [[1, 1], [0, 0], [1, 0]]])
        >>> preds = tensor([[[0.59, 0.91], [0.91, 0.99], [0.63, 0.04]],
        ...                 [[0.38, 0.04], [0.86, 0.780], [0.45, 0.37]]])
        >>> multilabel_stat_scores(preds, target, num_labels=3, multidim_average='samplewise', average='micro')
        tensor([[2, 3, 0, 1, 3],
                [0, 2, 1, 3, 3]])
        >>> multilabel_stat_scores(preds, target, num_labels=3, multidim_average='samplewise', average=None)
        tensor([[[1, 1, 0, 0, 1],
                 [1, 1, 0, 0, 1],
                 [0, 1, 0, 1, 1]],
                [[0, 0, 0, 2, 2],
                 [0, 2, 0, 0, 0],
                 [0, 0, 1, 1, 1]]])

    """
    if validate_args:
        _multilabel_stat_scores_arg_validation(num_labels, threshold, average, multidim_average, ignore_index)
        _multilabel_stat_scores_tensor_validation(preds, target, num_labels, multidim_average, ignore_index)
    preds, target = _multilabel_stat_scores_format(preds, target, num_labels, threshold, ignore_index)
    tp, fp, tn, fn = _multilabel_stat_scores_update(preds, target, multidim_average)
    return _multilabel_stat_scores_compute(tp, fp, tn, fn, average, multidim_average)


def _del_column(data: Tensor, idx: int) -> Tensor:
    """Delete the column at index."""
    return torch.cat([data[:, :idx], data[:, (idx + 1) :]], 1)


def _drop_negative_ignored_indices(
    preds: Tensor, target: Tensor, ignore_index: int, mode: DataType
) -> tuple[Tensor, Tensor]:
    """Remove negative ignored indices.

    Args:
        preds: Predicted tensor
        target: Ground truth tensor
        ignore_index: Specify a class (label) to ignore. If given, this class index does not contribute
            to the returned score, regardless of reduction method. If an index is ignored, and
            ``reduce='macro'``, the class statistics for the ignored class will all be returned
            as ``-1``.
        mode: Mode of the input tensors

    Return:
        Tensors of preds and target without negative ignore target values.

    """
    if mode == mode.MULTIDIM_MULTICLASS and preds.dtype == torch.float:
        # In case or multi-dimensional multi-class with logits
        num_dims = len(preds.shape)
        num_classes = preds.shape[1]
        # move class dim to last so that we can flatten the additional dimensions into N: [N, C, ...] -> [N, ..., C]
        preds = preds.transpose(1, num_dims - 1)

        # flatten: [N, ..., C] -> [N', C]
        preds = preds.reshape(-1, num_classes)
        target = target.reshape(-1)

    if mode in [mode.MULTICLASS, mode.MULTIDIM_MULTICLASS]:
        preds = preds[target != ignore_index]
        target = target[target != ignore_index]

    return preds, target


def _stat_scores(
    preds: Tensor,
    target: Tensor,
    reduce: Optional[str] = "micro",
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Calculate the number of tp, fp, tn, fn.

    Args:
        preds: An ``(N, C)`` or ``(N, C, X)`` tensor of predictions (0 or 1)
        target: An ``(N, C)`` or ``(N, C, X)`` tensor of true labels (0 or 1)
        reduce: One of ``'micro'``, ``'macro'``, ``'samples'``

    Return:
        Returns a list of 4 tensors; tp, fp, tn, fn.
        The shape of the returned tensors depends on the shape of the inputs
        and the ``reduce`` parameter:

        If inputs are of the shape ``(N, C)``, then:

        - If ``reduce='micro'``, the returned tensors are 1 element tensors
        - If ``reduce='macro'``, the returned tensors are ``(C,)`` tensors
        - If ``reduce='samples'``, the returned tensors are ``(N,)`` tensors

        If inputs are of the shape ``(N, C, X)``, then:

        - If ``reduce='micro'``, the returned tensors are ``(N,)`` tensors
        - If ``reduce='macro'``, the returned tensors are ``(N,C)`` tensors
        - If ``reduce='samples'``, the returned tensors are ``(N,X)`` tensors

    """
    dim: Union[int, list[int]] = 1  # for "samples"
    if reduce == "micro":
        dim = [0, 1] if preds.ndim == 2 else [1, 2]
    elif reduce == "macro":
        dim = 0 if preds.ndim == 2 else 2

    true_pred, false_pred = target == preds, target != preds
    pos_pred, neg_pred = preds == 1, preds == 0

    tp = (true_pred * pos_pred).sum(dim=dim)
    fp = (false_pred * pos_pred).sum(dim=dim)

    tn = (true_pred * neg_pred).sum(dim=dim)
    fn = (false_pred * neg_pred).sum(dim=dim)

    return tp.long(), fp.long(), tn.long(), fn.long()


def _stat_scores_update(
    preds: Tensor,
    target: Tensor,
    reduce: Optional[str] = "micro",
    mdmc_reduce: Optional[str] = None,
    num_classes: Optional[int] = None,
    top_k: Optional[int] = 1,
    threshold: float = 0.5,
    multiclass: Optional[bool] = None,
    ignore_index: Optional[int] = None,
    mode: Optional[DataType] = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Calculate true positives, false positives, true negatives, false negatives.

    Raises:
        ValueError:
            The `ignore_index` is not valid
        ValueError:
            When `ignore_index` is used with binary data
        ValueError:
            When inputs are multi-dimensional multi-class, and the ``mdmc_reduce`` parameter is not set

    Args:
        preds: Predicted tensor
        target: Ground truth tensor
        reduce: Defines the reduction that is applied
        mdmc_reduce: Defines how the multi-dimensional multi-class inputs are handled
        num_classes: Number of classes. Necessary for (multi-dimensional) multi-class or multi-label data.
        top_k: Number of the highest probability or logit score predictions considered finding the correct label,
            relevant only for (multi-dimensional) multi-class inputs
        threshold: Threshold for transforming probability or logit predictions to binary (0,1) predictions, in the case
            of binary or multi-label inputs. Default value of 0.5 corresponds to input being probabilities
        multiclass: Used only in certain special cases, where you want to treat inputs as a different type
            than what they appear to be
        ignore_index: Specify a class (label) to ignore. If given, this class index does not contribute
            to the returned score, regardless of reduction method. If an index is ignored, and
            ``reduce='macro'``, the class statistics for the ignored class will all be returned
            as ``-1``.
        mode: Mode of the input tensors

    """
    _negative_index_dropped = False

    if ignore_index is not None and ignore_index < 0 and mode is not None:
        preds, target = _drop_negative_ignored_indices(preds, target, ignore_index, mode)
        _negative_index_dropped = True

    preds, target, _ = _input_format_classification(
        preds,
        target,
        threshold=threshold,
        num_classes=num_classes,
        multiclass=multiclass,
        top_k=top_k,
        ignore_index=ignore_index,
    )

    if ignore_index is not None and ignore_index >= preds.shape[1]:
        raise ValueError(f"The `ignore_index` {ignore_index} is not valid for inputs with {preds.shape[1]} classes")

    if ignore_index is not None and preds.shape[1] == 1:
        raise ValueError("You can not use `ignore_index` with binary data.")

    if preds.ndim == 3:
        if not mdmc_reduce:
            raise ValueError(
                "When your inputs are multi-dimensional multi-class, you have to set the `mdmc_reduce` parameter"
            )
        if mdmc_reduce == "global":
            preds = torch.transpose(preds, 1, 2).reshape(-1, preds.shape[1])
            target = torch.transpose(target, 1, 2).reshape(-1, target.shape[1])

    # Delete what is in ignore_index, if applicable (and classes don't matter):
    if ignore_index is not None and reduce != "macro" and not _negative_index_dropped:
        preds = _del_column(preds, ignore_index)
        target = _del_column(target, ignore_index)

    tp, fp, tn, fn = _stat_scores(preds, target, reduce=reduce)

    # Take care of ignore_index
    if ignore_index is not None and reduce == "macro" and not _negative_index_dropped:
        tp[..., ignore_index] = -1
        fp[..., ignore_index] = -1
        tn[..., ignore_index] = -1
        fn[..., ignore_index] = -1

    return tp, fp, tn, fn


def _stat_scores_compute(tp: Tensor, fp: Tensor, tn: Tensor, fn: Tensor) -> Tensor:
    """Compute the number of true positives, false positives, true negatives, false negatives.

    Concatenates the input tensors along with the support into one output.

    Args:
        tp: True positives
        fp: False positives
        tn: True negatives
        fn: False negatives

    """
    stats = [
        tp.unsqueeze(-1),
        fp.unsqueeze(-1),
        tn.unsqueeze(-1),
        fn.unsqueeze(-1),
        tp.unsqueeze(-1) + fn.unsqueeze(-1),  # support
    ]
    outputs: Tensor = torch.cat(stats, -1)
    return torch.where(outputs < 0, tensor(-1, device=outputs.device), outputs)


def _reduce_stat_scores(
    numerator: Tensor,
    denominator: Tensor,
    weights: Optional[Tensor],
    average: Optional[str],
    mdmc_average: Optional[str],
    zero_division: int = 0,
) -> Tensor:
    """Reduces scores of type ``numerator/denominator`` or.

    ``weights * (numerator/denominator)``, if ``average='weighted'``.

    Args:
        numerator: A tensor with numerator numbers.
        denominator: A tensor with denominator numbers. If a denominator is
            negative, the class will be ignored (if averaging), or its score
            will be returned as ``nan`` (if ``average=None``).
            If the denominator is zero, then ``zero_division`` score will be
            used for those elements.
        weights: A tensor of weights to be used if ``average='weighted'``.
        average: The method to average the scores
        mdmc_average: The method to average the scores if inputs were multi-dimensional multi-class (MDMC)
        zero_division: The value to use for the score if denominator equals zero.

    """
    numerator, denominator = numerator.float(), denominator.float()
    zero_div_mask = denominator == 0
    ignore_mask = denominator < 0

    weights = torch.ones_like(denominator) if weights is None else weights.float()

    numerator = torch.where(
        zero_div_mask, tensor(zero_division, dtype=numerator.dtype, device=numerator.device), numerator
    )
    denominator = torch.where(
        zero_div_mask | ignore_mask, tensor(1.0, dtype=denominator.dtype, device=denominator.device), denominator
    )
    weights = torch.where(ignore_mask, tensor(0.0, dtype=weights.dtype, device=weights.device), weights)

    if average not in (AverageMethod.MICRO, AverageMethod.NONE, None):
        weights = weights / weights.sum(dim=-1, keepdim=True)

    scores = weights * (numerator / denominator)

    # This is in case where sum(weights) = 0, which happens if we ignore the only present class with average='weighted'
    scores = torch.where(torch.isnan(scores), tensor(zero_division, dtype=scores.dtype, device=scores.device), scores)

    if mdmc_average == MDMCAverageMethod.SAMPLEWISE:
        scores = scores.mean(dim=0)
        ignore_mask = ignore_mask.sum(dim=0).bool()

    if average in (AverageMethod.NONE, None):
        return torch.where(ignore_mask, tensor(float("nan"), device=scores.device), scores)
    return scores.sum()


def stat_scores(
    preds: Tensor,
    target: Tensor,
    task: Literal["binary", "multiclass", "multilabel"],
    threshold: float = 0.5,
    num_classes: Optional[int] = None,
    num_labels: Optional[int] = None,
    average: Optional[Literal["micro", "macro", "weighted", "none"]] = "micro",
    multidim_average: Optional[Literal["global", "samplewise"]] = "global",
    top_k: Optional[int] = 1,
    ignore_index: Optional[int] = None,
    validate_args: bool = True,
) -> Tensor:
    r"""Compute the number of true positives, false positives, true negatives, false negatives and the support.

    This function is a simple wrapper to get the task specific versions of this metric, which is done by setting the
    ``task`` argument to either ``'binary'``, ``'multiclass'`` or ``multilabel``. See the documentation of
    :func:`~torchmetrics.functional.classification.binary_stat_scores`,
    :func:`~torchmetrics.functional.classification.multiclass_stat_scores` and
    :func:`~torchmetrics.functional.classification.multilabel_stat_scores` for the specific
    details of each argument influence and examples.

    Legacy Example:
        >>> from torch import tensor
        >>> preds  = tensor([1, 0, 2, 1])
        >>> target = tensor([1, 1, 2, 0])
        >>> stat_scores(preds, target, task='multiclass', num_classes=3, average='micro')
        tensor([2, 2, 6, 2, 4])
        >>> stat_scores(preds, target, task='multiclass', num_classes=3, average=None)
        tensor([[0, 1, 2, 1, 1],
                [1, 1, 1, 1, 2],
                [1, 0, 3, 0, 1]])

    """
    task = ClassificationTask.from_str(task)
    assert multidim_average is not None  # noqa: S101  # needed for mypy
    if task == ClassificationTask.BINARY:
        return binary_stat_scores(preds, target, threshold, multidim_average, ignore_index, validate_args)
    if task == ClassificationTask.MULTICLASS:
        if not isinstance(num_classes, int):
            raise ValueError(f"`num_classes` is expected to be `int` but `{type(num_classes)} was passed.`")
        if not isinstance(top_k, int):
            raise ValueError(f"`top_k` is expected to be `int` but `{type(top_k)} was passed.`")
        return multiclass_stat_scores(
            preds, target, num_classes, average, top_k, multidim_average, ignore_index, validate_args
        )
    if task == ClassificationTask.MULTILABEL:
        if not isinstance(num_labels, int):
            raise ValueError(f"`num_labels` is expected to be `int` but `{type(num_labels)} was passed.`")
        return multilabel_stat_scores(
            preds, target, num_labels, threshold, average, multidim_average, ignore_index, validate_args
        )
    raise ValueError(f"Unsupported task `{task}`")
