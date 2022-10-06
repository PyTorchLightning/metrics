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
from typing_extensions import Literal

from torchmetrics.utilities.checks import _check_same_shape
from torchmetrics.utilities.data import _bincount, _movedim, select_topk


def _binary_stat_scores_arg_validation(
    threshold: float = 0.5,
    multidim_average: Literal["global", "samplewise"] = "global",
    ignore_index: Optional[int] = None,
) -> None:
    """Validate non tensor input.

    - ``threshold`` has to be a float in the [0,1] range
    - ``multidim_average`` has to be either "global" or "samplewise"
    - ``ignore_index`` has to be None or int
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
    - if ``multidim_average`` is set to ``samplewise`` preds tensor needs to be atleast 2 dimensional
    """
    # Check that they have same shape
    _check_same_shape(preds, target)

    # Check that target only contains [0,1] values or value in ignore_index
    unique_values = torch.unique(target)
    if ignore_index is None:
        check = torch.any((unique_values != 0) & (unique_values != 1))
    else:
        check = torch.any((unique_values != 0) & (unique_values != 1) & (unique_values != ignore_index))
    if check:
        raise RuntimeError(
            f"Detected the following values in `target`: {unique_values} but expected only"
            f" the following values {[0,1] + [] if ignore_index is None else [ignore_index]}."
        )

    # If preds is label tensor, also check that it only contains [0,1] values
    if not preds.is_floating_point():
        unique_values = torch.unique(preds)
        if torch.any((unique_values != 0) & (unique_values != 1)):
            raise RuntimeError(
                f"Detected the following values in `preds`: {unique_values} but expected only"
                " the following values [0,1] since `preds` is a label tensor."
            )

    if multidim_average != "global" and preds.ndim < 2:
        raise ValueError("Expected input to be atleast 2D when multidim_average is set to `samplewise`")


def _binary_stat_scores_format(
    preds: Tensor,
    target: Tensor,
    threshold: float = 0.5,
    ignore_index: Optional[int] = None,
) -> Tuple[Tensor, Tensor]:
    """Convert all input to label format.

    - If preds tensor is floating point, applies sigmoid if pred tensor not in [0,1] range
    - If preds tensor is floating point, thresholds afterwards
    - Mask all datapoints that should be ignored with negative values
    """
    if preds.is_floating_point():
        if not torch.all((0 <= preds) * (preds <= 1)):
            # preds is logits, convert with sigmoid
            preds = preds.sigmoid()
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
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Computes the statistics."""
    sum_dim = [0, 1] if multidim_average == "global" else 1
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
    r"""Computes the number of true positives, false positives, true negatives, false negatives and the support for
    binary tasks. Related to `Type I and Type II errors`_.

    Accepts the following input tensors:

    - ``preds`` (int or float tensor): ``(N, ...)``. If preds is a floating point tensor with values outside
      [0,1] range we consider the input to be logits and will auto apply sigmoid per element. Addtionally,
      we convert to int tensor with thresholding using the value in ``threshold``.
    - ``target`` (int tensor): ``(N, ...)``

    The influence of the additional dimension ``...`` (if present) will be determined by the `multidim_average`
    argument.

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
        >>> from torchmetrics.functional.classification import binary_stat_scores
        >>> target = torch.tensor([0, 1, 0, 1, 0, 1])
        >>> preds = torch.tensor([0, 0, 1, 1, 0, 1])
        >>> binary_stat_scores(preds, target)
        tensor([2, 1, 2, 1, 3])

    Example (preds is float tensor):
        >>> from torchmetrics.functional.classification import binary_stat_scores
        >>> target = torch.tensor([0, 1, 0, 1, 0, 1])
        >>> preds = torch.tensor([0.11, 0.22, 0.84, 0.73, 0.33, 0.92])
        >>> binary_stat_scores(preds, target)
        tensor([2, 1, 2, 1, 3])

    Example (multidim tensors):
        >>> from torchmetrics.functional.classification import binary_stat_scores
        >>> target = torch.tensor([[[0, 1], [1, 0], [0, 1]], [[1, 1], [0, 0], [1, 0]]])
        >>> preds = torch.tensor(
        ...     [
        ...         [[0.59, 0.91], [0.91, 0.99], [0.63, 0.04]],
        ...         [[0.38, 0.04], [0.86, 0.780], [0.45, 0.37]],
        ...     ]
        ... )
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
    num_classes: int,
    top_k: int = 1,
    average: Optional[Literal["micro", "macro", "weighted", "none"]] = "macro",
    multidim_average: Literal["global", "samplewise"] = "global",
    ignore_index: Optional[int] = None,
) -> None:
    """Validate non tensor input.

    - ``num_classes`` has to be a int larger than 1
    - ``top_k`` has to be an int larger than 0 but no larger than number of classes
    - ``average`` has to be "micro" | "macro" | "weighted" | "none"
    - ``multidim_average`` has to be either "global" or "samplewise"
    - ``ignore_index`` has to be None or int
    """
    if not isinstance(num_classes, int) or num_classes < 2:
        raise ValueError(f"Expected argument `num_classes` to be an integer larger than 1, but got {num_classes}")
    if not isinstance(top_k, int) and top_k < 1:
        raise ValueError(f"Expected argument `top_k` to be an integer larger than or equal to 1, but got {top_k}")
    if top_k > num_classes:
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


def _multiclass_stat_scores_tensor_validation(
    preds: Tensor,
    target: Tensor,
    num_classes: int,
    multidim_average: Literal["global", "samplewise"] = "global",
    ignore_index: Optional[int] = None,
) -> None:
    """Validate tensor input.

    - if target has one more dimension than preds, then all dimensions except for preds.shape[1] should match
    exactly. preds.shape[1] should have size equal to number of classes
    - if preds and target have same number of dims, then all dimensions should match
    - if ``multidim_average`` is set to ``samplewise`` preds tensor needs to be atleast 2 dimensional in the
    int case and 3 dimensional in the float case
    - all values in target tensor that are not ignored have to be {0, ..., num_classes - 1}
    - if pred tensor is not floating point, then all values also have to be in {0, ..., num_classes - 1}
    """
    if preds.ndim == target.ndim + 1:
        if not preds.is_floating_point():
            raise ValueError("If `preds` have one dimension more than `target`, `preds` should be a float tensor.")
        if preds.shape[1] != num_classes:
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
                "If `preds` have one dimension more than `target`, the shape of `preds` should "
                " atleast 3D when multidim_average is set to `samplewise`"
            )

    elif preds.ndim == target.ndim:
        if preds.shape != target.shape:
            raise ValueError(
                "The `preds` and `target` should have the same shape,",
                f" got `preds` with shape={preds.shape} and `target` with shape={target.shape}.",
            )
        if multidim_average != "global" and preds.ndim < 2:
            raise ValueError(
                "When `preds` and `target` have the same shape, the shape of `preds` should "
                " atleast 2D when multidim_average is set to `samplewise`"
            )
    else:
        raise ValueError(
            "Either `preds` and `target` both should have the (same) shape (N, ...), or `target` should be (N, ...)"
            " and `preds` should be (N, C, ...)."
        )

    num_unique_values = len(torch.unique(target))
    if ignore_index is None:
        check = num_unique_values > num_classes
    else:
        check = num_unique_values > num_classes + 1
    if check:
        raise RuntimeError(
            "Detected more unique values in `target` than `num_classes`. Expected only "
            f"{num_classes if ignore_index is None else num_classes + 1} but found"
            f"{num_unique_values} in `target`."
        )

    if not preds.is_floating_point():
        unique_values = torch.unique(preds)
        if len(unique_values) > num_classes:
            raise RuntimeError(
                "Detected more unique values in `preds` than `num_classes`. Expected only "
                f"{num_classes} but found {len(unique_values)} in `preds`."
            )


def _multiclass_stat_scores_format(
    preds: Tensor,
    target: Tensor,
    top_k: int = 1,
) -> Tuple[Tensor, Tensor]:
    """Convert all input to label format except if ``top_k`` is not 1.

    - Applies argmax if preds have one more dimension than target
    - Flattens additional dimensions
    """
    # Apply argmax if we have one more dimension
    if preds.ndim == target.ndim + 1 and top_k == 1:
        preds = preds.argmax(dim=1)
    if top_k != 1:
        preds = preds.reshape(*preds.shape[:2], -1)
    else:
        preds = preds.reshape(preds.shape[0], -1)
    target = target.reshape(target.shape[0], -1)
    return preds, target


def _multiclass_stat_scores_update(
    preds: Tensor,
    target: Tensor,
    num_classes: int,
    top_k: int = 1,
    multidim_average: Literal["global", "samplewise"] = "global",
    ignore_index: Optional[int] = None,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Computes the statistics.

    - If ``multidim_average`` is equal to samplewise or ``top_k`` is not 1, we transform both preds and
    target into one hot format.
    - Else we calculate statistics by first calculating the confusion matrix and afterwards deriving the
    statistics from that
    - Remove all datapoints that should be ignored. Depending on if ``ignore_index`` is in the set of labels
    or outside we have do use different augmentation stategies when one hot encoding.
    """
    if multidim_average == "samplewise" or top_k != 1:
        ignore_in = 0 <= ignore_index <= num_classes - 1 if ignore_index is not None else None
        if ignore_index is not None and not ignore_in:
            preds = preds.clone()
            target = target.clone()
            idx = target == ignore_index
            preds[idx] = num_classes
            target[idx] = num_classes

        if top_k > 1:
            preds_oh = _movedim(select_topk(preds, topk=top_k, dim=1), 1, -1)
        else:
            preds_oh = torch.nn.functional.one_hot(
                preds, num_classes + 1 if ignore_index is not None and not ignore_in else num_classes
            )
        target_oh = torch.nn.functional.one_hot(
            target, num_classes + 1 if ignore_index is not None and not ignore_in else num_classes
        )
        if ignore_index is not None:
            if 0 <= ignore_index <= num_classes - 1:
                target_oh[target == ignore_index, :] = -1
            else:
                preds_oh = preds_oh[..., :-1]
                target_oh = target_oh[..., :-1]
                target_oh[target == num_classes, :] = -1
        sum_dim = [0, 1] if multidim_average == "global" else [1]
        tp = ((target_oh == preds_oh) & (target_oh == 1)).sum(sum_dim)
        fn = ((target_oh != preds_oh) & (target_oh == 1)).sum(sum_dim)
        fp = ((target_oh != preds_oh) & (target_oh == 0)).sum(sum_dim)
        tn = ((target_oh == preds_oh) & (target_oh == 0)).sum(sum_dim)
        return tp, fp, tn, fn
    else:
        preds = preds.flatten()
        target = target.flatten()
        if ignore_index is not None:
            idx = target != ignore_index
            preds = preds[idx]
            target = target[idx]
        unique_mapping = (target * num_classes + preds).to(torch.long)
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
    average: Optional[Literal["micro", "macro", "weighted", "none"]] = "macro",
    multidim_average: Literal["global", "samplewise"] = "global",
) -> Tensor:
    """Stack statistics and compute support also.

    Applies average strategy afterwards.
    """
    res = torch.stack([tp, fp, tn, fn, tp + fn], dim=-1)
    sum_dim = 0 if multidim_average == "global" else 1
    if average == "micro":
        return res.sum(sum_dim)
    elif average == "macro":
        return res.float().mean(sum_dim)
    elif average == "weighted":
        weight = tp + fn
        if multidim_average == "global":
            return (res * (weight / weight.sum()).reshape(*weight.shape, 1)).sum(sum_dim)
        else:
            return (res * (weight / weight.sum(-1, keepdim=True)).reshape(*weight.shape, 1)).sum(sum_dim)
    elif average is None or average == "none":
        return res


def multiclass_stat_scores(
    preds: Tensor,
    target: Tensor,
    num_classes: int,
    average: Optional[Literal["micro", "macro", "weighted", "none"]] = "macro",
    top_k: int = 1,
    multidim_average: Literal["global", "samplewise"] = "global",
    ignore_index: Optional[int] = None,
    validate_args: bool = True,
) -> Tensor:
    r"""Computes the number of true positives, false positives, true negatives, false negatives and the support for
    multiclass tasks. Related to `Type I and Type II errors`_.

    Accepts the following input tensors:

    - ``preds``: ``(N, ...)`` (int tensor) or ``(N, C, ..)`` (float tensor). If preds is a floating point
      we apply ``torch.argmax`` along the ``C`` dimension to automatically convert probabilities/logits into
      an int tensor.
    - ``target`` (int tensor): ``(N, ...)``

    The influence of the additional dimension ``...`` (if present) will be determined by the `multidim_average`
    argument.

    Args:
        preds: Tensor with predictions
        target: Tensor with true labels
        num_classes: Integer specifing the number of classes
        average:
            Defines the reduction that is applied over labels. Should be one of the following:

            - ``micro``: Sum statistics over all labels
            - ``macro``: Calculate statistics for each label and average them
            - ``weighted``: Calculates statistics for each label and computes weighted average using their support
            - ``"none"`` or ``None``: Calculates statistic for each label and applies no reduction
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
        >>> from torchmetrics.functional.classification import multiclass_stat_scores
        >>> target = torch.tensor([2, 1, 0, 0])
        >>> preds = torch.tensor([2, 1, 0, 1])
        >>> multiclass_stat_scores(preds, target, num_classes=3, average='micro')
        tensor([3, 1, 7, 1, 4])
        >>> multiclass_stat_scores(preds, target, num_classes=3, average=None)
        tensor([[1, 0, 2, 1, 2],
                [1, 1, 2, 0, 1],
                [1, 0, 3, 0, 1]])

    Example (preds is float tensor):
        >>> from torchmetrics.functional.classification import multiclass_stat_scores
        >>> target = torch.tensor([2, 1, 0, 0])
        >>> preds = torch.tensor([
        ...   [0.16, 0.26, 0.58],
        ...   [0.22, 0.61, 0.17],
        ...   [0.71, 0.09, 0.20],
        ...   [0.05, 0.82, 0.13],
        ... ])
        >>> multiclass_stat_scores(preds, target, num_classes=3, average='micro')
        tensor([3, 1, 7, 1, 4])
        >>> multiclass_stat_scores(preds, target, num_classes=3, average=None)
        tensor([[1, 0, 2, 1, 2],
                [1, 1, 2, 0, 1],
                [1, 0, 3, 0, 1]])

    Example (multidim tensors):
        >>> from torchmetrics.functional.classification import multiclass_stat_scores
        >>> target = torch.tensor([[[0, 1], [2, 1], [0, 2]], [[1, 1], [2, 0], [1, 2]]])
        >>> preds = torch.tensor([[[0, 2], [2, 0], [0, 1]], [[2, 2], [2, 1], [1, 0]]])
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
    tp, fp, tn, fn = _multiclass_stat_scores_update(preds, target, num_classes, top_k, multidim_average, ignore_index)
    return _multiclass_stat_scores_compute(tp, fp, tn, fn, average, multidim_average)


def _multilabel_stat_scores_arg_validation(
    num_labels: int,
    threshold: float = 0.5,
    average: Optional[Literal["micro", "macro", "weighted", "none"]] = "macro",
    multidim_average: Literal["global", "samplewise"] = "global",
    ignore_index: Optional[int] = None,
) -> None:
    """Validate non tensor input.

    - ``num_labels`` should be an int larger than 1
    - ``threshold`` has to be a float in the [0,1] range
    - ``average`` has to be "micro" | "macro" | "weighted" | "none"
    - ``multidim_average`` has to be either "global" or "samplewise"
    - ``ignore_index`` has to be None or int
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
    - if ``multidim_average`` is set to ``samplewise`` preds tensor needs to be atleast 3 dimensional
    """
    # Check that they have same shape
    _check_same_shape(preds, target)

    if preds.shape[1] != num_labels:
        raise ValueError(
            "Expected both `target.shape[1]` and `preds.shape[1]` to be equal to the number of labels"
            f" but got {preds.shape[1]} and expected {num_labels}"
        )

    # Check that target only contains [0,1] values or value in ignore_index
    unique_values = torch.unique(target)
    if ignore_index is None:
        check = torch.any((unique_values != 0) & (unique_values != 1))
    else:
        check = torch.any((unique_values != 0) & (unique_values != 1) & (unique_values != ignore_index))
    if check:
        raise RuntimeError(
            f"Detected the following values in `target`: {unique_values} but expected only"
            f" the following values {[0,1] + [] if ignore_index is None else [ignore_index]}."
        )

    # If preds is label tensor, also check that it only contains [0,1] values
    if not preds.is_floating_point():
        unique_values = torch.unique(preds)
        if torch.any((unique_values != 0) & (unique_values != 1)):
            raise RuntimeError(
                f"Detected the following values in `preds`: {unique_values} but expected only"
                " the following values [0,1] since preds is a label tensor."
            )

    if multidim_average != "global" and preds.ndim < 3:
        raise ValueError("Expected input to be atleast 3D when multidim_average is set to `samplewise`")


def _multilabel_stat_scores_format(
    preds: Tensor, target: Tensor, num_labels: int, threshold: float = 0.5, ignore_index: Optional[int] = None
) -> Tuple[Tensor, Tensor]:
    """Convert all input to label format.

    - If preds tensor is floating point, applies sigmoid if pred tensor not in [0,1] range
    - If preds tensor is floating point, thresholds afterwards
    - Mask all elements that should be ignored with negative numbers for later filtration
    """
    if preds.is_floating_point():
        if not torch.all((0 <= preds) * (preds <= 1)):
            preds = preds.sigmoid()
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
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Computes the statistics."""
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
    average: Optional[Literal["micro", "macro", "weighted", "none"]] = "macro",
    multidim_average: Literal["global", "samplewise"] = "global",
) -> Tensor:
    """Stack statistics and compute support also.

    Applies average strategy afterwards.
    """
    res = torch.stack([tp, fp, tn, fn, tp + fn], dim=-1)
    sum_dim = 0 if multidim_average == "global" else 1
    if average == "micro":
        return res.sum(sum_dim)
    elif average == "macro":
        return res.float().mean(sum_dim)
    elif average == "weighted":
        w = tp + fn
        return (res * (w / w.sum()).reshape(*w.shape, 1)).sum(sum_dim)
    elif average is None or average == "none":
        return res


def multilabel_stat_scores(
    preds: Tensor,
    target: Tensor,
    num_labels: int,
    threshold: float = 0.5,
    average: Optional[Literal["micro", "macro", "weighted", "none"]] = "macro",
    multidim_average: Literal["global", "samplewise"] = "global",
    ignore_index: Optional[int] = None,
    validate_args: bool = True,
) -> Tensor:
    r"""Computes the number of true positives, false positives, true negatives, false negatives and the support for
    multilabel tasks. Related to `Type I and Type II errors`_.

    Accepts the following input tensors:

    - ``preds`` (int or float tensor): ``(N, C, ...)``. If preds is a floating point tensor with values outside
      [0,1] range we consider the input to be logits and will auto apply sigmoid per element. Addtionally,
      we convert to int tensor with thresholding using the value in ``threshold``.
    - ``target`` (int tensor): ``(N, C, ...)``

    The influence of the additional dimension ``...`` (if present) will be determined by the `multidim_average`
    argument.

    Args:
        preds: Tensor with predictions
        target: Tensor with true labels
        num_labels: Integer specifing the number of labels
        threshold: Threshold for transforming probability to binary (0,1) predictions
        average:
            Defines the reduction that is applied over labels. Should be one of the following:

            - ``micro``: Sum statistics over all labels
            - ``macro``: Calculate statistics for each label and average them
            - ``weighted``: Calculates statistics for each label and computes weighted average using their support
            - ``"none"`` or ``None``: Calculates statistic for each label and applies no reduction

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
        >>> from torchmetrics.functional.classification import multilabel_stat_scores
        >>> target = torch.tensor([[0, 1, 0], [1, 0, 1]])
        >>> preds = torch.tensor([[0, 0, 1], [1, 0, 1]])
        >>> multilabel_stat_scores(preds, target, num_labels=3, average='micro')
        tensor([2, 1, 2, 1, 3])
        >>> multilabel_stat_scores(preds, target, num_labels=3, average=None)
        tensor([[1, 0, 1, 0, 1],
                [0, 0, 1, 1, 1],
                [1, 1, 0, 0, 1]])

    Example (preds is float tensor):
        >>> from torchmetrics.functional.classification import multilabel_stat_scores
        >>> target = torch.tensor([[0, 1, 0], [1, 0, 1]])
        >>> preds = torch.tensor([[0.11, 0.22, 0.84], [0.73, 0.33, 0.92]])
        >>> multilabel_stat_scores(preds, target, num_labels=3, average='micro')
        tensor([2, 1, 2, 1, 3])
        >>> multilabel_stat_scores(preds, target, num_labels=3, average=None)
        tensor([[1, 0, 1, 0, 1],
                [0, 0, 1, 1, 1],
                [1, 1, 0, 0, 1]])

    Example (multidim tensors):
        >>> from torchmetrics.functional.classification import multilabel_stat_scores
        >>> target = torch.tensor([[[0, 1], [1, 0], [0, 1]], [[1, 1], [0, 0], [1, 0]]])
        >>> preds = torch.tensor(
        ...     [
        ...         [[0.59, 0.91], [0.91, 0.99], [0.63, 0.04]],
        ...         [[0.38, 0.04], [0.86, 0.780], [0.45, 0.37]],
        ...     ]
        ... )
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


def stat_scores(
    preds: Tensor,
    target: Tensor,
    reduce: str = "micro",
    mdmc_reduce: Optional[str] = None,
    num_classes: Optional[int] = None,
    top_k: Optional[int] = None,
    threshold: float = 0.5,
    multiclass: Optional[bool] = None,
    ignore_index: Optional[int] = None,
    task: Optional[Literal["binary", "multiclass", "multilabel"]] = None,
    num_labels: Optional[int] = None,
    average: Optional[Literal["micro", "macro", "weighted", "none"]] = "micro",
    multidim_average: Optional[Literal["global", "samplewise"]] = "global",
    validate_args: bool = True,
) -> Tensor:
    r"""Stat scores.

    .. note::
        From v0.10 an ``'binary_*'``, ``'multiclass_*'``, ``'multilabel_*'`` version now exist of each classification
        metric. Moving forward we recommend using these versions. This base metric will still work as it did
        prior to v0.10 until v0.11. From v0.11 the `task` argument introduced in this metric will be required
        and the general order of arguments may change, such that this metric will just function as an single
        entrypoint to calling the three specialized versions.


    Computes the number of true positives, false positives, true negatives, false negatives.
    Related to `Type I and Type II errors`_ and the `confusion matrix`_.

    The reduction method (how the statistics are aggregated) is controlled by the
    ``reduce`` parameter, and additionally by the ``mdmc_reduce`` parameter in the
    multi-dimensional multi-class case. Accepts all inputs listed in :ref:`pages/classification:input types`.

    Args:
        preds: Predictions from model (probabilities, logits or labels)
        target: Ground truth values
        threshold:
            Threshold for transforming probability or logit predictions to binary (0,1) predictions, in the case
            of binary or multi-label inputs. Default value of 0.5 corresponds to input being probabilities.
        top_k:
            Number of highest probability or logit score predictions considered to find the correct label,
            relevant only for (multi-dimensional) multi-class inputs. The
            default value (``None``) will be interpreted as 1 for these inputs.

            Should be left at default (``None``) for all other types of inputs.
        reduce:
            Defines the reduction that is applied. Should be one of the following:

            - ``'micro'`` [default]: Counts the statistics by summing over all [sample, class]
              combinations (globally). Each statistic is represented by a single integer.
            - ``'macro'``: Counts the statistics for each class separately (over all samples).
              Each statistic is represented by a ``(C,)`` tensor. Requires ``num_classes``
              to be set.
            - ``'samples'``: Counts the statistics for each sample separately (over all classes).
              Each statistic is represented by a ``(N, )`` 1d tensor.

            .. note:: What is considered a sample in the multi-dimensional multi-class case
                depends on the value of ``mdmc_reduce``.

        num_classes:
            Number of classes. Necessary for (multi-dimensional) multi-class or multi-label data.
        ignore_index:
            Specify a class (label) to ignore. If given, this class index does not contribute
            to the returned score, regardless of reduction method. If an index is ignored, and
            ``reduce='macro'``, the class statistics for the ignored class will all be returned
            as ``-1``.
        mdmc_reduce:
            Defines how the multi-dimensional multi-class inputs are handeled. Should be
            one of the following:

            - ``None`` [default]: Should be left unchanged if your data is not multi-dimensional
              multi-class (see :ref:`pages/classification:input types` for the definition of input types).

            - ``'samplewise'``: In this case, the statistics are computed separately for each
              sample on the ``N`` axis, and then the outputs are concatenated together. In each
              sample the extra axes ``...`` are flattened to become the sub-sample axis, and
              statistics for each sample are computed by treating the sub-sample axis as the
              ``N`` axis for that sample.

            - ``'global'``: In this case the ``N`` and ``...`` dimensions of the inputs are
              flattened into a new ``N_X`` sample axis, i.e. the inputs are treated as if they
              were ``(N_X, C)``. From here on the ``reduce`` parameter applies as usual.

        multiclass:
            Used only in certain special cases, where you want to treat inputs as a different type
            than what they appear to be. See the parameter's
            :ref:`documentation section <pages/classification:using the multiclass parameter>`
            for a more detailed explanation and examples.

    Return:
        The metric returns a tensor of shape ``(..., 5)``, where the last dimension corresponds
        to ``[tp, fp, tn, fn, sup]`` (``sup`` stands for support and equals ``tp + fn``). The
        shape depends on the ``reduce`` and ``mdmc_reduce`` (in case of multi-dimensional
        multi-class data) parameters:

        - If the data is not multi-dimensional multi-class, then

          - If ``reduce='micro'``, the shape will be ``(5, )``
          - If ``reduce='macro'``, the shape will be ``(C, 5)``, where ``C`` stands for the number of classes
          - If ``reduce='samples'``, the shape will be ``(N, 5)``, where ``N`` stands for
            the number of samples

        - If the data is multi-dimensional multi-class and ``mdmc_reduce='global'``, then

          - If ``reduce='micro'``, the shape will be ``(5, )``
          - If ``reduce='macro'``, the shape will be ``(C, 5)``
          - If ``reduce='samples'``, the shape will be ``(N*X, 5)``, where ``X`` stands for
            the product of sizes of all "extra" dimensions of the data (i.e. all dimensions
            except for ``C`` and ``N``)

        - If the data is multi-dimensional multi-class and ``mdmc_reduce='samplewise'``, then

          - If ``reduce='micro'``, the shape will be ``(N, 5)``
          - If ``reduce='macro'``, the shape will be ``(N, C, 5)``
          - If ``reduce='samples'``, the shape will be ``(N, X, 5)``

    Raises:
        ValueError:
            If ``reduce`` is none of ``"micro"``, ``"macro"`` or ``"samples"``.
        ValueError:
            If ``mdmc_reduce`` is none of ``None``, ``"samplewise"``, ``"global"``.
        ValueError:
            If ``reduce`` is set to ``"macro"`` and ``num_classes`` is not provided.
        ValueError:
            If ``num_classes`` is set and ``ignore_index`` is not in the range ``[0, num_classes)``.
        ValueError:
            If ``ignore_index`` is used with ``binary data``.
        ValueError:
            If inputs are ``multi-dimensional multi-class`` and ``mdmc_reduce`` is not provided.

    Example:
        >>> from torchmetrics.functional import stat_scores
        >>> preds  = torch.tensor([1, 0, 2, 1])
        >>> target = torch.tensor([1, 1, 2, 0])
        >>> stat_scores(preds, target, reduce='macro', num_classes=3)
        tensor([[0, 1, 2, 1, 1],
                [1, 1, 1, 1, 2],
                [1, 0, 3, 0, 1]])
        >>> stat_scores(preds, target, reduce='micro')
        tensor([2, 2, 6, 2, 4])
    """
    assert multidim_average is not None
    if task == "binary":
        return binary_stat_scores(preds, target, threshold, multidim_average, ignore_index, validate_args)
    if task == "multiclass":
        assert isinstance(num_classes, int)
        assert isinstance(top_k, int)
        return multiclass_stat_scores(
            preds, target, num_classes, average, top_k, multidim_average, ignore_index, validate_args
        )
    if task == "multilabel":
        assert isinstance(num_labels, int)
        return multilabel_stat_scores(
            preds, target, num_labels, threshold, average, multidim_average, ignore_index, validate_args
        )
    raise ValueError(
        f"Expected argument `task` to either be `'binary'`, `'multiclass'` or `'multilabel'` but got {task}"
    )
