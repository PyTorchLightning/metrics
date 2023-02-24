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

from typing import List, Optional, Sequence, Tuple, Union

import humanfriendly
import torch
from torch import Tensor, tensor
from torch.nn import functional as F  # noqa: N812
from typing_extensions import Literal

from torchmetrics.utilities.checks import _check_same_shape
from torchmetrics.utilities.compute import _safe_divide
from torchmetrics.utilities.data import _bincount, _cumsum
from torchmetrics.utilities.enums import ClassificationTask


def _binary_clf_curve(
    preds: Tensor,
    target: Tensor,
    sample_weights: Optional[Sequence] = None,
    pos_label: int = 1,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Calculate the TPs and false positives for all unique thresholds in the preds tensor. Adapted from
    https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/metrics/_ranking.py.

    Args:
        preds: 1d tensor with predictions
        target: 1d tensor with true values
        sample_weights: a 1d tensor with a weight per sample
        pos_label: interger determining what the positive class in target tensor is

    Returns:
        fps: 1d tensor with false positives for different thresholds
        tps: 1d tensor with true positives for different thresholds
        thresholds: the unique thresholds use for calculating fps and tps
    """
    with torch.no_grad():
        if sample_weights is not None and not isinstance(sample_weights, Tensor):
            sample_weights = tensor(sample_weights, device=preds.device, dtype=torch.float)

        # remove class dimension if necessary
        if preds.ndim > target.ndim:
            preds = preds[:, 0]
        desc_score_indices = torch.argsort(preds, descending=True)

        preds = preds[desc_score_indices]
        target = target[desc_score_indices]

        weight = sample_weights[desc_score_indices] if sample_weights is not None else 1.0

        # pred typically has many tied values. Here we extract
        # the indices associated with the distinct values. We also
        # concatenate a value for the end of the curve.
        distinct_value_indices = torch.where(preds[1:] - preds[:-1])[0]
        threshold_idxs = F.pad(distinct_value_indices, [0, 1], value=target.size(0) - 1)
        target = (target == pos_label).to(torch.long)
        tps = _cumsum(target * weight, dim=0)[threshold_idxs]

        if sample_weights is not None:
            # express fps as a cumsum to ensure fps is increasing even in
            # the presence of floating point errors
            fps = _cumsum((1 - target) * weight, dim=0)[threshold_idxs]
        else:
            fps = 1 + threshold_idxs - tps

        return fps, tps, preds[threshold_idxs]


def _adjust_threshold_arg(
    thresholds: Optional[Union[int, List[float], Tensor, str]] = None, device: Optional[torch.device] = None
) -> Optional[Union[Tensor, int]]:
    """Utility function for converting the threshold arg.

    - list and int -> tensor
    - None -> None
    - str -> int (memory budget) in Mb
    """
    if isinstance(thresholds, int):
        thresholds = torch.linspace(0, 1, thresholds, device=device)
    if isinstance(thresholds, list):
        thresholds = torch.tensor(thresholds, device=device)
    if isinstance(thresholds, str):
        thresholds = humanfriendly.parse_size(thresholds, binary=True)
    return thresholds


def _binary_precision_recall_curve_arg_validation(
    thresholds: Optional[Union[int, List[float], Tensor]] = None,
    ignore_index: Optional[int] = None,
) -> None:
    """Validate non tensor input.

    - ``threshold`` has to be None | a 1d tensor | a list of floats in the [0,1] range | an int
    - ``ignore_index`` has to be None or int
    """
    if thresholds is not None and not isinstance(thresholds, (list, int, Tensor)):
        raise ValueError(
            "Expected argument `thresholds` to either be an integer, list of floats or"
            f" tensor of floats, but got {thresholds}"
        )
    else:
        if isinstance(thresholds, int) and thresholds < 2:
            raise ValueError(
                f"If argument `thresholds` is an integer, expected it to be larger than 1, but got {thresholds}"
            )
        if isinstance(thresholds, list) and not all(isinstance(t, float) and 0 <= t <= 1 for t in thresholds):
            raise ValueError(
                "If argument `thresholds` is a list, expected all elements to be floats in the [0,1] range,"
                f" but got {thresholds}"
            )
        if isinstance(thresholds, Tensor) and not thresholds.ndim == 1:
            raise ValueError("If argument `thresholds` is an tensor, expected the tensor to be 1d")

    if ignore_index is not None and not isinstance(ignore_index, int):
        raise ValueError(f"Expected argument `ignore_index` to either be `None` or an integer, but got {ignore_index}")


def _binary_precision_recall_curve_tensor_validation(
    preds: Tensor, target: Tensor, ignore_index: Optional[int] = None
) -> None:
    """Validate tensor input.

    - tensors have to be of same shape
    - all values in target tensor that are not ignored have to be in {0, 1}
    - that the pred tensor is floating point
    """
    _check_same_shape(preds, target)

    if target.is_floating_point():
        raise ValueError(
            "Expected argument `target` to be an int or long tensor with ground truth labels"
            f" but got tensor with dtype {target.dtype}"
        )

    if not preds.is_floating_point():
        raise ValueError(
            "Expected argument `preds` to be an floating tensor with probability/logit scores,"
            f" but got tensor with dtype {preds.dtype}"
        )

    # Check that target only contains {0,1} values or value in ignore_index
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


def _binary_precision_recall_curve_format(
    preds: Tensor,
    target: Tensor,
    thresholds: Optional[Union[int, List[float], Tensor]] = None,
    ignore_index: Optional[int] = None,
) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
    """Convert all input to the right format.

    - flattens additional dimensions
    - Remove all datapoints that should be ignored
    - Applies sigmoid if pred tensor not in [0,1] range
    - Format thresholds arg to be a tensor
    """
    preds = preds.flatten()
    target = target.flatten()
    if ignore_index is not None:
        idx = target != ignore_index
        preds = preds[idx]
        target = target[idx]

    if not torch.all((preds >= 0) * (preds <= 1)):
        preds = preds.sigmoid()

    thresholds = _adjust_threshold_arg(thresholds, preds.device)
    return preds, target, thresholds


def _binary_precision_recall_curve_update(
    preds: Tensor,
    target: Tensor,
    thresholds: Optional[Tensor],
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """Returns the state to calculate the pr-curve with.

    If thresholds is `None` the direct preds and targets are used. If thresholds is not `None` we compute a multi
    threshold confusion matrix.
    """
    if thresholds is None:
        return preds, target
    if preds.numel() <= 50_000:
        update_fn = _binary_precision_recall_curve_update_vectorized
    else:
        update_fn = _binary_precision_recall_curve_update_loop
    return update_fn(preds, target, thresholds)


def _binary_precision_recall_curve_update_vectorized(
    preds: Tensor,
    target: Tensor,
    thresholds: Tensor,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """Returns the multi-threshold confusion matrix to calculate the pr-curve with.

    This implementation is vectorized and faster than `_binary_precision_recall_curve_update_loop` for small
    numbers of samples (up to 50k) but less memory- and time-efficient for more samples.
    """
    len_t = len(thresholds)
    preds_t = (preds.unsqueeze(-1) >= thresholds.unsqueeze(0)).long()  # num_samples x num_thresholds
    unique_mapping = preds_t + 2 * target.unsqueeze(-1) + 4 * torch.arange(len_t, device=target.device)
    bins = _bincount(unique_mapping.flatten(), minlength=4 * len_t)
    return bins.reshape(len_t, 2, 2)


def _binary_precision_recall_curve_update_loop(
    preds: Tensor,
    target: Tensor,
    thresholds: Tensor,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """Returns the multi-threshold confusion matrix to calculate the pr-curve with.

    This implementation loops over thresholds and is more memory-efficient than
    `_binary_precision_recall_curve_update_vectorized`. However, it is slowwer for small
    numbers of samples (up to 50k).
    """
    len_t = len(thresholds)
    target = target == 1
    confmat = thresholds.new_empty((len_t, 2, 2), dtype=torch.int64)
    # Iterate one threshold at a time to conserve memory
    for i in range(len_t):
        preds_t = preds >= thresholds[i]
        confmat[i, 1, 1] = (target & preds_t).sum()
        confmat[i, 0, 1] = ((~target) & preds_t).sum()
        confmat[i, 1, 0] = (target & (~preds_t)).sum()
    confmat[:, 0, 0] = len(preds_t) - confmat[:, 0, 1] - confmat[:, 1, 0] - confmat[:, 1, 1]
    return confmat


def _binary_precision_recall_curve_compute(
    state: Union[Tensor, Tuple[Tensor, Tensor]],
    thresholds: Optional[Tensor],
    pos_label: int = 1,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Compute the final pr-curve.

    If state is a single tensor, then we calculate the pr-curve from a multi threshold confusion matrix. If state is
    original input, then we dynamically compute the binary classification curve.
    """
    if isinstance(state, Tensor):
        tps = state[:, 1, 1]
        fps = state[:, 0, 1]
        fns = state[:, 1, 0]
        precision = _safe_divide(tps, tps + fps)
        recall = _safe_divide(tps, tps + fns)
        precision = torch.cat([precision, torch.ones(1, dtype=precision.dtype, device=precision.device)])
        recall = torch.cat([recall, torch.zeros(1, dtype=recall.dtype, device=recall.device)])
        return precision, recall, thresholds
    else:
        fps, tps, thresholds = _binary_clf_curve(state[0], state[1], pos_label=pos_label)
        precision = tps / (tps + fps)
        recall = tps / tps[-1]

        # stop when full recall attained and reverse the outputs so recall is decreasing
        last_ind = torch.where(tps == tps[-1])[0][0]
        sl = slice(0, last_ind.item() + 1)

        # need to call reversed explicitly, since including that to slice would
        # introduce negative strides that are not yet supported in pytorch
        precision = torch.cat([reversed(precision[sl]), torch.ones(1, dtype=precision.dtype, device=precision.device)])
        recall = torch.cat([reversed(recall[sl]), torch.zeros(1, dtype=recall.dtype, device=recall.device)])
        thresholds = reversed(thresholds[sl]).detach().clone()

    return precision, recall, thresholds


def binary_precision_recall_curve(
    preds: Tensor,
    target: Tensor,
    thresholds: Optional[Union[int, List[float], Tensor]] = None,
    ignore_index: Optional[int] = None,
    validate_args: bool = True,
) -> Tuple[Tensor, Tensor, Tensor]:
    r"""Compute the precision-recall curve for binary tasks. The curve consist of multiple pairs of precision and
    recall values evaluated at different thresholds, such that the tradeoff between the two values can been seen.

    Accepts the following input tensors:

    - ``preds`` (float tensor): ``(N, ...)``. Preds should be a tensor containing probabilities or logits for each
      observation. If preds has values outside [0,1] range we consider the input to be logits and will auto apply
      sigmoid per element.
    - ``target`` (int tensor): ``(N, ...)``. Target should be a tensor containing ground truth labels, and therefore
      only contain {0,1} values (except if `ignore_index` is specified). The value 1 always encodes the positive class.

    Additional dimension ``...`` will be flattened into the batch dimension.

    The implementation both supports calculating the metric in a non-binned but accurate version and a binned version
    that is less accurate but more memory efficient. Setting the `thresholds` argument to `None` will activate the
    non-binned  version that uses memory of size :math:`\mathcal{O}(n_{samples})` whereas setting the `thresholds`
    argument to either an integer, list or a 1d tensor will use a binned version that uses memory of
    size :math:`\mathcal{O}(n_{thresholds})` (constant memory).

    Args:
        preds: Tensor with predictions
        target: Tensor with true labels
        thresholds:
            Can be one of:

            - If set to `None`, will use a non-binned approach where thresholds are dynamically calculated from
              all the data. Most accurate but also most memory consuming approach.
            - If set to an `int` (larger than 1), will use that number of thresholds linearly spaced from
              0 to 1 as bins for the calculation.
            - If set to an `list` of floats, will use the indicated thresholds in the list as bins for the calculation
            - If set to an 1d `tensor` of floats, will use the indicated thresholds in the tensor as
              bins for the calculation.

        ignore_index:
            Specifies a target value that is ignored and does not contribute to the metric calculation
        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.

    Returns:
        (tuple): a tuple of 3 tensors containing:

        - precision: an 1d tensor of size (n_thresholds+1, ) with precision values
        - recall: an 1d tensor of size (n_thresholds+1, ) with recall values
        - thresholds: an 1d tensor of size (n_thresholds, ) with increasing threshold values

    Example:
        >>> from torchmetrics.functional.classification import binary_precision_recall_curve
        >>> preds = torch.tensor([0, 0.5, 0.7, 0.8])
        >>> target = torch.tensor([0, 1, 1, 0])
        >>> binary_precision_recall_curve(preds, target, thresholds=None)  # doctest: +NORMALIZE_WHITESPACE
        (tensor([0.6667, 0.5000, 0.0000, 1.0000]),
         tensor([1.0000, 0.5000, 0.0000, 0.0000]),
         tensor([0.5000, 0.7000, 0.8000]))
        >>> binary_precision_recall_curve(preds, target, thresholds=5)  # doctest: +NORMALIZE_WHITESPACE
        (tensor([0.5000, 0.6667, 0.6667, 0.0000, 0.0000, 1.0000]),
         tensor([1., 1., 1., 0., 0., 0.]),
         tensor([0.0000, 0.2500, 0.5000, 0.7500, 1.0000]))
    """
    if validate_args:
        _binary_precision_recall_curve_arg_validation(thresholds, ignore_index)
        _binary_precision_recall_curve_tensor_validation(preds, target, ignore_index)
    preds, target, thresholds = _binary_precision_recall_curve_format(preds, target, thresholds, ignore_index)
    state = _binary_precision_recall_curve_update(preds, target, thresholds)
    return _binary_precision_recall_curve_compute(state, thresholds)


def _multiclass_precision_recall_curve_arg_validation(
    num_classes: int,
    thresholds: Optional[Union[int, List[float], Tensor]] = None,
    ignore_index: Optional[int] = None,
) -> None:
    """Validate non tensor input.

    - ``num_classes`` has to be an int larger than 1
    - ``threshold`` has to be None | a 1d tensor | a list of floats in the [0,1] range | an int
    - ``ignore_index`` has to be None or int
    """
    if not isinstance(num_classes, int) or num_classes < 2:
        raise ValueError(f"Expected argument `num_classes` to be an integer larger than 1, but got {num_classes}")
    _binary_precision_recall_curve_arg_validation(thresholds, ignore_index)


def _multiclass_precision_recall_curve_tensor_validation(
    preds: Tensor, target: Tensor, num_classes: int, ignore_index: Optional[int] = None
) -> None:
    """Validate tensor input.

    - target should have one more dimension than preds and all dimensions except for preds.shape[1] should match
    exactly. preds.shape[1] should have size equal to number of classes
    - all values in target tensor that are not ignored have to be in {0, 1}
    """
    if not preds.ndim == target.ndim + 1:
        raise ValueError(
            f"Expected `preds` to have one more dimension than `target` but got {preds.ndim} and {target.ndim}"
        )
    if target.is_floating_point():
        raise ValueError(
            f"Expected argument `target` to be an int or long tensor, but got tensor with dtype {target.dtype}"
        )
    if not preds.is_floating_point():
        raise ValueError(f"Expected `preds` to be a float tensor, but got {preds.dtype}")
    if preds.shape[1] != num_classes:
        raise ValueError(
            "Expected `preds.shape[1]` to be equal to the number of classes but"
            f" got {preds.shape[1]} and {num_classes}."
        )
    if preds.shape[0] != target.shape[0] or preds.shape[2:] != target.shape[1:]:
        raise ValueError(
            "Expected the shape of `preds` should be (N, C, ...) and the shape of `target` should be (N, ...)"
            f" but got {preds.shape} and {target.shape}"
        )

    num_unique_values = len(torch.unique(target))
    check = num_unique_values > num_classes if ignore_index is None else num_unique_values > num_classes + 1
    if check:
        raise RuntimeError(
            "Detected more unique values in `target` than `num_classes`. Expected only "
            f"{num_classes if ignore_index is None else num_classes + 1} but found "
            f"{num_unique_values} in `target`."
        )


def _multiclass_precision_recall_curve_format(
    preds: Tensor,
    target: Tensor,
    num_classes: int,
    thresholds: Optional[Union[int, List[float], Tensor]] = None,
    ignore_index: Optional[int] = None,
) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
    """Convert all input to the right format.

    - flattens additional dimensions
    - Remove all datapoints that should be ignored
    - Applies softmax if pred tensor not in [0,1] range
    - Format thresholds arg to be a tensor
    """
    preds = preds.transpose(0, 1).reshape(num_classes, -1).T
    target = target.flatten()

    if ignore_index is not None:
        idx = target != ignore_index
        preds = preds[idx]
        target = target[idx]

    if not torch.all((preds >= 0) * (preds <= 1)):
        preds = preds.softmax(1)

    thresholds = _adjust_threshold_arg(thresholds, preds.device)
    return preds, target, thresholds


def _multiclass_precision_recall_curve_update(
    preds: Tensor,
    target: Tensor,
    num_classes: int,
    thresholds: Optional[Tensor],
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """Returns the state to calculate the pr-curve with.

    If thresholds is `None` the direct preds and targets are used. If thresholds is not `None` we compute a multi
    threshold confusion matrix.
    """
    if thresholds is None:
        return preds, target
    if preds.numel() * num_classes <= 1_000_000:
        update_fn = _multiclass_precision_recall_curve_update_vectorized
    else:
        update_fn = _multiclass_precision_recall_curve_update_loop
    return update_fn(preds, target, num_classes, thresholds)


def _multiclass_precision_recall_curve_update_vectorized(
    preds: Tensor,
    target: Tensor,
    num_classes: int,
    thresholds: Tensor,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """Returns the multi-threshold confusion matrix to calculate the pr-curve with.

    This implementation is vectorized and faster than `_binary_precision_recall_curve_update_loop` for small
    numbers of samples but less memory- and time-efficient for more samples.
    """
    len_t = len(thresholds)
    preds_t = (preds.unsqueeze(-1) >= thresholds.unsqueeze(0).unsqueeze(0)).long()
    target_t = torch.nn.functional.one_hot(target, num_classes=num_classes)
    unique_mapping = preds_t + 2 * target_t.unsqueeze(-1)
    unique_mapping += 4 * torch.arange(num_classes, device=preds.device).unsqueeze(0).unsqueeze(-1)
    unique_mapping += 4 * num_classes * torch.arange(len_t, device=preds.device)
    bins = _bincount(unique_mapping.flatten(), minlength=4 * num_classes * len_t)
    return bins.reshape(len_t, num_classes, 2, 2)


def _multiclass_precision_recall_curve_update_loop(
    preds: Tensor,
    target: Tensor,
    num_classes: int,
    thresholds: Tensor,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """Returns the state to calculate the pr-curve with.

    This implementation loops over thresholds and is more memory-efficient than
    `_binary_precision_recall_curve_update_vectorized`. However, it is slowwer for small
    numbers of samples.
    """
    len_t = len(thresholds)
    target_t = torch.nn.functional.one_hot(target, num_classes=num_classes)
    confmat = thresholds.new_empty((len_t, num_classes, 2, 2), dtype=torch.int64)
    # Iterate one threshold at a time to conserve memory
    for i in range(len_t):
        preds_t = preds >= thresholds[i]
        confmat[i, :, 1, 1] = (target_t & preds_t).sum(dim=0)
        confmat[i, :, 0, 1] = ((~target_t) & preds_t).sum(dim=0)
        confmat[i, :, 1, 0] = (target_t & (~preds_t)).sum(dim=0)
    confmat[:, :, 0, 0] = len(preds_t) - confmat[:, :, 0, 1] - confmat[:, :, 1, 0] - confmat[:, :, 1, 1]
    return confmat


def _multiclass_precision_recall_curve_compute(
    state: Union[Tensor, Tuple[Tensor, Tensor]],
    num_classes: int,
    thresholds: Optional[Tensor],
) -> Union[Tuple[Tensor, Tensor, Tensor], Tuple[List[Tensor], List[Tensor], List[Tensor]]]:
    """Compute the final pr-curve.

    If state is a single tensor, then we calculate the pr-curve from a multi threshold confusion matrix. If state is
    original input, then we dynamically compute the binary classification curve in an iterative way.
    """
    if isinstance(state, Tensor):
        tps = state[:, :, 1, 1]
        fps = state[:, :, 0, 1]
        fns = state[:, :, 1, 0]
        precision = _safe_divide(tps, tps + fps)
        recall = _safe_divide(tps, tps + fns)
        precision = torch.cat([precision, torch.ones(1, num_classes, dtype=precision.dtype, device=precision.device)])
        recall = torch.cat([recall, torch.zeros(1, num_classes, dtype=recall.dtype, device=recall.device)])
        return precision.T, recall.T, thresholds
    else:
        precision, recall, thresholds = [], [], []
        for i in range(num_classes):
            res = _binary_precision_recall_curve_compute([state[0][:, i], state[1]], thresholds=None, pos_label=i)
            precision.append(res[0])
            recall.append(res[1])
            thresholds.append(res[2])
    return precision, recall, thresholds


def multiclass_precision_recall_curve(
    preds: Tensor,
    target: Tensor,
    num_classes: int,
    thresholds: Optional[Union[int, List[float], Tensor]] = None,
    ignore_index: Optional[int] = None,
    validate_args: bool = True,
) -> Union[Tuple[Tensor, Tensor, Tensor], Tuple[List[Tensor], List[Tensor], List[Tensor]]]:
    r"""Compute the precision-recall curve for multiclass tasks. The curve consist of multiple pairs of precision
    and recall values evaluated at different thresholds, such that the tradeoff between the two values can been
    seen.

    Accepts the following input tensors:

    - ``preds`` (float tensor): ``(N, C, ...)``. Preds should be a tensor containing probabilities or logits for each
      observation. If preds has values outside [0,1] range we consider the input to be logits and will auto apply
      softmax per sample.
    - ``target`` (int tensor): ``(N, ...)``. Target should be a tensor containing ground truth labels, and therefore
      only contain values in the [0, n_classes-1] range (except if `ignore_index` is specified).

    Additional dimension ``...`` will be flattened into the batch dimension.

    The implementation both supports calculating the metric in a non-binned but accurate version and a binned version
    that is less accurate but more memory efficient. Setting the `thresholds` argument to `None` will activate the
    non-binned  version that uses memory of size :math:`\mathcal{O}(n_{samples})` whereas setting the `thresholds`
    argument to either an integer, list or a 1d tensor will use a binned version that uses memory of
    size :math:`\mathcal{O}(n_{thresholds} \times n_{classes})` (constant memory).

    Args:
        preds: Tensor with predictions
        target: Tensor with true labels
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

        ignore_index:
            Specifies a target value that is ignored and does not contribute to the metric calculation
        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.

    Returns:
        (tuple): a tuple of either 3 tensors or 3 lists containing

        - precision: if `thresholds=None` a list for each class is returned with an 1d tensor of size (n_thresholds+1, )
          with precision values (length may differ between classes). If `thresholds` is set to something else,
          then a single 2d tensor of size (n_classes, n_thresholds+1) with precision values is returned.
        - recall: if `thresholds=None` a list for each class is returned with an 1d tensor of size (n_thresholds+1, )
          with recall values (length may differ between classes). If `thresholds` is set to something else,
          then a single 2d tensor of size (n_classes, n_thresholds+1) with recall values is returned.
        - thresholds: if `thresholds=None` a list for each class is returned with an 1d tensor of size (n_thresholds, )
          with increasing threshold values (length may differ between classes). If `threshold` is set to something else,
          then a single 1d tensor of size (n_thresholds, ) is returned with shared threshold values for all classes.

    Example:
        >>> from torchmetrics.functional.classification import multiclass_precision_recall_curve
        >>> preds = torch.tensor([[0.75, 0.05, 0.05, 0.05, 0.05],
        ...                       [0.05, 0.75, 0.05, 0.05, 0.05],
        ...                       [0.05, 0.05, 0.75, 0.05, 0.05],
        ...                       [0.05, 0.05, 0.05, 0.75, 0.05]])
        >>> target = torch.tensor([0, 1, 3, 2])
        >>> precision, recall, thresholds = multiclass_precision_recall_curve(
        ...    preds, target, num_classes=5, thresholds=None
        ... )
        >>> precision  # doctest: +NORMALIZE_WHITESPACE
        [tensor([1., 1.]), tensor([1., 1.]), tensor([0.2500, 0.0000, 1.0000]),
         tensor([0.2500, 0.0000, 1.0000]), tensor([0., 1.])]
        >>> recall
        [tensor([1., 0.]), tensor([1., 0.]), tensor([1., 0., 0.]), tensor([1., 0., 0.]), tensor([nan, 0.])]
        >>> thresholds
        [tensor([0.7500]), tensor([0.7500]), tensor([0.0500, 0.7500]), tensor([0.0500, 0.7500]), tensor([0.0500])]
        >>> multiclass_precision_recall_curve(
        ...     preds, target, num_classes=5, thresholds=5
        ... )  # doctest: +NORMALIZE_WHITESPACE
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
    if validate_args:
        _multiclass_precision_recall_curve_arg_validation(num_classes, thresholds, ignore_index)
        _multiclass_precision_recall_curve_tensor_validation(preds, target, num_classes, ignore_index)
    preds, target, thresholds = _multiclass_precision_recall_curve_format(
        preds, target, num_classes, thresholds, ignore_index
    )
    state = _multiclass_precision_recall_curve_update(preds, target, num_classes, thresholds)
    return _multiclass_precision_recall_curve_compute(state, num_classes, thresholds)


def _multilabel_precision_recall_curve_arg_validation(
    num_labels: int,
    thresholds: Optional[Union[int, List[float], Tensor]] = None,
    ignore_index: Optional[int] = None,
) -> None:
    """Validate non tensor input.

    - ``num_labels`` has to be an int larger than 1
    - ``threshold`` has to be None | a 1d tensor | a list of floats in the [0,1] range | an int
    - ``ignore_index`` has to be None or int
    """
    _multiclass_precision_recall_curve_arg_validation(num_labels, thresholds, ignore_index)


def _multilabel_precision_recall_curve_tensor_validation(
    preds: Tensor, target: Tensor, num_labels: int, ignore_index: Optional[int] = None
) -> None:
    """Validate tensor input.

    - tensors have to be of same shape
    - preds.shape[1] is equal to the number of labels
    - all values in target tensor that are not ignored have to be in {0, 1}
    - that the pred tensor is floating point
    """
    _binary_precision_recall_curve_tensor_validation(preds, target, ignore_index)
    if preds.shape[1] != num_labels:
        raise ValueError(
            "Expected both `target.shape[1]` and `preds.shape[1]` to be equal to the number of labels"
            f" but got {preds.shape[1]} and expected {num_labels}"
        )


def _multilabel_precision_recall_curve_format(
    preds: Tensor,
    target: Tensor,
    num_labels: int,
    thresholds: Optional[Union[int, List[float], Tensor]] = None,
    ignore_index: Optional[int] = None,
) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
    """Convert all input to the right format.

    - flattens additional dimensions
    - Mask all datapoints that should be ignored with negative values
    - Applies sigmoid if pred tensor not in [0,1] range
    - Format thresholds arg to be a tensor
    """
    preds = preds.transpose(0, 1).reshape(num_labels, -1).T
    target = target.transpose(0, 1).reshape(num_labels, -1).T
    if not torch.all((preds >= 0) * (preds <= 1)):
        preds = preds.sigmoid()

    thresholds = _adjust_threshold_arg(thresholds, preds.device)
    if ignore_index is not None and thresholds is not None:
        preds = preds.clone()
        target = target.clone()
        # Make sure that when we map, it will always result in a negative number that we can filter away
        idx = target == ignore_index
        preds[idx] = -4 * num_labels * (len(thresholds) if thresholds is not None else 1)
        target[idx] = -4 * num_labels * (len(thresholds) if thresholds is not None else 1)

    return preds, target, thresholds


def _multilabel_precision_recall_curve_update(
    preds: Tensor,
    target: Tensor,
    num_labels: int,
    thresholds: Optional[Tensor],
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """Returns the state to calculate the pr-curve with.

    If thresholds is `None` the direct preds and targets are used. If thresholds is not `None` we compute a multi
    threshold confusion matrix.
    """
    if thresholds is None:
        return preds, target
    len_t = len(thresholds)
    # num_samples x num_labels x num_thresholds
    preds_t = (preds.unsqueeze(-1) >= thresholds.unsqueeze(0).unsqueeze(0)).long()
    unique_mapping = preds_t + 2 * target.unsqueeze(-1)
    unique_mapping += 4 * torch.arange(num_labels, device=preds.device).unsqueeze(0).unsqueeze(-1)
    unique_mapping += 4 * num_labels * torch.arange(len_t, device=preds.device)
    unique_mapping = unique_mapping[unique_mapping >= 0]
    bins = _bincount(unique_mapping, minlength=4 * num_labels * len_t)
    return bins.reshape(len_t, num_labels, 2, 2)


def _multilabel_precision_recall_curve_compute(
    state: Union[Tensor, Tuple[Tensor, Tensor]],
    num_labels: int,
    thresholds: Optional[Tensor],
    ignore_index: Optional[int] = None,
) -> Union[Tuple[Tensor, Tensor, Tensor], Tuple[List[Tensor], List[Tensor], List[Tensor]]]:
    """Compute the final pr-curve.

    If state is a single tensor, then we calculate the pr-curve from a multi threshold confusion matrix. If state is
    original input, then we dynamically compute the binary classification curve in an iterative way.
    """
    if isinstance(state, Tensor):
        tps = state[:, :, 1, 1]
        fps = state[:, :, 0, 1]
        fns = state[:, :, 1, 0]
        precision = _safe_divide(tps, tps + fps)
        recall = _safe_divide(tps, tps + fns)
        precision = torch.cat([precision, torch.ones(1, num_labels, dtype=precision.dtype, device=precision.device)])
        recall = torch.cat([recall, torch.zeros(1, num_labels, dtype=recall.dtype, device=recall.device)])
        return precision.T, recall.T, thresholds
    else:
        precision, recall, thresholds = [], [], []
        for i in range(num_labels):
            preds = state[0][:, i]
            target = state[1][:, i]
            if ignore_index is not None:
                idx = target == ignore_index
                preds = preds[~idx]
                target = target[~idx]
            res = _binary_precision_recall_curve_compute([preds, target], thresholds=None, pos_label=1)
            precision.append(res[0])
            recall.append(res[1])
            thresholds.append(res[2])
    return precision, recall, thresholds


def multilabel_precision_recall_curve(
    preds: Tensor,
    target: Tensor,
    num_labels: int,
    thresholds: Optional[Union[int, List[float], Tensor]] = None,
    ignore_index: Optional[int] = None,
    validate_args: bool = True,
) -> Union[Tuple[Tensor, Tensor, Tensor], Tuple[List[Tensor], List[Tensor], List[Tensor]]]:
    r"""Compute the precision-recall curve for multilabel tasks. The curve consist of multiple pairs of precision
    and recall values evaluated at different thresholds, such that the tradeoff between the two values can been
    seen.

    Accepts the following input tensors:

    - ``preds`` (float tensor): ``(N, C, ...)``. Preds should be a tensor containing probabilities or logits for each
      observation. If preds has values outside [0,1] range we consider the input to be logits and will auto apply
      sigmoid per element.
    - ``target`` (int tensor): ``(N, C, ...)``. Target should be a tensor containing ground truth labels, and therefore
      only contain {0,1} values (except if `ignore_index` is specified).

    Additional dimension ``...`` will be flattened into the batch dimension.

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

        ignore_index:
            Specifies a target value that is ignored and does not contribute to the metric calculation
        validate_args: bool indicating if input arguments and tensors should be validated for correctness.
            Set to ``False`` for faster computations.

    Returns:
        (tuple): a tuple of either 3 tensors or 3 lists containing

        - precision: if `thresholds=None` a list for each label is returned with an 1d tensor of size (n_thresholds+1, )
          with precision values (length may differ between labels). If `thresholds` is set to something else,
          then a single 2d tensor of size (n_labels, n_thresholds+1) with precision values is returned.
        - recall: if `thresholds=None` a list for each label is returned with an 1d tensor of size (n_thresholds+1, )
          with recall values (length may differ between labels). If `thresholds` is set to something else,
          then a single 2d tensor of size (n_labels, n_thresholds+1) with recall values is returned.
        - thresholds: if `thresholds=None` a list for each label is returned with an 1d tensor of size (n_thresholds, )
          with increasing threshold values (length may differ between labels). If `threshold` is set to something else,
          then a single 1d tensor of size (n_thresholds, ) is returned with shared threshold values for all labels.

    Example:
        >>> from torchmetrics.functional.classification import multilabel_precision_recall_curve
        >>> preds = torch.tensor([[0.75, 0.05, 0.35],
        ...                       [0.45, 0.75, 0.05],
        ...                       [0.05, 0.55, 0.75],
        ...                       [0.05, 0.65, 0.05]])
        >>> target = torch.tensor([[1, 0, 1],
        ...                        [0, 0, 0],
        ...                        [0, 1, 1],
        ...                        [1, 1, 1]])
        >>> precision, recall, thresholds = multilabel_precision_recall_curve(
        ...    preds, target, num_labels=3, thresholds=None
        ... )
        >>> precision  # doctest: +NORMALIZE_WHITESPACE
        [tensor([0.5000, 0.5000, 1.0000, 1.0000]), tensor([0.6667, 0.5000, 0.0000, 1.0000]),
         tensor([0.7500, 1.0000, 1.0000, 1.0000])]
        >>> recall  # doctest: +NORMALIZE_WHITESPACE
        [tensor([1.0000, 0.5000, 0.5000, 0.0000]), tensor([1.0000, 0.5000, 0.0000, 0.0000]),
         tensor([1.0000, 0.6667, 0.3333, 0.0000])]
        >>> thresholds  # doctest: +NORMALIZE_WHITESPACE
        [tensor([0.0500, 0.4500, 0.7500]), tensor([0.5500, 0.6500, 0.7500]),
         tensor([0.0500, 0.3500, 0.7500])]
        >>> multilabel_precision_recall_curve(
        ...     preds, target, num_labels=3, thresholds=5
        ... )  # doctest: +NORMALIZE_WHITESPACE
        (tensor([[0.5000, 0.5000, 1.0000, 1.0000, 0.0000, 1.0000],
                 [0.5000, 0.6667, 0.6667, 0.0000, 0.0000, 1.0000],
                 [0.7500, 1.0000, 1.0000, 1.0000, 0.0000, 1.0000]]),
         tensor([[1.0000, 0.5000, 0.5000, 0.5000, 0.0000, 0.0000],
                 [1.0000, 1.0000, 1.0000, 0.0000, 0.0000, 0.0000],
                 [1.0000, 0.6667, 0.3333, 0.3333, 0.0000, 0.0000]]),
         tensor([0.0000, 0.2500, 0.5000, 0.7500, 1.0000]))
    """
    if validate_args:
        _multilabel_precision_recall_curve_arg_validation(num_labels, thresholds, ignore_index)
        _multilabel_precision_recall_curve_tensor_validation(preds, target, num_labels, ignore_index)
    preds, target, thresholds = _multilabel_precision_recall_curve_format(
        preds, target, num_labels, thresholds, ignore_index
    )
    state = _multilabel_precision_recall_curve_update(preds, target, num_labels, thresholds)
    return _multilabel_precision_recall_curve_compute(state, num_labels, thresholds, ignore_index)


def precision_recall_curve(
    preds: Tensor,
    target: Tensor,
    task: Literal["binary", "multiclass", "multilabel"],
    thresholds: Optional[Union[int, List[float], Tensor]] = None,
    num_classes: Optional[int] = None,
    num_labels: Optional[int] = None,
    ignore_index: Optional[int] = None,
    validate_args: bool = True,
) -> Union[Tuple[Tensor, Tensor, Tensor], Tuple[List[Tensor], List[Tensor], List[Tensor]]]:
    r"""Compute the precision-recall curve. The curve consist of multiple pairs of precision and recall values
    evaluated at different thresholds, such that the tradeoff between the two values can been seen.

    This function is a simple wrapper to get the task specific versions of this metric, which is done by setting the
    ``task`` argument to either ``'binary'``, ``'multiclass'`` or ``multilabel``. See the documentation of
    :func:`binary_precision_recall_curve`, :func:`multiclass_precision_recall_curve` and
    :func:`multilabel_precision_recall_curve` for the specific details of each argument influence and examples.

    Legacy Example:
        >>> pred = torch.tensor([0.0, 1.0, 2.0, 3.0])
        >>> target = torch.tensor([0, 1, 1, 0])
        >>> precision, recall, thresholds = precision_recall_curve(pred, target, task='binary')
        >>> precision
        tensor([0.6667, 0.5000, 0.0000, 1.0000])
        >>> recall
        tensor([1.0000, 0.5000, 0.0000, 0.0000])
        >>> thresholds
        tensor([0.7311, 0.8808, 0.9526])

        >>> pred = torch.tensor([[0.75, 0.05, 0.05, 0.05, 0.05],
        ...                      [0.05, 0.75, 0.05, 0.05, 0.05],
        ...                      [0.05, 0.05, 0.75, 0.05, 0.05],
        ...                      [0.05, 0.05, 0.05, 0.75, 0.05]])
        >>> target = torch.tensor([0, 1, 3, 2])
        >>> precision, recall, thresholds = precision_recall_curve(pred, target, task='multiclass', num_classes=5)
        >>> precision
        [tensor([1., 1.]), tensor([1., 1.]), tensor([0.2500, 0.0000, 1.0000]),
         tensor([0.2500, 0.0000, 1.0000]), tensor([0., 1.])]
        >>> recall
        [tensor([1., 0.]), tensor([1., 0.]), tensor([1., 0., 0.]), tensor([1., 0., 0.]), tensor([nan, 0.])]
        >>> thresholds
        [tensor([0.7500]), tensor([0.7500]), tensor([0.0500, 0.7500]), tensor([0.0500, 0.7500]), tensor([0.0500])]
    """
    task = ClassificationTask.from_str(task)
    if task == ClassificationTask.BINARY:
        return binary_precision_recall_curve(preds, target, thresholds, ignore_index, validate_args)
    if task == ClassificationTask.MULTICLASS:
        assert isinstance(num_classes, int)
        return multiclass_precision_recall_curve(preds, target, num_classes, thresholds, ignore_index, validate_args)
    if task == ClassificationTask.MULTILABEL:
        assert isinstance(num_labels, int)
        return multilabel_precision_recall_curve(preds, target, num_labels, thresholds, ignore_index, validate_args)
