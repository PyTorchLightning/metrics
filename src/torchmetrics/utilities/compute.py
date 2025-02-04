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
from torch import Tensor
from typing_extensions import Literal

from torchmetrics.utilities import rank_zero_warn


def _safe_matmul(x: Tensor, y: Tensor) -> Tensor:
    """Safe calculation of matrix multiplication.

    If input is float16, will cast to float32 for computation and back again.

    """
    if x.dtype == torch.float16 or y.dtype == torch.float16:
        return (x.float() @ y.T.float()).half()
    return x @ y.T


def _safe_xlogy(x: Tensor, y: Tensor) -> Tensor:
    """Compute x * log(y). Returns 0 if x=0.

    Example:
        >>> import torch
        >>> x = torch.zeros(1)
        >>> _safe_xlogy(x, 1/x)
        tensor([0.])

    """
    res = x * torch.log(y)
    res[x == 0] = 0.0
    return res


def _safe_divide(
    num: Tensor,
    denom: Tensor,
    zero_division: Union[float, Literal["warn", "nan"]] = 0.0,
) -> Tensor:
    """Safe division, by preventing division by zero.

    Function will cast to float if input is not already to secure backwards compatibility.

    Args:
        num: numerator tensor
        denom: denominator tensor, which may contain zeros
        zero_division: value to replace elements divided by zero

    Example:
        >>> import torch
        >>> num = torch.tensor([1.0, 2.0, 3.0])
        >>> denom = torch.tensor([0.0, 1.0, 2.0])
        >>> _safe_divide(num, denom)
        tensor([0.0000, 2.0000, 1.5000])

    """
    num = num if num.is_floating_point() else num.float()
    denom = denom if denom.is_floating_point() else denom.float()
    if isinstance(zero_division, float) or zero_division == "warn":
        if zero_division == "warn" and torch.any(denom == 0):
            rank_zero_warn("Detected zero division in _safe_divide. Setting 0/0 to 0.0")
        zero_division = 0.0 if zero_division == "warn" else zero_division
        zero_division_tensor = torch.tensor(zero_division, dtype=num.dtype).to(num.device, non_blocking=True)
        return torch.where(denom != 0, num / denom, zero_division_tensor)
    return torch.true_divide(num, denom)


def _adjust_weights_safe_divide(
    score: Tensor, average: Optional[str], multilabel: bool, tp: Tensor, fp: Tensor, fn: Tensor, top_k: int = 1
) -> Tensor:
    if average is None or average == "none":
        return score
    if average == "weighted":
        weights = tp + fn
    else:
        weights = torch.ones_like(score)
        if not multilabel:
            weights[tp + fp + fn == 0 if top_k == 1 else tp + fn == 0] = 0.0
    return _safe_divide(weights * score, weights.sum(-1, keepdim=True)).sum(-1)


def _auc_format_inputs(x: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
    """Check that auc input is correct."""
    x = x.squeeze() if x.ndim > 1 else x
    y = y.squeeze() if y.ndim > 1 else y

    if x.ndim > 1 or y.ndim > 1:
        raise ValueError(
            f"Expected both `x` and `y` tensor to be 1d, but got tensors with dimension {x.ndim} and {y.ndim}"
        )
    if x.numel() != y.numel():
        raise ValueError(
            f"Expected the same number of elements in `x` and `y` tensor but received {x.numel()} and {y.numel()}"
        )
    return x, y


def _auc_compute_without_check(x: Tensor, y: Tensor, direction: float, axis: int = -1) -> Tensor:
    """Compute area under the curve using the trapezoidal rule.

    Assumes increasing or decreasing order of `x`.

    """
    with torch.no_grad():
        auc_score: Tensor = torch.trapz(y, x, dim=axis) * direction
    return auc_score


def _auc_compute(x: Tensor, y: Tensor, reorder: bool = False) -> Tensor:
    """Compute area under the curve using the trapezoidal rule.

    Example:
        >>> import torch
        >>> x = torch.tensor([1, 2, 3, 4])
        >>> y = torch.tensor([1, 2, 3, 4])
        >>> _auc_compute(x, y)
        tensor(7.5000)

    """
    with torch.no_grad():
        if reorder:
            x, x_idx = torch.sort(x, stable=True)
            y = y[x_idx]

        dx = x[1:] - x[:-1]
        if (dx < 0).any():
            if (dx <= 0).all():
                direction = -1.0
            else:
                raise ValueError(
                    "The `x` tensor is neither increasing or decreasing. Try setting the reorder argument to `True`."
                )
        else:
            direction = 1.0
        return _auc_compute_without_check(x, y, direction)


def auc(x: Tensor, y: Tensor, reorder: bool = False) -> Tensor:
    """Compute Area Under the Curve (AUC) using the trapezoidal rule.

    Args:
        x: x-coordinates, must be either increasing or decreasing
        y: y-coordinates
        reorder: if True, will reorder the arrays to make it either increasing or decreasing

    Return:
        Tensor containing AUC score

    """
    x, y = _auc_format_inputs(x, y)
    return _auc_compute(x, y, reorder=reorder)


def interp(x: Tensor, xp: Tensor, fp: Tensor) -> Tensor:
    """One-dimensional linear interpolation for monotonically increasing sample points.

    Returns the one-dimensional piecewise linear interpolation to a function with
    given discrete data points :math:`(xp, fp)`, evaluated at :math:`x`.

    Adjusted version of this https://github.com/pytorch/pytorch/issues/50334#issuecomment-1000917964

    Args:
        x: the :math:`x`-coordinates at which to evaluate the interpolated values.
        xp: the :math:`x`-coordinates of the data points, must be increasing.
        fp: the :math:`y`-coordinates of the data points, same length as `xp`.

    Returns:
        the interpolated values, same size as `x`.

    Example:
        >>> x = torch.tensor([0.5, 1.5, 2.5])
        >>> xp = torch.tensor([1, 2, 3])
        >>> fp = torch.tensor([1, 2, 3])
        >>> interp(x, xp, fp)
        tensor([0.5000, 1.5000, 2.5000])

    """
    m = _safe_divide(fp[1:] - fp[:-1], xp[1:] - xp[:-1])
    b = fp[:-1] - (m * xp[:-1])

    indices = torch.sum(torch.ge(x[:, None], xp[None, :]), 1) - 1
    indices = torch.clamp(indices, 0, len(m) - 1)

    return m[indices] * x + b[indices]


def normalize_logits_if_needed(tensor: Tensor, normalization: Literal["sigmoid", "softmax"]) -> Tensor:
    """Normalize logits if needed.

    If input tensor is outside the [0,1] we assume that logits are provided and apply the normalization.
    Use torch.where to prevent device-host sync.

    Args:
        tensor: input tensor that may be logits or probabilities
        normalization: normalization method, either 'sigmoid' or 'softmax'

    Returns:
        normalized tensor if needed

    Example:
        >>> import torch
        >>> tensor = torch.tensor([-1.0, 0.0, 1.0])
        >>> normalize_logits_if_needed(tensor, normalization="sigmoid")
        tensor([0.2689, 0.5000, 0.7311])
        >>> tensor = torch.tensor([[-1.0, 0.0, 1.0], [1.0, 0.0, -1.0]])
        >>> normalize_logits_if_needed(tensor, normalization="softmax")
        tensor([[0.0900, 0.2447, 0.6652],
                [0.6652, 0.2447, 0.0900]])
        >>> tensor = torch.tensor([0.0, 0.5, 1.0])
        >>> normalize_logits_if_needed(tensor, normalization="sigmoid")
        tensor([0.0000, 0.5000, 1.0000])

    """
    # decrease sigmoid on cpu .
    if tensor.device == torch.device("cpu"):
        if not torch.all((tensor >= 0) * (tensor <= 1)):
            tensor = tensor.sigmoid() if normalization == "sigmoid" else torch.softmax(tensor, dim=1)
        return tensor

    # decrease device-host sync on device .
    condition = ((tensor < 0) | (tensor > 1)).any()
    return torch.where(
        condition,
        torch.sigmoid(tensor) if normalization == "sigmoid" else torch.softmax(tensor, dim=1),
        tensor,
    )
