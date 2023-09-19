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
import math
from typing import Optional, Tuple

import torch
from torch import Tensor

from torchmetrics.functional.regression.utils import _check_data_shape_to_num_outputs, _check_data_shape_to_weights
from torchmetrics.utilities import rank_zero_warn
from torchmetrics.utilities.checks import _check_same_shape


def _pearson_corrcoef_update(
    preds: Tensor,
    target: Tensor,
    mean_x: Tensor,
    mean_y: Tensor,
    var_x: Tensor,
    var_y: Tensor,
    corr_xy: Tensor,
    n_prior: Tensor,
    num_outputs: int,
    weights: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Update and returns variables required to compute Pearson Correlation Coefficient.

    Check for same shape of input tensors.

    Args:
        preds: estimated scores
        target: ground truth scores
        mean_x: current mean estimate of x tensor
        mean_y: current mean estimate of y tensor
        var_x: current variance estimate of x tensor
        var_y: current variance estimate of y tensor
        corr_xy: current covariance estimate between x and y tensor
        n_prior: current number of observed observations
        num_outputs: Number of outputs in multioutput setting
        weights: weights associated with scores

    """
    # Data checking
    _check_same_shape(preds, target)
    _check_data_shape_to_num_outputs(preds, target, num_outputs)
    if weights is not None:
        _check_data_shape_to_weights(preds, weights)

    n_obs = preds.shape[0] if weights is None else weights.sum()
    cond = n_prior.mean() > 0 or n_obs == 1

    if cond:
        if weights is None:
            mx_new = (n_prior * mean_x + preds.sum(0)) / (n_prior + n_obs)
            my_new = (n_prior * mean_y + target.sum(0)) / (n_prior + n_obs)
        else:
            mx_new = (n_prior * mean_x + torch.matmul(weights, preds)) / (n_prior + n_obs)
            my_new = (n_prior * mean_y + torch.matmul(weights, target)) / (n_prior + n_obs)
    else:
        if weights is None:
            mx_new = preds.mean(0)
            my_new = target.mean(0)
        else:
            mx_new = torch.matmul(weights, preds) / weights.sum()
            my_new = torch.matmul(weights, target) / weights.sum()

    n_prior += n_obs

    # Calculate variances
    if cond:
        if weights is None:
            var_x += ((preds - mx_new) * (preds - mean_x)).sum(0)
            var_y += ((target - my_new) * (target - mean_y)).sum(0)
        else:
            var_x += torch.matmul(weights, (preds - mx_new) * (preds - mean_x))
            var_y += torch.matmul(weights, (preds - my_new) * (preds - mean_y))
    else:
        if weights is None:
            var_x += preds.var(0) * (n_obs - 1)
            var_y += target.var(0) * (n_obs - 1)
        else:
            var_x += torch.matmul(weights, (preds - mx_new) ** 2)
            var_y += torch.matmul(weights, (target - my_new) ** 2)

    if weights is None:
        corr_xy += ((preds - mx_new) * (target - my_new)).sum(0)
    else:
        corr_xy += torch.matmul(weights, (preds - mx_new) * (target - my_new))

    return mx_new, my_new, var_x, var_y, corr_xy, n_prior


def _pearson_corrcoef_compute(
    var_x: Tensor,
    var_y: Tensor,
    corr_xy: Tensor,
    nb: Tensor,
) -> Tensor:
    """Compute the final pearson correlation based on accumulated statistics.

    Args:
        var_x: variance estimate of x tensor
        var_y: variance estimate of y tensor
        corr_xy: covariance estimate between x and y tensor
        nb: number of observations

    """
    # if var_x, var_y is float16 and on cpu, make it bfloat16 as sqrt is not supported for float16
    # on cpu, remove this after https://github.com/pytorch/pytorch/issues/54774 is fixed
    if var_x.dtype == torch.float16 and var_x.device == torch.device("cpu"):
        var_x = var_x.bfloat16()
        var_y = var_y.bfloat16()

    bound = math.sqrt(torch.finfo(var_x.dtype).eps)
    if (var_x < bound).any() or (var_y < bound).any():
        rank_zero_warn(
            "The variance of predictions or target is close to zero. This can cause instability in Pearson correlation"
            "coefficient, leading to wrong results. Consider re-scaling the input if possible or computing using a"
            f"larger dtype (currently using {var_x.dtype}).",
            UserWarning,
        )

    corrcoef = (corr_xy / (var_x * var_y).sqrt()).squeeze()
    return torch.clamp(corrcoef, -1.0, 1.0)


def pearson_corrcoef(preds: Tensor, target: Tensor, weights: Optional[Tensor] = None) -> Tensor:
    """Compute pearson correlation coefficient.

    Args:
        preds: torch.Tensor of shape (n_samples,) or (n_samples, n_outputs)
            Estimated scores
        target: torch.Tensor of shape (n_samples,) or (n_samples, n_outputs)
            Ground truth scores
        weights: torch.Tensor of shape (n_samples,), default=None
            Sample weights

    Example (single output regression):
        >>> from torchmetrics.functional.regression import pearson_corrcoef
        >>> target = torch.tensor([3, -0.5, 2, 7])
        >>> preds = torch.tensor([2.5, 0.0, 2, 8])
        >>> pearson_corrcoef(preds, target)
        tensor(0.9849)

    Example (weighted single output regression):
        >>> from torchmetrics.functional.regression import pearson_corrcoef
        >>> target = torch.tensor([3, -0.5, 2, 7])
        >>> preds = torch.tensor([2.5, 0.0, 2, 8])
        >>> weights = torch.tensor([2.5, 0.0, 2, 8])
        >>> pearson_corrcoef(preds, target, weights)
        tensor(0.9849)

    Example (multi output regression):
        >>> from torchmetrics.functional.regression import pearson_corrcoef
        >>> target = torch.tensor([[3, -0.5], [2, 7]])
        >>> preds = torch.tensor([[2.5, 0.0], [2, 8]])
        >>> pearson_corrcoef(preds, target)
        tensor([1., 1.])

    Example (weighted multiple output regression):
        >>> from torchmetrics.functional.regression import pearson_corrcoef
        >>> target = torch.tensor([3, -0.5, 2, 7])
        >>> preds = torch.tensor([2.5, 0.0, 2, 8])
        >>> weights = torch.tensor([2.5, 0.0, 2, 8])
        >>> pearson_corrcoef(preds, target, weights)
        tensor(0.9849)

    """
    d = preds.shape[1] if preds.ndim == 2 else 1
    _temp = torch.zeros(d, dtype=preds.dtype, device=preds.device)
    mean_x, mean_y, var_x = _temp.clone(), _temp.clone(), _temp.clone()
    var_y, corr_xy, nb = _temp.clone(), _temp.clone(), _temp.clone()
    _, _, var_x, var_y, corr_xy, nb = _pearson_corrcoef_update(
        preds,
        target,
        mean_x,
        mean_y,
        var_x,
        var_y,
        corr_xy,
        nb,
        num_outputs=1 if preds.ndim == 1 else preds.shape[-1],
        weights=weights,
    )
    return _pearson_corrcoef_compute(var_x, var_y, corr_xy, nb)
