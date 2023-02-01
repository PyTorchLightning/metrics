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
from typing import Optional, Union

import torch
from torch import Tensor
from typing_extensions import Literal

from torchmetrics.functional.pairwise.helpers import _check_input, _reduce_distance_matrix


def _pairwise_minkowski_distance_update(
    x: Tensor, y: Optional[Tensor] = None, p: Union[int, float] = 2, zero_diagonal: Optional[bool] = None
) -> Tensor:
    """Calculates the pairwise minkowski distance matrix.

    Args:
        x: tensor of shape ``[N,d]``
        y: tensor of shape ``[M,d]``
        zero_diagonal: determines if the diagonal of the distance matrix should be set to zero
    """
    x, y, zero_diagonal = _check_input(x, y, zero_diagonal)
    # upcast to float64 to prevent precision issues
    _orig_dtype = x.dtype
    x = x.to(torch.float64)
    y = y.to(torch.float64)
    distance = (x.unsqueeze(1) - y.unsqueeze(0)).abs().pow(p).sum(-1).pow(1.0 / p)
    if zero_diagonal:
        distance.fill_diagonal_(0)
    return distance.to(_orig_dtype)


def pairwise_minkowski_distance(
    x: Tensor,
    y: Optional[Tensor] = None,
    p: Union[int, float] = 2,
    reduction: Literal["mean", "sum", "none", None] = None,
    zero_diagonal: Optional[bool] = None,
) -> Tensor:
    r"""Calculates pairwise minkowski distances:

    .. math::
        d_{minkowski}(x,y,p) = ||x - y||_p = \sqrt[p]{\sum_{d=1}^D (x_d - y_d)^p}

    If both :math:`x` and :math:`y` are passed in, the calculation will be performed pairwise between the rows of
    :math:`x` and :math:`y`. If only :math:`x` is passed in, the calculation will be performed between the rows
    of :math:`x`.

    Args:
        x: Tensor with shape ``[N, d]``
        y: Tensor with shape ``[M, d]``, optional
        p: Integer or float to use for the exponents in the calculation
        reduction: reduction to apply along the last dimension. Choose between `'mean'`, `'sum'`
            (applied along column dimension) or  `'none'`, `None` for no reduction
        zero_diagonal: if the diagonal of the distance matrix should be set to 0. If only `x` is given
            this defaults to `True` else if `y` is also given it defaults to `False`

    Returns:
        A ``[N,N]`` matrix of distances if only ``x`` is given, else a ``[N,M]`` matrix

    Example:
        >>> import torch
        >>> from torchmetrics.functional import pairwise_minkowski_distance
        >>> x = torch.tensor([[2, 3], [3, 5], [5, 8]], dtype=torch.float32)
        >>> y = torch.tensor([[1, 0], [2, 1]], dtype=torch.float32)
        >>> pairwise_minkowski_distance(x, y, p=4)
        tensor([[3.0092, 2.0000],
                [5.0317, 4.0039],
                [8.1222, 7.0583]])
        >>> pairwise_minkowski_distance(x, p=4)
        tensor([[0.0000, 2.0305, 5.1547],
                [2.0305, 0.0000, 3.1383],
                [5.1547, 3.1383, 0.0000]])
    """
    distance = _pairwise_minkowski_distance_update(x, y, p, zero_diagonal)
    return _reduce_distance_matrix(distance, reduction)
