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

from typing import Any, Dict, List, Optional, Sequence, Union

from torch import Tensor
from typing_extensions import Literal

from torchmetrics.functional.image.d_lambda import _spectral_distortion_index_compute, _spectral_distortion_index_update
from torchmetrics.functional.image.d_s import _spatial_distortion_index_compute, _spatial_distortion_index_update
from torchmetrics.metric import Metric
from torchmetrics.utilities import rank_zero_warn
from torchmetrics.utilities.data import dim_zero_cat
from torchmetrics.utilities.imports import _MATPLOTLIB_AVAILABLE, _TORCHVISION_AVAILABLE
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE

if not _MATPLOTLIB_AVAILABLE:
    __doctest_skip__ = ["QualityWithNoReference.plot"]

if not _TORCHVISION_AVAILABLE:
    __doctest_skip__ = ["QualityWithNoReference", "QualityWithNoReference.plot"]


class QualityWithNoReference(Metric):
    """Compute Quality with No Reference (QualityWithNoReference_) also now as QNR.

    The metric is used to compare the joint spectral and spatial distortion between two images.

    As input to ``forward`` and ``update`` the metric accepts the following input

    - ``preds`` (:class:`~torch.Tensor`): High resolution multispectral image of shape ``(N,C,H,W)``.
    - ``target`` (:class:`~Dict`): A dictionary containing the following keys:

      - ``ms`` (:class:`~torch.Tensor`): Low resolution multispectral image of shape ``(N,C,H',W')``.
      - ``pan`` (:class:`~torch.Tensor`): High resolution panchromatic image of shape ``(N,C,H,W)``.
      - ``pan_lr`` (:class:`~torch.Tensor`): (optional) Low resolution panchromatic image of shape ``(N,C,H',W')``.

    where H and W must be multiple of H' and W'.

    When ``pan_lr`` is ``None``, a uniform filter will be applied on ``pan`` to produce a degraded image. The degraded
    image is then resized to match the size of ``ms`` and served as ``pan_lr`` in the calculation.

    As output of `forward` and `compute` the metric returns the following output

    - ``qnr`` (:class:`~torch.Tensor`): if ``reduction!='none'`` returns float scalar tensor with average QNR value
      over sample else returns tensor of shape ``(N,)`` with QNR values per sample

    Args:
        alpha: Relevance of spectral distortion.
        beta: Relevance of spatial distortion.
        norm_order: Order of the norm applied on the difference.
        window_size: Window size of the filter applied to degrade the high resolution panchromatic image.
        reduction: a method to reduce metric score over labels.

            - ``'elementwise_mean'``: takes the mean (default)
            - ``'sum'``: takes the sum
            - ``'none'``: no reduction will be applied

        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Example:
        >>> import torch
        >>> _ = torch.manual_seed(42)
        >>> from torchmetrics.image import QualityWithNoReference
        >>> preds = torch.rand([16, 3, 32, 32])
        >>> target = {
        ...     'ms': torch.rand([16, 3, 16, 16]),
        ...     'pan': torch.rand([16, 3, 32, 32]),
        ... }
        >>> qnr = QualityWithNoReference()
        >>> qnr(preds, target)
        tensor(0.9694)

    """

    higher_is_better: bool = True
    is_differentiable: bool = True
    full_state_update: bool = False
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0

    preds: List[Tensor]
    ms: List[Tensor]
    pan: List[Tensor]
    pan_lr: List[Tensor]

    def __init__(
        self,
        alpha: float = 1,
        beta: float = 1,
        norm_order: int = 1,
        window_size: int = 7,
        reduction: Literal["elementwise_mean", "sum", "none"] = "elementwise_mean",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        rank_zero_warn(
            "Metric `QualityWithNoReference` will save all targets and predictions in buffer."
            " For large datasets this may lead to large memory footprint."
        )

        if not isinstance(alpha, (int, float)) or alpha < 0:
            raise ValueError(f"Expected `alpha` to be a non-negative real number. Got alpha: {alpha}.")
        self.alpha = alpha
        if not isinstance(beta, (int, float)) or beta < 0:
            raise ValueError(f"Expected `beta` to be a non-negative real number. Got beta: {beta}.")
        self.beta = beta
        if not isinstance(norm_order, int) or norm_order <= 0:
            raise ValueError(f"Expected `norm_order` to be a positive integer. Got norm_order: {norm_order}.")
        self.norm_order = norm_order
        if not isinstance(window_size, int) or window_size <= 0:
            raise ValueError(f"Expected `window_size` to be a positive integer. Got window_size: {window_size}.")
        self.window_size = window_size
        allowed_reductions = ("elementwise_mean", "sum", "none")
        if reduction not in allowed_reductions:
            raise ValueError(f"Expected argument `reduction` be one of {allowed_reductions} but got {reduction}")
        self.reduction = reduction
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("ms", default=[], dist_reduce_fx="cat")
        self.add_state("pan", default=[], dist_reduce_fx="cat")
        self.add_state("pan_lr", default=[], dist_reduce_fx="cat")

    def update(self, preds: Tensor, target: Dict[str, Tensor]) -> None:
        """Update state with preds and target.

        Args:
            preds: High resolution multispectral image.
            target: A dictionary containing the following keys:

                - ``'ms'``: low resolution multispectral image.
                - ``'pan'``: high resolution panchromatic image.
                - ``'pan_lr'``: (optional) low resolution panchromatic image.

        Raises:
            ValueError:
                If ``target`` doesn't have ``ms`` and ``pan``.

        """
        if "ms" not in target:
            raise ValueError(f"Expected `target` to have key `ms`. Got target: {target.keys()}.")
        if "pan" not in target:
            raise ValueError(f"Expected `target` to have key `pan`. Got target: {target.keys()}.")
        ms = target["ms"]
        pan = target["pan"]
        pan_lr = target.get("pan_lr")
        preds, ms = _spectral_distortion_index_update(preds, ms)
        preds, ms, pan, pan_lr = _spatial_distortion_index_update(preds, ms, pan, pan_lr)
        self.preds.append(preds)
        self.ms.append(target["ms"])
        self.pan.append(target["pan"])
        if "pan_lr" in target:
            self.pan_lr.append(target["pan_lr"])

    def compute(self) -> Tensor:
        """Compute and returns quality with no reference."""
        preds = dim_zero_cat(self.preds)
        ms = dim_zero_cat(self.ms)
        pan = dim_zero_cat(self.pan)
        pan_lr = dim_zero_cat(self.pan_lr) if len(self.pan_lr) > 0 else None
        d_lambda = _spectral_distortion_index_compute(preds, ms, self.norm_order, self.reduction)
        d_s = _spatial_distortion_index_compute(
            preds, ms, pan, pan_lr, self.norm_order, self.window_size, self.reduction
        )
        return (1 - d_lambda) ** self.alpha * (1 - d_s) ** self.beta

    def plot(
        self, val: Optional[Union[Tensor, Sequence[Tensor]]] = None, ax: Optional[_AX_TYPE] = None
    ) -> _PLOT_OUT_TYPE:
        """Plot a single or multiple values from the metric.

        Args:
            val: Either a single result from calling `metric.forward` or `metric.compute` or a list of these results.
                If no value is provided, will automatically call `metric.compute` and plot that result.
            ax: An matplotlib axis object. If provided will add plot to that axis

        Returns:
            Figure and Axes object

        Raises:
            ModuleNotFoundError:
                If `matplotlib` is not installed

        .. plot::
            :scale: 75

            >>> # Example plotting a single value
            >>> import torch
            >>> _ = torch.manual_seed(42)
            >>> from torchmetrics.image import QualityWithNoReference
            >>> preds = torch.rand([16, 3, 32, 32])
            >>> target = {
            ...     'ms': torch.rand([16, 3, 16, 16]),
            ...     'pan': torch.rand([16, 3, 32, 32]),
            ... }
            >>> metric = QualityWithNoReference()
            >>> metric.update(preds, target)
            >>> fig_, ax_ = metric.plot()

        .. plot::
            :scale: 75

            >>> # Example plotting multiple values
            >>> import torch
            >>> _ = torch.manual_seed(42)
            >>> from torchmetrics.image import QualityWithNoReference
            >>> preds = torch.rand([16, 3, 32, 32])
            >>> target = {
            ...     'ms': torch.rand([16, 3, 16, 16]),
            ...     'pan': torch.rand([16, 3, 32, 32]),
            ... }
            >>> metric = QualityWithNoReference()
            >>> values = [ ]
            >>> for _ in range(10):
            ...     values.append(metric(preds, target))
            >>> fig_, ax_ = metric.plot(values)

        """
        return self._plot(val, ax)
