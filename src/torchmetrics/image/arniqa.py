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
from collections.abc import Sequence
from typing import Any, Optional, Union

import torch
from torch import Tensor
from typing_extensions import Literal

from torchmetrics.functional.image.arniqa import _ARNIQA, _arniqa_compute, _arniqa_update, _NoTrainArniqa
from torchmetrics.metric import Metric
from torchmetrics.utilities.checks import _SKIP_SLOW_DOCTEST, _try_proceed_with_timeout
from torchmetrics.utilities.imports import _MATPLOTLIB_AVAILABLE, _TORCHVISION_AVAILABLE
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE

if not _MATPLOTLIB_AVAILABLE:
    __doctest_skip__ = ["ARNIQA.plot"]

if _TORCHVISION_AVAILABLE:

    def _download_arniqa() -> None:
        _ARNIQA(regressor_dataset="koniq10k")

    if _SKIP_SLOW_DOCTEST and not _try_proceed_with_timeout(_download_arniqa):
        __doctest_skip__ = ["ARNIQA", "ARNIQA.plot"]
else:
    __doctest_skip__ = ["ARNIQA", "ARNIQA.plot"]


class ARNIQA(Metric):
    """`ARNIQA`_ is a No-Reference Image Quality Assessment metric that predicts the technical quality of an image with
    a high correlation with human opinions.

    ARNIQA consists of an encoder and a regressor. The encoder is a ResNet-50 model trained in a self-supervised way to
    model the image distortion manifold to generate similar representation for images with similar distortions, regardless
    of the image content. The regressor is a linear model trained on IQA datasets using the ground-truth quality scores.
    ARNIQA extracts the features from the full- and half-scale versions of the input image and then outputs a quality score
    in the [0, 1] range, where higher is better.

    The input image is expected to have shape ``(N, 3, H, W)``. The image should be in the [0, 1] range if `normalize`
    is set to ``True``, otherwise it should be normalized with the ImageNet mean and standard deviation.

    .. hint::
        Using this metric requires you to have ``torchvision`` package installed. Either install as
        ``pip install torchmetrics[image]`` or ``pip install torchvision``.

    As input to ``forward`` and ``update`` the metric accepts the following input

    - ``img`` (:class:`~torch.Tensor`): tensor with images of shape ``(N, 3, H, W)``

    As output of `forward` and `compute` the metric returns the following output

    - ``arniqa`` (:class:`~torch.Tensor`): tensor with ARNIQA score. If `reduction` is set to ``none``, the output will
      have shape ``(N,)``, otherwise it will be a scalar tensor. Tensor values are in the [0, 1] range, where higher
      is better.

    Args:
        img: the input image
        regressor_dataset: dataset used for training the regressor. Choose between [``koniq10k``, ``kadid10k``]. ``koniq10k``
            corresponds to the `KonIQ-10k`_ dataset, which consists of real-world images with authentic distortions.
            ``kadid10k`` corresponds to the `KADID-10k`_ dataset, which consists of images with synthetically generated distortions.
        reduction: indicates how to reduce over the batch dimension. Choose between [``sum``, ``mean``, ``none``].
        normalize: by default this is ``True`` meaning that the input is expected to be in the [0, 1] range. If set
            to ``False`` will instead expect input to be already normalized with the ImageNet mean and standard deviation.
        kwargs: additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Raises:
        ModuleNotFoundError:
            If ``torchvision`` package is not installed
        ValueError:
            If ``regressor_dataset`` is not in [``"kadid10k"``, ``"koniq10k"``]
        ValueError:
            If ``reduction`` is not in [``"sum"``, ``"mean"``, ``"none"``]
        ValueError:
            If ``normalize`` is not a bool
        ValueError:
            If the input image is not a valid image tensor with shape [N, 3, H, W].
        ValueError:
            If the input image values are not in the [0, 1] range when ``normalize`` is set to ``True``

    Example::
        Non-normalized input:

        >>> from torch import rand
        >>> from torchmetrics.image.arniqa import ARNIQA
        >>> img = rand(8, 3, 224, 224)
        >>> metric = ARNIQA(regressor_dataset='koniq10k', normalize=True)
        >>> metric(img)
        tensor(0.4875)

    Example::
        Normalized input:

        >>> from torch import rand
        >>> from torchmetrics.image.arniqa import ARNIQA
        >>> from torchvision.transforms import Normalize
        >>> img = rand(8, 3, 224, 224)
        >>> img = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)
        >>> metric = ARNIQA(regressor_dataset='koniq10k', normalize=False)
        >>> metric(img)
        tensor(0.4875)

    """

    is_differentiable: bool = True
    higher_is_better: bool = True
    full_state_update: bool = False
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0

    sum_scores: Tensor
    total: Tensor
    feature_network: str = "model"

    def __init__(
        self,
        regressor_dataset: Literal["kadid10k", "koniq10k"] = "koniq10k",
        reduction: Literal["sum", "mean", "none"] = "mean",
        normalize: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self.model = _NoTrainArniqa(regressor_dataset=regressor_dataset)

        valid_reduction = ("mean", "sum", "none")
        if reduction not in valid_reduction:
            raise ValueError(f"Argument `reduction` must be one of {valid_reduction}, but got {reduction}")
        self.reduction = reduction

        if not isinstance(normalize, bool):
            raise ValueError(f"Argument `normalize` should be a bool but got {normalize}")
        self.normalize = normalize

        self.add_state("sum_scores", torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("num_scores", torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, img: Tensor) -> None:
        """Update internal states with arniqa score."""
        loss, num_scores = _arniqa_update(img, model=self.model, normalize=self.normalize)
        self.sum_scores += loss.sum()
        self.num_scores += num_scores

    def compute(self) -> Tensor:
        """Compute final arniqa metric."""
        return _arniqa_compute(self.sum_scores, self.num_scores, self.reduction)

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
            >>> from torchmetrics.image.arniqa import ARNIQA
            >>> metric = ARNIQA(regressor_dataset='koniq10k')
            >>> metric.update(torch.rand(8, 3, 224, 224))
            >>> fig_, ax_ = metric.plot()

        .. plot::
            :scale: 75

            >>> # Example plotting multiple values
            >>> import torch
            >>> from torchmetrics.image.arniqa import ARNIQA
            >>> metric = ARNIQA(regressor_dataset='koniq10k')
            >>> values = [ ]
            >>> for _ in range(3):
            ...     values.append(metric(torch.rand(8, 3, 224, 224)))
            >>> fig_, ax_ = metric.plot(values)

        """
        return self._plot(val, ax)
