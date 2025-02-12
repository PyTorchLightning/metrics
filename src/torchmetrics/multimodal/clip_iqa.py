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
from typing import TYPE_CHECKING, Any, List, Literal, Optional, Union

import torch
from torch import Tensor

from torchmetrics.functional.multimodal.clip_iqa import (
    _clip_iqa_compute,
    _clip_iqa_format_prompts,
    _clip_iqa_get_anchor_vectors,
    _clip_iqa_update,
    _get_clip_iqa_model_and_processor,
)
from torchmetrics.metric import Metric
from torchmetrics.utilities.checks import _SKIP_SLOW_DOCTEST, _try_proceed_with_timeout
from torchmetrics.utilities.data import dim_zero_cat
from torchmetrics.utilities.imports import (
    _MATPLOTLIB_AVAILABLE,
    _PIQ_GREATER_EQUAL_0_8,
    _TRANSFORMERS_GREATER_EQUAL_4_10,
)
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE

if not _PIQ_GREATER_EQUAL_0_8:
    __doctest_skip__ = ["CLIPImageQualityAssessment", "CLIPImageQualityAssessment.plot"]

if not _MATPLOTLIB_AVAILABLE:
    __doctest_skip__ = ["CLIPImageQualityAssessment.plot"]

if _SKIP_SLOW_DOCTEST and _TRANSFORMERS_GREATER_EQUAL_4_10:
    from transformers import CLIPModel as _CLIPModel
    from transformers import CLIPProcessor as _CLIPProcessor

    def _download_clip_iqa_metric() -> None:
        _CLIPModel.from_pretrained("openai/clip-vit-large-patch14", resume_download=True)
        _CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14", resume_download=True)

    if not _try_proceed_with_timeout(_download_clip_iqa_metric):
        __doctest_skip__ = ["CLIPImageQualityAssessment", "CLIPImageQualityAssessment.plot"]
else:
    __doctest_skip__ = ["CLIPImageQualityAssessment", "CLIPImageQualityAssessment.plot"]


class CLIPImageQualityAssessment(Metric):
    """Calculates `CLIP-IQA`_, that can be used to measure the visual content of images.

    The metric is based on the `CLIP`_ model, which is a neural network trained on a variety of (image, text) pairs to
    be able to generate a vector representation of the image and the text that is similar if the image and text are
    semantically similar.

    The metric works by calculating the cosine similarity between user provided images and pre-defined prompts. The
    prompts always comes in pairs of "positive" and "negative" such as "Good photo." and "Bad photo.". By calculating
    the similartity between image embeddings and both the "positive" and "negative" prompt, the metric can determine
    which prompt the image is more similar to. The metric then returns the probability that the image is more similar
    to the first prompt than the second prompt.

    Build in prompts are:
        * quality: "Good photo." vs "Bad photo."
        * brightness: "Bright photo." vs "Dark photo."
        * noisiness: "Clean photo." vs "Noisy photo."
        * colorfullness: "Colorful photo." vs "Dull photo."
        * sharpness: "Sharp photo." vs "Blurry photo."
        * contrast: "High contrast photo." vs "Low contrast photo."
        * complexity: "Complex photo." vs "Simple photo."
        * natural: "Natural photo." vs "Synthetic photo."
        * happy: "Happy photo." vs "Sad photo."
        * scary: "Scary photo." vs "Peaceful photo."
        * new: "New photo." vs "Old photo."
        * warm: "Warm photo." vs "Cold photo."
        * real: "Real photo." vs "Abstract photo."
        * beautiful: "Beautiful photo." vs "Ugly photo."
        * lonely: "Lonely photo." vs "Sociable photo."
        * relaxing: "Relaxing photo." vs "Stressful photo."

    As input to ``forward`` and ``update`` the metric accepts the following input

    - ``images`` (:class:`~torch.Tensor`): tensor with images feed to the feature extractor with shape ``(N,C,H,W)``

    As output of `forward` and `compute` the metric returns the following output

    - ``clip_iqa`` (:class:`~torch.Tensor` or dict of tensors): tensor with the CLIP-IQA score. If a single prompt is
      provided, a single tensor with shape ``(N,)`` is returned. If a list of prompts is provided, a dict of tensors
      is returned with the prompt as key and the tensor with shape ``(N,)`` as value.

    Args:
        model_name_or_path: string indicating the version of the CLIP model to use. Available models are:

            - `"clip_iqa"`, model corresponding to the CLIP-IQA paper.
            - `"openai/clip-vit-base-patch16"`
            - `"openai/clip-vit-base-patch32"`
            - `"openai/clip-vit-large-patch14-336"`
            - `"openai/clip-vit-large-patch14"`

        data_range: The maximum value of the input tensor. For example, if the input images are in range [0, 255],
            data_range should be 255. The images are normalized by this value.
        prompts: A string, tuple of strings or nested tuple of strings. If a single string is provided, it must be one
            of the available prompts (see above). Else the input is expected to be a tuple, where each element can
            be one of two things: either a string or a tuple of strings. If a string is provided, it must be one of the
            available prompts (see above). If tuple is provided, it must be of length 2 and the first string must be a
            positive prompt and the second string must be a negative prompt.
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    .. hint::
        If using the default `clip_iqa` model, the package `piq` must be installed. Either install with
        `pip install piq` or `pip install torchmetrics[image]`.

    Raises:
        ModuleNotFoundError:
            If transformers package is not installed or version is lower than 4.10.0
        ValueError:
            If `prompts` is a tuple and it is not of length 2
        ValueError:
            If `prompts` is a string and it is not one of the available prompts
        ValueError:
            If `prompts` is a list of strings and not all strings are one of the available prompts

    Example::
        Single prompt:

        >>> from torch import randint
        >>> from torchmetrics.multimodal import CLIPImageQualityAssessment
        >>> imgs = randint(255, (2, 3, 224, 224)).float()
        >>> metric = CLIPImageQualityAssessment()
        >>> metric(imgs)
        tensor([0.8894, 0.8902])

    Example::
        Multiple prompts:

        >>> from torch import randint
        >>> from torchmetrics.multimodal import CLIPImageQualityAssessment
        >>> imgs = randint(255, (2, 3, 224, 224)).float()
        >>> metric = CLIPImageQualityAssessment(prompts=("quality", "brightness"))
        >>> metric(imgs)
        {'quality': tensor([0.8693, 0.8705]), 'brightness': tensor([0.5722, 0.4762])}

    Example::
        Custom prompts. Must always be a tuple of length 2, with a positive and negative prompt.

        >>> from torch import randint
        >>> from torchmetrics.multimodal import CLIPImageQualityAssessment
        >>> imgs = randint(255, (2, 3, 224, 224)).float()
        >>> metric = CLIPImageQualityAssessment(prompts=(("Super good photo.", "Super bad photo."), "brightness"))
        >>> metric(imgs)
        {'user_defined_0': tensor([0.9578, 0.9654]), 'brightness': tensor([0.5495, 0.5764])}

    """

    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = True
    plot_lower_bound = 0.0
    plot_upper_bound = 100.0

    anchors: Tensor
    probs_list: List[Tensor]
    feature_network: str = "model"

    def __init__(
        self,
        model_name_or_path: Literal[
            "clip_iqa",
            "openai/clip-vit-base-patch16",
            "openai/clip-vit-base-patch32",
            "openai/clip-vit-large-patch14-336",
            "openai/clip-vit-large-patch14",
        ] = "clip_iqa",
        data_range: float = 1.0,
        prompts: tuple[Union[str, tuple[str, str]], ...] = ("quality",),
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if not (isinstance(data_range, (int, float)) and data_range > 0):
            raise ValueError("Argument `data_range` should be a positive number.")
        self.data_range = data_range

        prompts_list, prompts_name = _clip_iqa_format_prompts(prompts)
        self.prompts_list = prompts_list
        self.prompts_name = prompts_name

        self.model, self.processor = _get_clip_iqa_model_and_processor(model_name_or_path)
        self.model_name_or_path = model_name_or_path

        with torch.inference_mode():
            anchors = _clip_iqa_get_anchor_vectors(
                model_name_or_path, self.model, self.processor, self.prompts_list, self.device
            )
        self.register_buffer("anchors", anchors)

        self.add_state("probs_list", [], dist_reduce_fx="cat")

    def update(self, images: Tensor) -> None:
        """Update metric state with new data."""
        with torch.inference_mode():
            img_features = _clip_iqa_update(
                self.model_name_or_path, images, self.model, self.processor, self.data_range, self.device
            )
            probs = _clip_iqa_compute(img_features, self.anchors, self.prompts_name, format_as_dict=False)
            if not isinstance(probs, Tensor):
                raise ValueError("Output probs should be a tensor")
            self.probs_list.append(probs)

    def compute(self) -> Union[Tensor, dict[str, Tensor]]:
        """Compute metric."""
        probs = dim_zero_cat(self.probs_list)
        if len(self.prompts_name) == 1:
            return probs.squeeze()
        return {p: probs[:, i] for i, p in enumerate(self.prompts_name)}

    def plot(self, val: Union[Tensor, Sequence[Tensor], None] = None, ax: Optional[_AX_TYPE] = None) -> _PLOT_OUT_TYPE:
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
            >>> from torchmetrics.multimodal.clip_iqa import CLIPImageQualityAssessment
            >>> metric = CLIPImageQualityAssessment()
            >>> metric.update(torch.rand(1, 3, 224, 224))
            >>> fig_, ax_ = metric.plot()

        .. plot::
            :scale: 75

            >>> # Example plotting multiple values
            >>> import torch
            >>> from torchmetrics.multimodal.clip_iqa import CLIPImageQualityAssessment
            >>> metric = CLIPImageQualityAssessment()
            >>> values = [ ]
            >>> for _ in range(10):
            ...     values.append(metric(torch.rand(1, 3, 224, 224)))
            >>> fig_, ax_ = metric.plot(values)

        """
        return self._plot(val, ax)


if TYPE_CHECKING:
    from typing import cast

    images = cast(Any, None)
    f = CLIPImageQualityAssessment
    f(prompts=("colorfullness",))
    f(
        prompts=("quality", "brightness", "noisiness"),
    )
    f(
        prompts=("quality", "brightness", "noisiness", "colorfullness"),
    )
    f(prompts=(("Photo of a cat", "Photo of a dog"), "quality", ("Colorful photo", "Black and white photo")))
