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

# Content inspired by the ARNIQA official repository:
# https://github.com/miccunifi/ARNIQA
# Copyright (c) 2024, Lorenzo Agnolucci, Leonardo Galteri, Marco Bertini, Alberto Del Bimbo
# All rights reserved.
# License under Apache-2.0 License
import warnings
from typing import Union

import torch
import torch.nn.functional as F
import torchvision
from torch import Tensor, nn
from typing_extensions import Literal

from torchmetrics.utilities.imports import _TORCHVISION_AVAILABLE

_available_regressor_datasets = {
    "kadid10k": (1, 5),
    "koniq10k": (1, 100),
}

_base_url = "https://github.com/miccunifi/ARNIQA/releases/download/weights"


if not _TORCHVISION_AVAILABLE:
    __doctest_skip__ = ["arniqa"]


class _ARNIQA(nn.Module):
    """Initializes a No-Reference Image Quality Assessment ARNIQA torch.nn.Module.

    Args:
        regressor_dataset: dataset used for training the regressor, choose between [``koniq10k``, ``kadid10k``]

    """

    def __init__(self, regressor_dataset: str = "koniq10k") -> None:
        super().__init__()

        if not _TORCHVISION_AVAILABLE:
            raise ModuleNotFoundError(
                "ARNIQA metric requires that torchvision is installed."
                " Either install as `pip install torchmetrics[image]` or `pip install torchvision`."
            )

        valid_regressor_dataset = ("kadid10k", "koniq10k")
        if regressor_dataset not in valid_regressor_dataset:
            raise ValueError(
                f"Argument `regressor_dataset` must be one of {valid_regressor_dataset}, but got {regressor_dataset}."
            )

        self.regressor_dataset = regressor_dataset
        self.imagenet_norm_mean = [0.485, 0.456, 0.406]
        self.imagenet_norm_std = [0.229, 0.224, 0.225]

        encoder = torchvision.models.resnet50()
        self.feat_dim = encoder.fc.in_features  # get dimensions of the last layer of the encoder
        encoder = nn.Sequential(*list(encoder.children())[:-1])  # remove the fully connected layer
        self.encoder = encoder
        self.regressor = nn.Linear(self.feat_dim * 2, 1)
        self._load_weights()

        self.encoder.eval()
        self.regressor.eval()

        for p in self.encoder.parameters():
            p.requires_grad = False
        for p in self.regressor.parameters():
            p.requires_grad = False

    def _load_weights(self) -> None:
        """Loads the weights of the encoder and regressor."""
        encoder_state_dict = torch.hub.load_state_dict_from_url(
            f"{_base_url}/ARNIQA.pth", progress=True, map_location="cpu"
        )
        filtered_encoder_state_dict = {
            k.replace("model.", ""): v for k, v in encoder_state_dict.items() if "projector" not in k
        }
        self.encoder.load_state_dict(filtered_encoder_state_dict, strict=True)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="torch.serialization")
            regressor_state_dict = torch.hub.load_state_dict_from_url(
                f"{_base_url}/regressor_{self.regressor_dataset}.pth", progress=True, map_location="cpu"
            ).state_dict()
            # Rename the keys to match the regressor's state_dict
            regressor_state_dict["weight"] = regressor_state_dict.pop("weights")
            regressor_state_dict["bias"] = regressor_state_dict.pop("biases").unsqueeze(0)
            self.regressor.load_state_dict(regressor_state_dict, strict=True)

    def _preprocess_input(self, img: Tensor, normalize: bool = False) -> tuple[Tensor, Tensor]:
        """Preprocesses the input to the model.

        Obtains the half-scale version of the input image and applies normalization if needed.

        """
        h, w = img.shape[-2:]
        img_ds = torchvision.transforms.Resize((h // 2, w // 2))(img)  # get the half-scale version of the image
        if normalize:
            img = torchvision.transforms.Normalize(mean=self.imagenet_norm_mean, std=self.imagenet_norm_std)(img)
            img_ds = torchvision.transforms.Normalize(mean=self.imagenet_norm_mean, std=self.imagenet_norm_std)(img_ds)
        return img, img_ds

    def _scale_score(self, score: Tensor) -> Tensor:
        """Scales the quality score to be in the [0, 1] range, where higher is better."""
        min_score, max_score = _available_regressor_datasets[self.regressor_dataset]
        return (score - min_score) / (max_score - min_score)

    def forward(self, img: Tensor, normalize: bool = False) -> Tensor:
        # Preprocessing
        img, img_ds = self._preprocess_input(img, normalize)

        # Extract features from full- and half-scale images
        img_f = self.encoder(img)
        img_f = img_f.view(-1, self.feat_dim)
        img_f = F.normalize(img_f, dim=1)
        img_ds_f = self.encoder(img_ds)
        img_ds_f = img_ds_f.view(-1, self.feat_dim)
        img_ds_f = F.normalize(img_ds_f, dim=1)
        f = torch.hstack((img_f, img_ds_f))

        # Get the quality score
        score = self.regressor(f)
        score = self._scale_score(score)

        return score


class _NoTrainArniqa(_ARNIQA):
    """Wrapper to make sure ARNIQA never leaves evaluation mode."""

    def train(self, mode: bool) -> "_NoTrainArniqa":  # type: ignore[override]
        """Force network to always be in evaluation mode."""
        return super().train(False)


def _arniqa_update(img: Tensor, model: nn.Module, normalize: bool) -> tuple[Tensor, Union[int, Tensor]]:
    # Check that the input image is valid
    if not (img.ndim == 4 and img.shape[1] == 3):
        raise ValueError(f"Input image must have shape [N, 3, H, W]. Got input with shape {img.shape}.")
    if not (img.max() <= 1.0 and img.min() >= 0.0) and normalize:
        raise ValueError(
            f"Input image values must be in the [0, 1] range when normalize==True. Got input with values"
            f" in range {img.min()} and {img.max()}."
        )

    with torch.amp.autocast(
        device_type=img.device.type,
        dtype=img.dtype,
        enabled=False if (img.dtype == torch.float32 and img.device.type == "cpu") else True,
    ):
        loss = model(img, normalize=normalize).squeeze()
    return loss, img.shape[0]


def _arniqa_compute(
    scores: Tensor, num_scores: Union[Tensor, int], reduction: Literal["sum", "mean", "none"] = "mean"
) -> Tensor:
    sum_scores = scores.sum()
    if reduction == "none":
        return scores
    if reduction == "mean":
        return sum_scores / num_scores
    return sum_scores


def arniqa(
    img: Tensor,
    regressor_dataset: Literal["kadid10k", "koniq10k"] = "koniq10k",
    reduction: Literal["sum", "mean", "none"] = "mean",
    normalize: bool = True,
) -> Tensor:
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

    Args:
        img: the input image
        regressor_dataset: dataset used for training the regressor. Choose between [``koniq10k``, ``kadid10k``]. ``koniq10k``
            corresponds to the `KonIQ-10k`_ dataset, which consists of real-world images with authentic distortions.
            ``kadid10k`` corresponds to the `KADID-10k`_ dataset, which consists of images with synthetically generated distortions.
        reduction: indicates how to reduce over the batch dimension. Choose between [``sum``, ``mean``, ``none``].
        normalize: by default this is ``True`` meaning that the input is expected to be in the [0, 1] range. If set
            to ``False`` will instead expect input to be already normalized with the ImageNet mean and standard deviation.

    Returns:
        A tensor in the [0, 1] range, where higher is better, representing the ARNIQA score of the input image. If
        `reduction` is set to ``none``, the output will have shape ``(N,)``, otherwise it will be a scalar tensor.

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
        >>> from torchmetrics.functional.image.arniqa import arniqa
        >>> img = rand(8, 3, 224, 224)
        >>> arniqa(img, regressor_dataset='koniq10k', normalize=True)
        tensor(0.4875)

    Example::
        Normalized input:

        >>> from torch import rand
        >>> from torchmetrics.functional.image.arniqa import arniqa
        >>> from torchvision.transforms import Normalize
        >>> img = rand(8, 3, 224, 224)
        >>> img = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)
        >>> arniqa(img, regressor_dataset='koniq10k', normalize=False)
        tensor(0.4875)

    """
    valid_reduction = ("mean", "sum", "none")
    if reduction not in valid_reduction:
        raise ValueError(f"Argument `reduction` must be one of {valid_reduction}, but got {reduction}")

    if not isinstance(normalize, bool):
        raise ValueError(f"Argument `normalize` should be a bool but got {normalize}")

    model = _NoTrainArniqa(regressor_dataset=regressor_dataset).to(device=img.device, dtype=img.dtype)
    loss, num_scores = _arniqa_update(img, model, normalize)
    return _arniqa_compute(loss, num_scores, reduction)
