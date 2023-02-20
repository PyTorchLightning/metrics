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
import warnings
from typing import Any, Collection

import torch
from torch import Tensor

from torchmetrics.functional.detection.panoptic_quality import (
    _get_category_id_to_continuous_id,
    _get_void_color,
    _panoptic_quality_compute,
    _panoptic_quality_update,
    _parse_categories,
    _prepocess_inputs,
    _validate_inputs,
)
from torchmetrics.metric import Metric


class PanopticQuality(Metric):
    r"""Compute the `Panoptic Quality`_ for panoptic segmentations.

        .. math::
            PQ = \frac{IOU}{TP + 0.5 FP + 0.5 FN}

        where IOU, TP, FP and FN are respectively the sum of the intersection over union for true positives,
        the number of true postitives, false positives and false negatives. This metric is inspired by the PQ
        implementation of panopticapi, a standard implementation for the PQ metric for panoptic segmentation.

    .. note:
        Metric is currently experimental

    Args:
            things:
                Set of ``category_id`` for countable things.
            stuffs:
                Set of ``category_id`` for uncountable stuffs.
            allow_unknown_preds_category:
                Boolean flag to specify if unknown categories in the predictions are to be ignored in the metric
                computation or raise an exception when found.

        TODO(marco): where to say that points in target that are not things or stuff
        TODO(cont.) are ignored from metric computation?

    Raises:
            ValueError:
                If ``things``, ``stuffs`` have at least one common ``category_id``.
            TypeError:
                If ``things``, ``stuffs`` contain non-integer ``category_id``.

    Example:
            >>> from torch import tensor
            >>> preds = tensor([[[[6, 0], [0, 0], [6, 0], [6, 0]],
            ...                  [[0, 0], [0, 0], [6, 0], [0, 1]],
            ...                  [[0, 0], [0, 0], [6, 0], [0, 1]],
            ...                  [[0, 0], [7, 0], [6, 0], [1, 0]],
            ...                  [[0, 0], [7, 0], [7, 0], [7, 0]]]])
            >>> target = tensor([[[[6, 0], [0, 1], [6, 0], [0, 1]],
            ...                   [[0, 1], [0, 1], [6, 0], [0, 1]],
            ...                   [[0, 1], [0, 1], [6, 0], [1, 0]],
            ...                   [[0, 1], [7, 0], [1, 0], [1, 0]],
            ...                   [[0, 1], [7, 0], [7, 0], [7, 0]]]])
            >>> panoptic_quality(preds, target, things = {0, 1}, stuffs = {6, 7})
            tensor(0.5463, dtype=torch.float64)

    """
    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = False

    iou_sum: Tensor
    true_positives: Tensor
    false_positives: Tensor
    false_negatives: Tensor

    def __init__(
        self,
        things: Collection[int],
        stuffs: Collection[int],
        allow_unknown_preds_category: bool = False,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)

        # todo: better testing for correctness of metric
        warnings.warn("This is experimental version and are actively working on its stability.")

        things, stuffs = _parse_categories(things, stuffs)
        self.things = things
        self.stuffs = stuffs
        self.void_color = _get_void_color(things, stuffs)
        self.cat_id_to_continuous_id = _get_category_id_to_continuous_id(things, stuffs)
        self.allow_unknown_preds_category = allow_unknown_preds_category

        # per category intermediate metrics
        n_categories = len(things) + len(stuffs)
        self.add_state("iou_sum", default=torch.zeros(n_categories, dtype=torch.double), dist_reduce_fx="sum")
        self.add_state("true_positives", default=torch.zeros(n_categories, dtype=torch.int), dist_reduce_fx="sum")
        self.add_state("false_positives", default=torch.zeros(n_categories, dtype=torch.int), dist_reduce_fx="sum")
        self.add_state("false_negatives", default=torch.zeros(n_categories, dtype=torch.int), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:
        r"""Update state with predictions and targets.

        Args:
            preds: panoptic detection of shape ``[batch, *spatial_dims, 2]`` containing
                the pair ``(category_id, instance_id)`` for each point.
                If the ``category_id`` refer to a stuff, the instance_id is ignored.

            target: ground truth of shape ``[batch, *spatial_dims, 2]`` containing
                the pair ``(category_id, instance_id)`` for each pixel of the image.
                If the ``category_id`` refer to a stuff, the instance_id is ignored.

        Raises:
            TypeError:
                If ``preds`` or ``target`` is not an ``torch.Tensor``.
            ValueError:
                If ``preds`` and ``target`` have different shape.
            ValueError:
                If ``preds`` has less than 3 dimensions.
            ValueError:
                If the final dimension of ``preds`` has size != 2.
        """
        _validate_inputs(preds, target)
        flatten_preds = _prepocess_inputs(
            self.things, self.stuffs, preds, self.void_color, self.allow_unknown_preds_category
        )
        flatten_target = _prepocess_inputs(self.things, self.stuffs, target, self.void_color, True)
        iou_sum, true_positives, false_positives, false_negatives = _panoptic_quality_update(
            flatten_preds, flatten_target, self.cat_id_to_continuous_id, self.void_color
        )
        self.iou_sum += iou_sum
        self.true_positives += true_positives
        self.false_positives += false_positives
        self.false_negatives += false_negatives

    def compute(self) -> Tensor:
        """Computes panoptic quality based on inputs passed in to ``update`` previously."""
        return _panoptic_quality_compute(self.iou_sum, self.true_positives, self.false_positives, self.false_negatives)
