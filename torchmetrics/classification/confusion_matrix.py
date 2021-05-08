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
from typing import Any, Optional

import torch
from torch import Tensor

from torchmetrics.functional.classification.confusion_matrix import _confusion_matrix_compute, _confusion_matrix_update
from torchmetrics.metric import Metric


class ConfusionMatrix(Metric):
    """
    Computes the `confusion matrix
    <https://scikit-learn.org/stable/modules/model_evaluation.html#confusion-matrix>`_.  Works with binary,
    multiclass, and multilabel data.  Accepts probabilities or logits from a model output or integer class
    values in prediction. Works with multi-dimensional preds and target, but it should be noted that
    additional dimensions will be flattened.

    Forward accepts

    - ``preds`` (float or long tensor): ``(N, ...)`` or ``(N, C, ...)`` where C is the number of classes
    - ``target`` (long tensor): ``(N, ...)``

    If preds and target are the same shape and preds is a float tensor, we use the ``self.threshold`` argument
    to convert into integer labels. This is the case for binary and multi-label probabilities or logits.

    If preds has an extra dimension as in the case of multi-class scores we perform an argmax on ``dim=1``.

    If working with multilabel data, setting the `is_multilabel` argument to `True` will make sure that a
    `confusion matrix gets calculated per label
    <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.multilabel_confusion_matrix.html>`_.

    Args:
        num_classes: Number of classes in the dataset.
        normalize: Normalization mode for confusion matrix. Choose from

            - ``None`` or ``'none'``: no normalization (default)
            - ``'true'``: normalization over the targets (most commonly used)
            - ``'pred'``: normalization over the predictions
            - ``'all'``: normalization over the whole matrix

        threshold:
            Threshold for transforming probability or logit predictions to binary (0,1) predictions, in the case
            of binary or multi-label inputs. Default value of 0.5 corresponds to input being probabilities.

        multilabel:
            determines if data is multilabel or not.
        compute_on_step:
            Forward only calls ``update()`` and return None if this is set to False. default: True
        dist_sync_on_step:
            Synchronize metric state across processes at each ``forward()``
            before returning the value at the step. default: False
        process_group:
            Specify the process group on which synchronization is called. default: None (which selects the entire world)

    Example (binary data):
        >>> from torchmetrics import ConfusionMatrix
        >>> target = torch.tensor([1, 1, 0, 0])
        >>> preds = torch.tensor([0, 1, 0, 0])
        >>> confmat = ConfusionMatrix(num_classes=2)
        >>> confmat(preds, target)
        tensor([[2., 0.],
                [1., 1.]])

    Example (multiclass data):
        >>> target = torch.tensor([2, 1, 0, 0])
        >>> preds = torch.tensor([2, 1, 0, 1])
        >>> confmat = ConfusionMatrix(num_classes=3)
        >>> confmat(preds, target)
        tensor([[1., 1., 0.],
                [0., 1., 0.],
                [0., 0., 1.]])

    Example (multilabel data):
        >>> target = torch.tensor([[0, 1, 0], [1, 0, 1]])
        >>> preds = torch.tensor([[0, 0, 1], [1, 0, 1]])
        >>> confmat = ConfusionMatrix(num_classes=3, multilabel=True)
        >>> confmat(preds, target)  # doctest: +NORMALIZE_WHITESPACE
        tensor([[[1., 0.], [0., 1.]],
                [[1., 0.], [1., 0.]],
                [[0., 1.], [0., 1.]]])
    """

    def __init__(
        self,
        num_classes: int,
        normalize: Optional[str] = None,
        threshold: float = 0.5,
        multilabel: bool = False,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
    ):

        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
        )
        self.num_classes = num_classes
        self.normalize = normalize
        self.threshold = threshold
        self.multilabel = multilabel

        allowed_normalize = ('true', 'pred', 'all', 'none', None)
        assert self.normalize in allowed_normalize, \
            f"Argument average needs to one of the following: {allowed_normalize}"

        default = torch.zeros(num_classes, 2, 2) if multilabel else torch.zeros(num_classes, num_classes)
        self.add_state("confmat", default=default, dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor):
        """
        Update state with predictions and targets.

        Args:
            preds: Predictions from model
            target: Ground truth values
        """
        confmat = _confusion_matrix_update(preds, target, self.num_classes, self.threshold, self.multilabel)
        self.confmat += confmat

    def compute(self) -> Tensor:
        """
        Computes confusion matrix.

        Returns:
            If `multilabel=False` this will be a `[n_classes, n_classes]` tensor and if `multilabel=True`
            this will be a `[n_classes, 2, 2]` tensor
        """
        return _confusion_matrix_compute(self.confmat, self.normalize)
