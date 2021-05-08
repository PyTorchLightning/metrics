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
from typing import Optional

import torch
from torch import Tensor

from torchmetrics.utilities import rank_zero_warn
from torchmetrics.utilities.checks import _input_format_classification
from torchmetrics.utilities.enums import DataType


def _confusion_matrix_update(
    preds: Tensor, target: Tensor, num_classes: int, threshold: float = 0.5, multilabel: bool = False
) -> Tensor:
    preds, target, mode = _input_format_classification(preds, target, threshold)
    if mode not in (DataType.BINARY, DataType.MULTILABEL):
        preds = preds.argmax(dim=1)
        target = target.argmax(dim=1)

    if multilabel:
        unique_mapping = ((2 * target + preds) + 4 * torch.arange(num_classes, device=preds.device)).flatten()
        minlength = 4 * num_classes
    else:
        unique_mapping = (target.view(-1) * num_classes + preds.view(-1)).to(torch.long)
        minlength = num_classes**2

    bins = torch.bincount(unique_mapping, minlength=minlength)
    if multilabel:
        confmat = bins.reshape(num_classes, 2, 2)
    else:
        confmat = bins.reshape(num_classes, num_classes)
    return confmat


def _confusion_matrix_compute(confmat: Tensor, normalize: Optional[str] = None) -> Tensor:
    allowed_normalize = ('true', 'pred', 'all', 'none', None)
    assert normalize in allowed_normalize, \
        f"Argument average needs to one of the following: {allowed_normalize}"
    confmat = confmat.float()
    if normalize is not None and normalize != 'none':
        cm = None
        if normalize == 'true':
            cm = confmat / confmat.sum(axis=1, keepdim=True)
        elif normalize == 'pred':
            cm = confmat / confmat.sum(axis=0, keepdim=True)
        elif normalize == 'all':
            cm = confmat / confmat.sum()
        nan_elements = cm[torch.isnan(cm)].nelement()
        if nan_elements != 0:
            cm[torch.isnan(cm)] = 0
            rank_zero_warn(f'{nan_elements} nan values found in confusion matrix have been replaced with zeros.')
        return cm
    return confmat


def confusion_matrix(
    preds: Tensor,
    target: Tensor,
    num_classes: int,
    normalize: Optional[str] = None,
    threshold: float = 0.5,
    multilabel: bool = False
) -> Tensor:
    """
    Computes the `confusion matrix
    <https://scikit-learn.org/stable/modules/model_evaluation.html#confusion-matrix>`_.  Works with binary,
    multiclass, and multilabel data.  Accepts probabilities or logits from a model output or integer class
    values in prediction. Works with multi-dimensional preds and target, but it should be noted that
    additional dimensions will be flattened.

    If preds and target are the same shape and preds is a float tensor, we use the ``self.threshold`` argument
    to convert into integer labels. This is the case for binary and multi-label probabilities or logits.

    If preds has an extra dimension as in the case of multi-class scores we perform an argmax on ``dim=1``.

    If working with multilabel data, setting the `is_multilabel` argument to `True` will make sure that a
    `confusion matrix gets calculated per label
    <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.multilabel_confusion_matrix.html>`_.

    Args:
        preds: (float or long tensor), Either a ``(N, ...)`` tensor with labels or
            ``(N, C, ...)`` where C is the number of classes, tensor with labels/logits/probabilities
        target: ``target`` (long tensor), tensor with shape ``(N, ...)`` with ground true labels
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
    confmat = _confusion_matrix_update(preds, target, num_classes, threshold, multilabel)
    return _confusion_matrix_compute(confmat, normalize)
