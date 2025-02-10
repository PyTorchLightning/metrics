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
import torch
from torch import Tensor

from torchmetrics.functional.classification import multiclass_confusion_matrix
from torchmetrics.functional.clustering.utils import check_cluster_labels
from torchmetrics.utilities.imports import _TORCH_LINEAR_ASSIGNMENT_AVAILABLE

if not _TORCH_LINEAR_ASSIGNMENT_AVAILABLE:
    __doctest_skip__ = ["cluster_accuracy"]


def _cluster_accuracy_compute(confmat: Tensor) -> Tensor:
    """Computes the clustering accuracy from a confusion matrix."""
    from torch_linear_assignment import batch_linear_assignment

    confmat = confmat[None]
    # solve the linear sum assignment problem
    assignment = batch_linear_assignment(confmat.max() - confmat)
    confmat = confmat[0]
    # extract the true positives
    tps = confmat[torch.arange(confmat.shape[0]), assignment.flatten()]
    return tps.sum() / confmat.sum()


def cluster_accuracy(preds: Tensor, target: Tensor, num_classes: int) -> Tensor:
    """Computes the clustering accuracy between the predicted and target clusters.

    Args:
        preds: predicted cluster labels
        target: ground truth cluster labels
        num_classes: number of classes

    Returns:
        Scalar tensor with clustering accuracy between 0.0 and 1.0

    Raises:
        RuntimeError:
            If `torch_linear_assignment` is not installed

    Example:
        >>> from torchmetrics.functional.clustering import cluster_accuracy
        >>> preds = torch.tensor([0, 0, 1, 1])
        >>> target = torch.tensor([1, 1, 0, 0])
        >>> cluster_accuracy(preds, target, 2)
        tensor(1.000)

    """
    if not _TORCH_LINEAR_ASSIGNMENT_AVAILABLE:
        raise RuntimeError(
            "Missing `torch_linear_assignment`. Please install it with `pip install torchmetrics[clustering]`."
        )
    check_cluster_labels(preds, target)
    confmat = multiclass_confusion_matrix(preds, target, num_classes=num_classes)
    return _cluster_accuracy_compute(confmat)
