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
import pytest
import torch
from aeon.performance_metrics.clustering import clustering_accuracy_score

from torchmetrics.clustering.cluster_accuracy import ClusterAccuracy
from torchmetrics.functional.clustering.cluster_accuracy import cluster_accuracy
from torchmetrics.utilities.imports import _TORCH_LINEAR_ASSIGNMENT_AVAILABLE
from unittests import NUM_CLASSES
from unittests._helpers import seed_all
from unittests._helpers.testers import MetricTester
from unittests.clustering._inputs import _float_inputs_extrinsic, _single_target_extrinsic1, _single_target_extrinsic2

seed_all(42)


@pytest.mark.skipif(not _TORCH_LINEAR_ASSIGNMENT_AVAILABLE, reason="test requires torch linear assignment package")
@pytest.mark.parametrize(
    "preds, target",
    [
        (_single_target_extrinsic1.preds, _single_target_extrinsic1.target),
        (_single_target_extrinsic2.preds, _single_target_extrinsic2.target),
    ],
)
class TestAdjustedMutualInfoScore(MetricTester):
    """Test class for `AdjustedMutualInfoScore` metric."""

    @pytest.mark.parametrize("ddp", [pytest.param(True, marks=pytest.mark.DDP), False])
    def test_cluster_accuracy(self, preds, target, ddp):
        """Test class implementation of metric."""
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=ClusterAccuracy,
            reference_metric=clustering_accuracy_score,
            metric_args={"num_classes": NUM_CLASSES},
        )

    def test_cluster_accuracy_functional(self, preds, target):
        """Test functional implementation of metric."""
        self.run_functional_metric_test(
            preds=preds,
            target=target,
            metric_functional=cluster_accuracy,
            reference_metric=clustering_accuracy_score,
            metric_args={"num_classes": NUM_CLASSES},
        )


@pytest.mark.skipif(not _TORCH_LINEAR_ASSIGNMENT_AVAILABLE, reason="test requires torch linear assignment package")
def test_cluster_accuracy_sanity_check():
    """Check that metric works with the simplest possible inputs."""
    preds = torch.tensor([0, 0, 1, 1])
    target = torch.tensor([1, 1, 0, 0])
    metric = ClusterAccuracy(num_classes=2)
    res = metric(preds, target)
    assert torch.allclose(res, torch.tensor(1.0))


@pytest.mark.skipif(not _TORCH_LINEAR_ASSIGNMENT_AVAILABLE, reason="test requires torch linear assignment package")
def test_cluster_accuracy_functional_raises_invalid_task():
    """Check that metric rejects continuous-valued inputs."""
    preds, target = _float_inputs_extrinsic
    with pytest.raises(ValueError, match=r"Expected *"):
        cluster_accuracy(preds, target, num_classes=NUM_CLASSES)
