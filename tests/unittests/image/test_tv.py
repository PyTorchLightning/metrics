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
from collections import namedtuple
from functools import partial

import pytest
import torch
from kornia.losses import total_variation as kornia_total_variation

from torchmetrics.functional.image.tv import total_variation
from torchmetrics.image.tv import TotalVariation
from torchmetrics.utilities.imports import _TORCH_GREATER_EQUAL_1_6, _TORCH_GREATER_EQUAL_1_8
from unittests.helpers import seed_all
from unittests.helpers.testers import BATCH_SIZE, NUM_BATCHES, MetricTester

seed_all(42)


# add extra argument to make the metric and reference fit into the testing framework
class TotalVariationTester(TotalVariation):
    def update(self, img, *args):
        super().update(img=img)


def total_variaion_tester(preds, target, reduction="mean"):
    return total_variation(preds, reduction)


def total_variation_kornia_tester(preds, target, reduction):
    score = kornia_total_variation(preds)
    return score.sum() if reduction == "sum" else score.mean()


# define inputs
Input = namedtuple("Input", ["preds", "target"])

_inputs = []
for size, channel, dtype in [
    (12, 3, torch.float),
    (13, 3, torch.float32),
    (14, 3, torch.double),
    (15, 3, torch.float64),
]:
    preds = torch.rand(NUM_BATCHES, BATCH_SIZE, channel, size, size, dtype=dtype)
    target = torch.rand(NUM_BATCHES, BATCH_SIZE, channel, size, size, dtype=dtype)
    _inputs.append(Input(preds=preds, target=target))


@pytest.mark.parametrize(
    "preds, target",
    [(i.preds, i.target) for i in _inputs],
)
@pytest.mark.parametrize("reduction", ["sum", "mean"])
@pytest.mark.skipif(not _TORCH_GREATER_EQUAL_1_8, reason="Kornia used as reference requires min PT version")
class TestTotalVariation(MetricTester):
    @pytest.mark.parametrize("ddp", [True, False])
    @pytest.mark.parametrize("dist_sync_on_step", [True, False])
    def test_total_variation(self, preds, target, reduction, ddp, dist_sync_on_step):
        """Test modular implementation."""
        self.run_class_metric_test(
            ddp,
            preds,
            target,
            TotalVariationTester,
            partial(total_variation_kornia_tester, reduction=reduction),
            dist_sync_on_step,
            metric_args={"reduction": reduction},
        )

    def test_total_variation_functional(self, preds, target, reduction):
        """Test for functional implementation."""
        self.run_functional_metric_test(
            preds,
            target,
            total_variaion_tester,
            partial(total_variation_kornia_tester, reduction=reduction),
            metric_args={"reduction": reduction},
        )

    @pytest.mark.skipif(
        not _TORCH_GREATER_EQUAL_1_6, reason="half support of core operations on not support before pytorch v1.6"
    )
    def test_sam_half_cpu(self, preds, target, reduction):
        """Test for half precision on CPU."""
        self.run_precision_test_cpu(
            preds,
            target,
            TotalVariationTester,
            total_variaion_tester,
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
    def test_sam_half_gpu(self, preds, target, reduction):
        """Test for half precision on GPU."""
        self.run_precision_test_gpu(preds, target, TotalVariationTester, total_variaion_tester)


def test_correct_args():
    """that that arguments have the right type and sizes."""
    with pytest.raises(ValueError, match="Expected argument `reduction`.*"):
        _ = TotalVariation(reduction="diff")

    with pytest.raises(RuntimeError, match="Expected input `img` to.*"):
        _ = total_variation(torch.randn(1, 2, 3))
