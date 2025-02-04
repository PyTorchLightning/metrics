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
from functools import partial

import pytest
import torch
from sklearn.metrics import f1_score
from torch import tensor

from torchmetrics import MetricCollection
from torchmetrics.functional.segmentation.dice import dice_score
from torchmetrics.segmentation.dice import DiceScore
from unittests import NUM_CLASSES
from unittests._helpers import seed_all
from unittests._helpers.testers import MetricTester
from unittests.segmentation.inputs import _input4, _inputs1, _inputs2, _inputs3

seed_all(42)


def _reference_dice_score(
    preds: torch.Tensor,
    target: torch.Tensor,
    input_format: str,
    include_background: bool = True,
    average: str = "micro",
    reduce: bool = True,
):
    """Calculate reference metric for dice score."""
    if input_format == "one-hot":
        preds = preds.argmax(dim=1)
        target = target.argmax(dim=1)
    preds = preds.cpu().numpy()
    target = target.cpu().numpy()

    labels = list(range(1, NUM_CLASSES) if not include_background else range(NUM_CLASSES))
    val = [f1_score(t.flatten(), p.flatten(), average=average, labels=labels) for t, p in zip(target, preds)]
    if reduce:
        val = torch.tensor(val).mean(dim=0)
    return val


@pytest.mark.parametrize(
    "preds, target, input_format",
    [
        (_inputs1.preds, _inputs1.target, "one-hot"),
        (_inputs2.preds, _inputs2.target, "one-hot"),
        (_inputs3.preds, _inputs3.target, "index"),
        (_input4.preds, _input4.target, "index"),
    ],
)
@pytest.mark.parametrize("include_background", [True, False])
@pytest.mark.parametrize("average", ["micro", "macro", "weighted", None])
class TestDiceScore(MetricTester):
    """Test class for `DiceScore` metric."""

    @pytest.mark.parametrize("ddp", [pytest.param(True, marks=pytest.mark.DDP), False])
    def test_dice_score_class(self, preds, target, input_format, include_background, average, ddp):
        """Test class implementation of metric."""
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=DiceScore,
            reference_metric=partial(
                _reference_dice_score,
                input_format=input_format,
                include_background=include_background,
                average=average,
                reduce=True,
            ),
            metric_args={
                "num_classes": NUM_CLASSES,
                "include_background": include_background,
                "average": average,
                "input_format": input_format,
            },
        )

    def test_dice_score_functional(self, preds, target, input_format, include_background, average):
        """Test functional implementation of metric."""
        self.run_functional_metric_test(
            preds=preds,
            target=target,
            metric_functional=dice_score,
            reference_metric=partial(
                _reference_dice_score,
                input_format=input_format,
                include_background=include_background,
                average=average,
                reduce=False,
            ),
            metric_args={
                "num_classes": NUM_CLASSES,
                "include_background": include_background,
                "average": average,
                "input_format": input_format,
            },
        )


@pytest.mark.parametrize("average", ["micro", "macro", "weighted", None])
def test_corner_case_no_overlap(average):
    """Check that if no overlap and intersection between target and preds, the dice score is 0.

    See issue: https://github.com/Lightning-AI/torchmetrics/issues/2851

    """
    target = torch.full((4, 4, 128, 128), 0, dtype=torch.int8)
    preds = torch.full((4, 4, 128, 128), 0, dtype=torch.int8)
    target[0, 0] = 1
    preds[0, 0] = 1
    dice = DiceScore(num_classes=3, average=average, include_background=False)
    assert dice(preds, target) == 0.0


@pytest.mark.parametrize("average", ["micro", "macro", "weighted", None])
@pytest.mark.parametrize("zero_division", [1.0, 0.0, "warn", "nan"])
def test_zero_division(zero_division, average):
    """Test different zero_division values."""
    target = torch.full((1, 3, 128, 128), 0, dtype=torch.int8)
    preds = torch.full((1, 3, 128, 128), 0, dtype=torch.int8)
    target[0, 0] = 1
    dice = DiceScore(num_classes=3, average=average, zero_division=zero_division)
    score = dice(preds, target)

    res_dict = {
        "micro": {1.0: tensor(0.0), 0.0: tensor(0.0), "warn": tensor(0.0), "nan": tensor(0.0)},
        "macro": {1.0: tensor(0.6667), 0.0: tensor(0.0), "warn": tensor(0.0), "nan": tensor(float("nan"))},
        "weighted": {1.0: tensor(0.0), 0.0: tensor(0.0), "warn": tensor(0.0), "nan": tensor(float("nan"))},
        None: {
            1.0: tensor([0.0, 1.0, 1.0]),
            0.0: tensor([0.0, 0.0, 0.0]),
            "warn": tensor([0.0, 0.0, 0.0]),
            "nan": tensor([0.0, float("nan"), float("nan")]),
        },
    }

    assert torch.allclose(score, res_dict[average][zero_division], atol=1e-4, equal_nan=True), (
        f"Expected {res_dict[average][zero_division]} but got {score}"
    )


@pytest.mark.parametrize("compute_groups", [True, False])
def test_dice_score_metric_collection(compute_groups: bool, num_batches: int = 4):
    """Test that the metric works within a metric collection with and without compute groups."""
    metric_collection = MetricCollection(
        metrics={
            "DiceScore (micro)": DiceScore(
                num_classes=NUM_CLASSES,
                average="micro",
            ),
            "DiceScore (macro)": DiceScore(
                num_classes=NUM_CLASSES,
                average="macro",
            ),
            "DiceScore (weighted)": DiceScore(
                num_classes=NUM_CLASSES,
                average="weighted",
            ),
        },
        compute_groups=compute_groups,
    )

    for _ in range(num_batches):
        metric_collection.update(_inputs1.preds, _inputs1.target)
    result = metric_collection.compute()

    assert isinstance(result, dict)
    assert len(set(metric_collection.keys()) - set(result.keys())) == 0
