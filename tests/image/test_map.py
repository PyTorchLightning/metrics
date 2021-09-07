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

import pytest as pytest
import torch

from tests.helpers.testers import MetricTester, NUM_BATCHES
from torchmetrics.image.map import MAP, MAPMetricResults
from torchmetrics.utilities.imports import _PYCOCOTOOLS_AVAILABLE

Input = namedtuple("Input", ["preds", "target", "num_classes"])

_inputs = Input(
    preds=[
        {
            'detection_boxes': torch.Tensor([
                [258.15, 41.29, 348.26, 243.78]
            ]),
            'detection_scores': torch.Tensor([0.236]),
            'detection_classes': torch.IntTensor([4])
        },  # coco image id 42
        {
            'detection_boxes': torch.Tensor([
                [61, 22.75, 504, 609.67],
                [12.66, 3.32, 268.6, 271.91]
            ]),
            'detection_scores': torch.Tensor([0.318, 0.726]),
            'detection_classes': torch.IntTensor([3, 2])
        },  # coco image id 73
        {
            'detection_boxes': torch.Tensor([
                [87.87, 276.25, 296.42, 103.18],
                [0, 3.66, 142.15, 312.4],
                [296.55, 93.96, 18.42, 58.83],
                [328.94, 97.05, 13.55, 25.93],
                [356.62, 95.47, 15.71, 52.08],
                [464.08, 105.09, 31.66, 41.9],
                [276.11, 103.84, 15.33, 46.88],
            ]),
            'detection_scores': torch.Tensor([0.546, 0.3, 0.407, 0.611, 0.335, 0.805, 0.953]),
            'detection_classes': torch.IntTensor([4, 1, 0, 0, 0, 0, 0])
        },  # coco image id 74
    ],
    target=[
        {
            'groundtruth_boxes': torch.Tensor([
                [214.15, 41.29, 348.26, 243.78]
            ]),
            'groundtruth_classes': torch.IntTensor([4])
        },  # coco image id 42
        {
            'groundtruth_boxes': torch.Tensor([
                [13.0, 22.75, 535.98, 609.67],
                [1.66, 3.32, 268.6, 271.91],
            ]),
            'groundtruth_classes': torch.IntTensor([2, 2])
        },  # coco image id 73
        {
            'groundtruth_boxes': torch.Tensor([
                [61.87, 276.25, 296.42, 103.18],
                [2.75, 3.66, 159.4, 312.4],
                [295.55, 93.96, 18.42, 58.83],
                [326.94, 97.05, 13.55, 25.93],
                [356.62, 95.47, 15.71, 52.08],
                [462.08, 105.09, 31.66, 41.9],
                [277.11, 103.84, 15.33, 46.88]
            ]),
            'groundtruth_classes': torch.IntTensor([4, 1, 0, 0, 0, 0, 0])
        },  # coco image id 74
    ],
    num_classes=5
)


def _compare_fn() -> MAPMetricResults:
    """Comparison function for map implementation.

    Official pycocotools results calculated from a subset of https://github.com/cocodataset/cocoapi/tree/master/results
        All classes
        Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.658
        Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.670

        Class 0
        Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.725
        Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.780

        Class 1
        Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.800
        Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.800

        Class 2
        Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.454
        Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.450

        Class 3
        Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = -1.000
        Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = -1.000

        Class 4
        Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.650
        Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.650
    """
    return MAPMetricResults(
        map_value=torch.Tensor([0.658]),
        mar_value=torch.Tensor([0.670]),
        map_per_class_value=[torch.Tensor([0.725]), torch.Tensor([0.800]), torch.Tensor([0.454]),
                             torch.Tensor([-1.000]), torch.Tensor([0.650])],
        mar_per_class_value=[torch.Tensor([0.780]), torch.Tensor([0.800]), torch.Tensor([0.450]),
                             torch.Tensor([-1.000]), torch.Tensor([0.650])])


@pytest.mark.skipif(not _PYCOCOTOOLS_AVAILABLE, reason="test requires that pycocotools is installed")
class TestMAP(MetricTester):

    @pytest.mark.parametrize("num_batches", [1, NUM_BATCHES])
    def test_map(self, num_batches):
        """Test modular implementation for correctness.
        Skipping the MetricTester method as it currently does not work for object detection inputs"""

        map_metric = MAP(num_classes=_inputs.num_classes)

        for _ in range(num_batches):
            map_metric.update(preds=_inputs.preds, target=_inputs.target)
        pl_result = map_metric.compute()

        pycoco_result = _compare_fn()
        assert pl_result.map_value.item() == pytest.approx(pycoco_result.map_value, 0.01)
        assert pl_result.mar_value.item() == pytest.approx(pycoco_result.mar_value, 0.01)
        for i in range(_inputs.num_classes):
            assert pl_result.map_per_class_value[i].item() == pytest.approx(pycoco_result.map_per_class_value[i], 0.01)
            assert pl_result.mar_per_class_value[i].item() == pytest.approx(pycoco_result.mar_per_class_value[i], 0.01)

    # TODO adjust testers.py to enable testing with object detection inputs
    # @pytest.mark.parametrize("ddp", [True, False])
    # @pytest.mark.parametrize("dist_sync_on_step", [True, False])
    # def test_map(self, ddp, dist_sync_on_step):
    #     """Test modular implementation for correctness."""

    #     self.run_class_metric_test(
    #         ddp=ddp,
    #         preds=_inputs.preds,
    #         target=_inputs.target,
    #         metric_class=MAP,
    #         sk_metric=_compare_fn,
    #         dist_sync_on_step=dist_sync_on_step,
    #         metric_args={"num_classes": 3},
    #     )


# noinspection PyTypeChecker
@pytest.mark.skipif(not _PYCOCOTOOLS_AVAILABLE, reason="test requires that pycocotools is installed")
def test_error_on_wrong_init():
    """Test class raises the expected errors."""
    with pytest.raises(ValueError, match="Expected argument `num_classes` to be a integer larger or equal to 0"):
        MAP(num_classes=-1)

    with pytest.raises(ValueError, match="Expected argument `num_classes` to be a integer larger or equal to 0"):
        MAP(num_classes=None)
