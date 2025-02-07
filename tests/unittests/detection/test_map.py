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
import contextlib
import io
import json
from copy import deepcopy
from functools import partial
from itertools import product

import numpy as np
import pytest
import torch
from lightning_utilities import apply_to_collection
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch import IntTensor, Tensor

from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchmetrics.utilities.imports import (
    _FASTER_COCO_EVAL_AVAILABLE,
    _PYCOCOTOOLS_AVAILABLE,
)
from unittests._helpers.testers import MetricTester
from unittests.detection import _DETECTION_BBOX, _DETECTION_SEGM, _DETECTION_VAL


def _skip_if_faster_coco_eval_missing(backend):
    if backend == "faster_coco_eval" and not _FASTER_COCO_EVAL_AVAILABLE:
        pytest.skip("test requires that faster_coco_eval is installed")


def _generate_coco_inputs(iou_type):
    """Generates inputs for the MAP metric.

    The inputs are generated from the official COCO results json files:
    https://github.com/cocodataset/cocoapi/tree/master/results
    and should therefore correspond directly to the result on the webpage

    """
    batched_preds, batched_target = MeanAveragePrecision().coco_to_tm(
        _DETECTION_BBOX if iou_type == "bbox" else _DETECTION_SEGM, _DETECTION_VAL, iou_type
    )

    # create 10 batches of 10 preds/targets each
    batched_preds = [batched_preds[10 * i : 10 * (i + 1)] for i in range(10)]
    batched_target = [batched_target[10 * i : 10 * (i + 1)] for i in range(10)]
    return batched_preds, batched_target


_coco_bbox_input = _generate_coco_inputs("bbox")
_coco_segm_input = _generate_coco_inputs("segm")


@pytest.mark.skipif(
    not _PYCOCOTOOLS_AVAILABLE, reason="test requires that torchvision=>0.8.0 and pycocotools is installed"
)
@pytest.mark.parametrize("iou_type", ["bbox", "segm"])
@pytest.mark.parametrize("backend", ["pycocotools", "faster_coco_eval"])
def test_tm_to_coco(tmpdir, iou_type, backend):
    """Test that the conversion from TM to COCO format works."""
    preds, target = _coco_bbox_input if iou_type == "bbox" else _coco_segm_input
    metric = MeanAveragePrecision(iou_type=iou_type, backend=backend, box_format="xywh")
    for bp, bt in zip(preds, target):
        metric.update(bp, bt)
    metric.tm_to_coco(f"{tmpdir}/tm_map_input")
    preds_2, target_2 = MeanAveragePrecision().coco_to_tm(
        f"{tmpdir}/tm_map_input_preds.json",
        f"{tmpdir}/tm_map_input_target.json",
        iou_type=iou_type,
        backend=backend,
    )

    preds = [p for batch in preds for p in batch]
    target = [t for batch in target for t in batch]

    # make sure that every prediction/target is found in the new prediction/target after saving and loading
    for sample1 in preds:
        sample_found = False
        for sample2 in preds_2:
            if iou_type == "segm":
                if sample1["masks"].shape == sample2["masks"].shape and torch.allclose(
                    sample1["masks"], sample2["masks"]
                ):
                    sample_found = True
            else:
                if sample1["boxes"].shape == sample2["boxes"].shape and torch.allclose(
                    sample1["boxes"], sample2["boxes"]
                ):
                    sample_found = True
        assert sample_found, "preds not found"

    for sample1 in target:
        sample_found = False
        for sample2 in target_2:
            if iou_type == "segm":
                if sample1["masks"].shape == sample2["masks"].shape and torch.allclose(
                    sample1["masks"], sample2["masks"]
                ):
                    sample_found = True
            else:
                if sample1["boxes"].shape == sample2["boxes"].shape and torch.allclose(
                    sample1["boxes"], sample2["boxes"]
                ):
                    sample_found = True
        assert sample_found, "target not found"


def _compare_against_coco_fn(preds, target, iou_type, iou_thresholds=None, rec_thresholds=None, class_metrics=True):
    """Taken from https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb."""
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(_DETECTION_VAL)
        dt = gt.loadRes(_DETECTION_BBOX) if iou_type == "bbox" else gt.loadRes(_DETECTION_SEGM)

        coco_eval = COCOeval(gt, dt, iou_type)
        if iou_thresholds is not None:
            coco_eval.params.iouThrs = np.array(iou_thresholds, dtype=np.float64)
        if rec_thresholds is not None:
            coco_eval.params.recThrs = np.array(rec_thresholds, dtype=np.float64)

        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
    global_stats = deepcopy(coco_eval.stats)

    map_per_class_values = torch.Tensor([-1])
    mar_100_per_class_values = torch.Tensor([-1])
    classes = torch.tensor(
        list(set(torch.arange(91).tolist()) - {0, 12, 19, 26, 29, 30, 45, 66, 68, 69, 71, 76, 83, 87, 89})
    )

    if class_metrics:
        map_per_class_list = []
        mar_100_per_class_list = []
        for class_id in classes.tolist():
            coco_eval.params.catIds = [class_id]
            with contextlib.redirect_stdout(io.StringIO()):
                coco_eval.evaluate()
                coco_eval.accumulate()
                coco_eval.summarize()
            class_stats = coco_eval.stats
            map_per_class_list.append(torch.Tensor([class_stats[0]]))
            mar_100_per_class_list.append(torch.Tensor([class_stats[8]]))

        map_per_class_values = torch.Tensor(map_per_class_list)
        mar_100_per_class_values = torch.Tensor(mar_100_per_class_list)

    return {
        "map": Tensor([global_stats[0]]),
        "map_50": Tensor([global_stats[1]]),
        "map_75": Tensor([global_stats[2]]),
        "map_small": Tensor([global_stats[3]]),
        "map_medium": Tensor([global_stats[4]]),
        "map_large": Tensor([global_stats[5]]),
        "mar_1": Tensor([global_stats[6]]),
        "mar_10": Tensor([global_stats[7]]),
        "mar_100": Tensor([global_stats[8]]),
        "mar_small": Tensor([global_stats[9]]),
        "mar_medium": Tensor([global_stats[10]]),
        "mar_large": Tensor([global_stats[11]]),
        "map_per_class": map_per_class_values,
        "mar_100_per_class": mar_100_per_class_values,
        "classes": classes,
    }


@pytest.mark.skipif(
    not _PYCOCOTOOLS_AVAILABLE, reason="test requires that torchvision=>0.8.0 and pycocotools is installed"
)
@pytest.mark.parametrize("iou_type", ["bbox", "segm"])
@pytest.mark.parametrize("ddp", [pytest.param(True, marks=pytest.mark.DDP), False])
@pytest.mark.parametrize("backend", ["pycocotools", "faster_coco_eval"])
class TestMAPUsingCOCOReference(MetricTester):
    """Test map metric on the reference coco data."""

    @pytest.mark.parametrize("iou_thresholds", [None, [0.25, 0.5, 0.75]])
    @pytest.mark.parametrize("rec_thresholds", [None, [0.25, 0.5, 0.75]])
    def test_map(self, iou_type, iou_thresholds, rec_thresholds, ddp, backend):
        """Test modular implementation for correctness."""
        _skip_if_faster_coco_eval_missing(backend)

        preds, target = _coco_bbox_input if iou_type == "bbox" else _coco_segm_input
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=MeanAveragePrecision,
            reference_metric=partial(
                _compare_against_coco_fn,
                iou_type=iou_type,
                iou_thresholds=iou_thresholds,
                rec_thresholds=rec_thresholds,
                class_metrics=False,
            ),
            metric_args={
                "iou_type": iou_type,
                "iou_thresholds": iou_thresholds,
                "rec_thresholds": rec_thresholds,
                "class_metrics": False,
                "box_format": "xywh",
                "backend": backend,
            },
            check_batch=False,
            atol=1e-2,
        )

    def test_map_classwise(self, iou_type, ddp, backend):
        """Test modular implementation for correctness with classwise=True.

        Needs bigger atol to be stable.

        """
        _skip_if_faster_coco_eval_missing(backend)
        preds, target = _coco_bbox_input if iou_type == "bbox" else _coco_segm_input
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=MeanAveragePrecision,
            reference_metric=partial(_compare_against_coco_fn, iou_type=iou_type, class_metrics=True),
            metric_args={"box_format": "xywh", "iou_type": iou_type, "class_metrics": True, "backend": backend},
            check_batch=False,
            atol=1e-1,
        )


@pytest.mark.parametrize("backend", ["pycocotools", "faster_coco_eval"])
def test_compare_both_same_time(tmpdir, backend):
    """Test that the class support evaluating both bbox and segm at the same time."""
    _skip_if_faster_coco_eval_missing(backend)

    with open(_DETECTION_BBOX) as f:
        boxes = json.load(f)
    with open(_DETECTION_SEGM) as f:
        segmentations = json.load(f)
    combined = [{**box, **seg} for box, seg in zip(boxes, segmentations)]
    with open(f"{tmpdir}/combined.json", "w") as f:
        json.dump(combined, f)
    batched_preds, batched_target = MeanAveragePrecision().coco_to_tm(
        f"{tmpdir}/combined.json", _DETECTION_VAL, iou_type=["bbox", "segm"]
    )
    batched_preds = [batched_preds[10 * i : 10 * (i + 1)] for i in range(10)]
    batched_target = [batched_target[10 * i : 10 * (i + 1)] for i in range(10)]

    metric = MeanAveragePrecision(iou_type=["bbox", "segm"], box_format="xywh", backend=backend)
    for bp, bt in zip(batched_preds, batched_target):
        metric.update(bp, bt)
    res = metric.compute()

    res1 = _compare_against_coco_fn([], [], iou_type="bbox", class_metrics=False)
    res2 = _compare_against_coco_fn([], [], iou_type="segm", class_metrics=False)

    for k, v in res1.items():
        if k == "classes":
            continue
        assert f"bbox_{k}" in res
        assert torch.allclose(res[f"bbox_{k}"], v, atol=1e-2)

    for k, v in res2.items():
        if k == "classes":
            continue
        assert f"segm_{k}" in res
        assert torch.allclose(res[f"segm_{k}"], v, atol=1e-2)


_inputs = {
    "preds": [
        [
            {
                "boxes": Tensor([[258.15, 41.29, 606.41, 285.07]]),
                "scores": Tensor([0.236]),
                "labels": IntTensor([4]),
            },  # coco image id 42
            {
                "boxes": Tensor([[61.00, 22.75, 565.00, 632.42], [12.66, 3.32, 281.26, 275.23]]),
                "scores": Tensor([0.318, 0.726]),
                "labels": IntTensor([3, 2]),
            },  # coco image id 73
        ],
        [
            {
                "boxes": Tensor([
                    [87.87, 276.25, 384.29, 379.43],
                    [0.00, 3.66, 142.15, 316.06],
                    [296.55, 93.96, 314.97, 152.79],
                    [328.94, 97.05, 342.49, 122.98],
                    [356.62, 95.47, 372.33, 147.55],
                    [464.08, 105.09, 495.74, 146.99],
                    [276.11, 103.84, 291.44, 150.72],
                ]),
                "scores": Tensor([0.546, 0.3, 0.407, 0.611, 0.335, 0.805, 0.953]),
                "labels": IntTensor([4, 1, 0, 0, 0, 0, 0]),
            },  # coco image id 74
            {
                "boxes": Tensor([
                    [72.92, 45.96, 91.23, 80.57],
                    [45.17, 45.34, 66.28, 79.83],
                    [82.28, 47.04, 99.66, 78.50],
                    [59.96, 46.17, 80.35, 80.48],
                    [75.29, 23.01, 91.85, 50.85],
                    [71.14, 1.10, 96.96, 28.33],
                    [61.34, 55.23, 77.14, 79.57],
                    [41.17, 45.78, 60.99, 78.48],
                    [56.18, 44.80, 64.42, 56.25],
                ]),
                "scores": Tensor([0.532, 0.204, 0.782, 0.202, 0.883, 0.271, 0.561, 0.204, 0.349]),
                "labels": IntTensor([49, 49, 49, 49, 49, 49, 49, 49, 49]),
            },  # coco image id 987 category_id 49
        ],
    ],
    "target": [
        [
            {
                "boxes": Tensor([[214.1500, 41.2900, 562.4100, 285.0700]]),
                "labels": IntTensor([4]),
            },  # coco image id 42
            {
                "boxes": Tensor([
                    [13.00, 22.75, 548.98, 632.42],
                    [1.66, 3.32, 270.26, 275.23],
                ]),
                "labels": IntTensor([2, 2]),
            },  # coco image id 73
        ],
        [
            {
                "boxes": Tensor([
                    [61.87, 276.25, 358.29, 379.43],
                    [2.75, 3.66, 162.15, 316.06],
                    [295.55, 93.96, 313.97, 152.79],
                    [326.94, 97.05, 340.49, 122.98],
                    [356.62, 95.47, 372.33, 147.55],
                    [462.08, 105.09, 493.74, 146.99],
                    [277.11, 103.84, 292.44, 150.72],
                ]),
                "labels": IntTensor([4, 1, 0, 0, 0, 0, 0]),
            },  # coco image id 74
            {
                "boxes": Tensor([
                    [72.92, 45.96, 91.23, 80.57],
                    [50.17, 45.34, 71.28, 79.83],
                    [81.28, 47.04, 98.66, 78.50],
                    [63.96, 46.17, 84.35, 80.48],
                    [75.29, 23.01, 91.85, 50.85],
                    [56.39, 21.65, 75.66, 45.54],
                    [73.14, 1.10, 98.96, 28.33],
                    [62.34, 55.23, 78.14, 79.57],
                    [44.17, 45.78, 63.99, 78.48],
                    [58.18, 44.80, 66.42, 56.25],
                ]),
                "labels": IntTensor([49, 49, 49, 49, 49, 49, 49, 49, 49, 49]),
            },  # coco image id 987 category_id 49
        ],
    ],
}

# example from this issue https://github.com/Lightning-AI/torchmetrics/issues/943
_inputs2 = {
    "preds": [
        [
            {
                "boxes": Tensor([[258.0, 41.0, 606.0, 285.0]]),
                "scores": Tensor([0.536]),
                "labels": IntTensor([0]),
            },
        ],
        [
            {
                "boxes": Tensor([[258.0, 41.0, 606.0, 285.0]]),
                "scores": Tensor([0.536]),
                "labels": IntTensor([0]),
            }
        ],
    ],
    "target": [
        [
            {
                "boxes": Tensor([[214.0, 41.0, 562.0, 285.0]]),
                "labels": IntTensor([0]),
            }
        ],
        [
            {
                "boxes": Tensor([]),
                "labels": IntTensor([]),
            }
        ],
    ],
}

# Test empty preds case, to ensure bool inputs are properly casted to uint8
# From https://github.com/Lightning-AI/torchmetrics/issues/981
# and https://github.com/Lightning-AI/torchmetrics/issues/1147
_inputs3 = {
    "preds": [
        [
            {
                "boxes": Tensor([[258.0, 41.0, 606.0, 285.0]]),
                "scores": Tensor([0.536]),
                "labels": IntTensor([0]),
            },
        ],
        [
            {"boxes": Tensor([]), "scores": Tensor([]), "labels": IntTensor([])},
        ],
    ],
    "target": [
        [
            {
                "boxes": Tensor([[214.0, 41.0, 562.0, 285.0]]),
                "labels": IntTensor([0]),
            }
        ],
        [
            {
                "boxes": Tensor([[1.0, 2.0, 3.0, 4.0]]),
                "scores": Tensor([0.8]),  # target does not have scores
                "labels": IntTensor([1]),
            },
        ],
    ],
}


def _generate_random_segm_input(device, batch_size=2, num_preds_size=10, num_gt_size=10, random_size=True):
    """Generate random inputs for mAP when iou_type=segm."""
    preds = []
    targets = []
    for _ in range(batch_size):
        result = {}
        num_preds = torch.randint(0, num_preds_size, (1,)).item() if random_size else num_preds_size
        result["scores"] = torch.rand((num_preds,), device=device)
        result["labels"] = torch.randint(0, 10, (num_preds,), device=device)
        result["masks"] = torch.randint(0, 2, (num_preds, 10, 10), device=device).bool()
        preds.append(result)
        gt = {}
        num_gt = torch.randint(0, num_gt_size, (1,)).item() if random_size else num_gt_size
        gt["labels"] = torch.randint(0, 10, (num_gt,), device=device)
        gt["masks"] = torch.randint(0, 2, (num_gt, 10, 10), device=device).bool()
        targets.append(gt)
    return preds, targets


@pytest.mark.skipif(
    not _PYCOCOTOOLS_AVAILABLE, reason="test requires that torchvision=>0.8.0 and pycocotools is installed"
)
@pytest.mark.parametrize(
    "backend",
    [
        pytest.param("pycocotools"),
        pytest.param(
            "faster_coco_eval",
            marks=pytest.mark.skipif(
                not _FASTER_COCO_EVAL_AVAILABLE, reason="test requires that faster_coco_eval is installed"
            ),
        ),
    ],
)
class TestMapProperties:
    """Test class collection different tests for different properties parametrized by backend argument."""

    def test_error_on_wrong_init(self, backend):
        """Test class raises the expected errors."""
        MeanAveragePrecision(backend=backend)  # no error

        with pytest.raises(ValueError, match="Expected argument `class_metrics` to be a boolean"):
            MeanAveragePrecision(class_metrics=0, backend=backend)

    def test_empty_preds(self, backend):
        """Test empty predictions."""
        metric = MeanAveragePrecision(backend=backend)

        metric.update(
            [{"boxes": Tensor([]), "scores": Tensor([]), "labels": IntTensor([])}],
            [{"boxes": Tensor([[214.1500, 41.2900, 562.4100, 285.0700]]), "labels": IntTensor([4])}],
        )
        metric.compute()

    def test_empty_ground_truths(self, backend):
        """Test empty ground truths."""
        metric = MeanAveragePrecision(backend=backend)

        metric.update(
            [
                {
                    "boxes": Tensor([[214.1500, 41.2900, 562.4100, 285.0700]]),
                    "scores": Tensor([0.5]),
                    "labels": IntTensor([4]),
                }
            ],
            [{"boxes": Tensor([]), "labels": IntTensor([])}],
        )
        metric.compute()

    def test_empty_ground_truths_xywh(self, backend):
        """Test empty ground truths in xywh format."""
        metric = MeanAveragePrecision(box_format="xywh", backend=backend)

        metric.update(
            [
                {
                    "boxes": Tensor([[214.1500, 41.2900, 348.2600, 243.7800]]),
                    "scores": Tensor([0.5]),
                    "labels": IntTensor([4]),
                }
            ],
            [{"boxes": Tensor([]), "labels": IntTensor([])}],
        )
        metric.compute()

    def test_empty_preds_xywh(self, backend):
        """Test empty predictions in xywh format."""
        metric = MeanAveragePrecision(box_format="xywh", backend=backend)

        metric.update(
            [{"boxes": Tensor([]), "scores": Tensor([]), "labels": IntTensor([])}],
            [{"boxes": Tensor([[214.1500, 41.2900, 348.2600, 243.7800]]), "labels": IntTensor([4])}],
        )
        metric.compute()

    def test_empty_ground_truths_cxcywh(self, backend):
        """Test empty ground truths in cxcywh format."""
        metric = MeanAveragePrecision(box_format="cxcywh", backend=backend)

        metric.update(
            [
                {
                    "boxes": Tensor([[388.2800, 163.1800, 348.2600, 243.7800]]),
                    "scores": Tensor([0.5]),
                    "labels": IntTensor([4]),
                }
            ],
            [{"boxes": Tensor([]), "labels": IntTensor([])}],
        )
        metric.compute()

    def test_empty_preds_cxcywh(self, backend):
        """Test empty predictions in cxcywh format."""
        metric = MeanAveragePrecision(box_format="cxcywh", backend=backend)

        metric.update(
            [{"boxes": Tensor([]), "scores": Tensor([]), "labels": IntTensor([])}],
            [{"boxes": Tensor([[388.2800, 163.1800, 348.2600, 243.7800]]), "labels": IntTensor([4])}],
        )
        metric.compute()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires CUDA availability")
    @pytest.mark.parametrize("inputs", [_inputs, _inputs2, _inputs3])
    def test_map_gpu(self, backend, inputs):
        """Test predictions on single gpu."""
        metric = MeanAveragePrecision(backend=backend)
        metric = metric.to("cuda")
        for preds, targets in zip(deepcopy(inputs["preds"]), deepcopy(inputs["target"])):
            metric.update(
                apply_to_collection(preds, Tensor, lambda x: x.to("cuda")),
                apply_to_collection(targets, Tensor, lambda x: x.to("cuda")),
            )
        metric.compute()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires CUDA availability")
    def test_map_with_custom_thresholds(self, backend):
        """Test that map works with custom iou thresholds."""
        metric = MeanAveragePrecision(iou_thresholds=[0.1, 0.2], backend=backend)
        metric = metric.to("cuda")
        for preds, targets in zip(deepcopy(_inputs["preds"]), deepcopy(_inputs["target"])):
            metric.update(
                apply_to_collection(preds, Tensor, lambda x: x.to("cuda")),
                apply_to_collection(targets, Tensor, lambda x: x.to("cuda")),
            )
        res = metric.compute()
        assert res["map_50"].item() == -1
        assert res["map_75"].item() == -1

    def test_empty_metric(self, backend):
        """Test empty metric."""
        metric = MeanAveragePrecision(backend=backend)
        metric.compute()

    def test_missing_pred(self, backend):
        """One good detection, one false negative.

        Map should be lower than 1. Actually it is 0.5, but the exact value depends on where we are sampling (i.e.
        recall's values)

        """
        gts = [
            {"boxes": Tensor([[10, 20, 15, 25]]), "labels": IntTensor([0])},
            {"boxes": Tensor([[10, 20, 15, 25]]), "labels": IntTensor([0])},
        ]
        preds = [
            {"boxes": Tensor([[10, 20, 15, 25]]), "scores": Tensor([0.9]), "labels": IntTensor([0])},
            # Empty prediction
            {"boxes": Tensor([]), "scores": Tensor([]), "labels": IntTensor([])},
        ]
        metric = MeanAveragePrecision(backend=backend)
        metric.update(preds, gts)
        result = metric.compute()
        assert result["map"] < 1, "MAP cannot be 1, as there is a missing prediction."

    def test_missing_gt(self, backend):
        """The symmetric case of test_missing_pred.

        One good detection, one false positive. Map should be lower than 1. Actually it is 0.5, but the exact value
        depends on where we are sampling (i.e. recall's values)

        """
        gts = [
            {"boxes": Tensor([[10, 20, 15, 25]]), "labels": IntTensor([0])},
            {"boxes": Tensor([]), "labels": IntTensor([])},
        ]
        preds = [
            {"boxes": Tensor([[10, 20, 15, 25]]), "scores": Tensor([0.9]), "labels": IntTensor([0])},
            {"boxes": Tensor([[10, 20, 15, 25]]), "scores": Tensor([0.95]), "labels": IntTensor([0])},
        ]

        metric = MeanAveragePrecision(backend=backend)
        metric.update(preds, gts)
        result = metric.compute()
        assert result["map"] < 1, "MAP cannot be 1, as there is an image with no ground truth, but some predictions."

    def test_segm_iou_empty_gt_mask(self, backend):
        """Test empty ground truths."""
        metric = MeanAveragePrecision(iou_type="segm", backend=backend)
        metric.update(
            [{"masks": torch.randint(0, 1, (1, 10, 10)).bool(), "scores": Tensor([0.5]), "labels": IntTensor([4])}],
            [{"masks": Tensor([]), "labels": IntTensor([])}],
        )
        res = metric.compute()
        for key, value in res.items():
            if key == "classes":
                continue
            assert value.item() == -1, f"Expected -1 for {key}"
        assert res["classes"] == 4

    def test_segm_iou_empty_pred_mask(self, backend):
        """Test empty predictions."""
        metric = MeanAveragePrecision(iou_type="segm", backend=backend)
        metric.update(
            [{"masks": torch.BoolTensor([]), "scores": Tensor([]), "labels": IntTensor([])}],
            [{"masks": torch.randint(0, 1, (1, 10, 10)).bool(), "labels": IntTensor([4])}],
        )
        res = metric.compute()
        for key, value in res.items():
            if key == "classes":
                continue
            assert value.item() == -1, f"Expected -1 for {key}"
        assert res["classes"] == 4

    def test_error_on_wrong_input(self, backend):
        """Test class input validation."""
        metric = MeanAveragePrecision(backend=backend)

        metric.update([], [])  # no error

        with pytest.raises(ValueError, match="Expected argument `preds` to be of type Sequence"):
            metric.update(Tensor(), [])  # type: ignore

        with pytest.raises(ValueError, match="Expected argument `target` to be of type Sequence"):
            metric.update([], Tensor())  # type: ignore

        with pytest.raises(ValueError, match="Expected argument `preds` and `target` to have the same length"):
            metric.update([{}], [{}, {}])

        with pytest.raises(ValueError, match="Expected all dicts in `preds` to contain the `boxes` key"):
            metric.update(
                [{"scores": Tensor(), "labels": IntTensor}],
                [{"boxes": Tensor(), "labels": IntTensor()}],
            )

        with pytest.raises(ValueError, match="Expected all dicts in `preds` to contain the `scores` key"):
            metric.update(
                [{"boxes": Tensor(), "labels": IntTensor}],
                [{"boxes": Tensor(), "labels": IntTensor()}],
            )

        with pytest.raises(ValueError, match="Expected all dicts in `preds` to contain the `labels` key"):
            metric.update(
                [{"boxes": Tensor(), "scores": IntTensor}],
                [{"boxes": Tensor(), "labels": IntTensor()}],
            )

        with pytest.raises(ValueError, match="Expected all dicts in `target` to contain the `boxes` key"):
            metric.update(
                [{"boxes": Tensor(), "scores": IntTensor, "labels": IntTensor}],
                [{"labels": IntTensor()}],
            )

        with pytest.raises(ValueError, match="Expected all dicts in `target` to contain the `labels` key"):
            metric.update(
                [{"boxes": Tensor(), "scores": IntTensor, "labels": IntTensor}],
                [{"boxes": IntTensor()}],
            )

        with pytest.raises(ValueError, match="Expected all boxes in `preds` to be of type Tensor"):
            metric.update(
                [{"boxes": [], "scores": Tensor(), "labels": IntTensor()}],
                [{"boxes": Tensor(), "labels": IntTensor()}],
            )

        with pytest.raises(ValueError, match="Expected all scores in `preds` to be of type Tensor"):
            metric.update(
                [{"boxes": Tensor(), "scores": [], "labels": IntTensor()}],
                [{"boxes": Tensor(), "labels": IntTensor()}],
            )

        with pytest.raises(ValueError, match="Expected all labels in `preds` to be of type Tensor"):
            metric.update(
                [{"boxes": Tensor(), "scores": Tensor(), "labels": []}],
                [{"boxes": Tensor(), "labels": IntTensor()}],
            )

        with pytest.raises(ValueError, match="Expected all boxes in `target` to be of type Tensor"):
            metric.update(
                [{"boxes": Tensor(), "scores": Tensor(), "labels": IntTensor()}],
                [{"boxes": [], "labels": IntTensor()}],
            )

        with pytest.raises(ValueError, match="Expected all labels in `target` to be of type Tensor"):
            metric.update(
                [{"boxes": Tensor(), "scores": Tensor(), "labels": IntTensor()}],
                [{"boxes": Tensor(), "labels": []}],
            )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
    def test_device_changing(self, backend):
        """See issue: https://github.com/Lightning-AI/torchmetrics/issues/1743.

        Checks that the custom apply function of the metric works as expected.
        """
        device = "cuda"
        metric = MeanAveragePrecision(iou_type="segm", backend=backend).to(device)

        for _ in range(2):
            preds, targets = _generate_random_segm_input(device)
            metric.update(preds, targets)

        metric = metric.cpu()
        val = metric.compute()
        assert isinstance(val, dict)

    @pytest.mark.parametrize(
        ("box_format", "iou_val_expected", "map_val_expected"),
        [
            ("xyxy", 0.25, 1),
            ("xywh", 0.143, 0.0),
            ("cxcywh", 0.143, 0.0),
        ],
    )
    def test_for_box_format(self, box_format, iou_val_expected, map_val_expected, backend):
        """Test that only the correct box format lead to a score of 1.

        See issue: https://github.com/Lightning-AI/torchmetrics/issues/1908.

        """
        predictions = [
            {"boxes": torch.tensor([[0.5, 0.5, 1, 1]]), "scores": torch.tensor([1.0]), "labels": torch.tensor([0])}
        ]

        targets = [{"boxes": torch.tensor([[0, 0, 1, 1]]), "labels": torch.tensor([0])}]

        metric = MeanAveragePrecision(
            box_format=box_format, iou_thresholds=[0.2], extended_summary=True, backend=backend
        )
        metric.update(predictions, targets)
        result = metric.compute()
        assert result["map"].item() == map_val_expected
        assert round(float(result["ious"][(0, 0)]), 3) == iou_val_expected

    @pytest.mark.parametrize("iou_type", ["bbox", "segm"])
    @pytest.mark.parametrize("warn_on_many_detections", [False, True])
    def test_warning_on_many_detections(self, iou_type, warn_on_many_detections, backend, recwarn):
        """Test that a warning is raised when there are many detections."""
        if iou_type == "bbox":
            preds = [
                {
                    "boxes": torch.tensor([[0.5, 0.5, 1, 1]]).repeat(101, 1),
                    "scores": torch.tensor([1.0]).repeat(101),
                    "labels": torch.tensor([0]).repeat(101),
                }
            ]
            targets = [{"boxes": torch.tensor([[0, 0, 1, 1]]), "labels": torch.tensor([0])}]
        else:
            preds, targets = _generate_random_segm_input("cpu", 1, 101, 10, False)

        metric = MeanAveragePrecision(iou_type=iou_type, backend=backend)
        metric.warn_on_many_detections = warn_on_many_detections

        if warn_on_many_detections:
            with pytest.warns(UserWarning, match="Encountered more than 100 detections in a single image.*"):
                metric.update(preds, targets)
        else:
            assert len(recwarn) == 0

    @pytest.mark.parametrize(
        ("preds", "target", "expected_iou_len", "iou_keys", "precision_shape", "recall_shape", "scores_shape"),
        [
            (
                [
                    [
                        {
                            "boxes": torch.tensor([[0.5, 0.5, 1, 1]]),
                            "scores": torch.tensor([1.0]),
                            "labels": torch.tensor([0]),
                        }
                    ]
                ],
                [[{"boxes": torch.tensor([[0, 0, 1, 1]]), "labels": torch.tensor([0])}]],
                1,  # 1 image x 1 class = 1
                [(0, 0)],
                (10, 101, 1, 4, 3),
                (10, 1, 4, 3),
                (10, 101, 1, 4, 3),
            ),
            (
                _inputs["preds"],
                _inputs["target"],
                24,  # 4 images x 6 classes = 24
                list(product([0, 1, 2, 3], [0, 1, 2, 3, 4, 49])),
                (10, 101, 6, 4, 3),
                (10, 6, 4, 3),
                (10, 101, 6, 4, 3),
            ),
        ],
    )
    def test_for_extended_stats(
        self, preds, target, expected_iou_len, iou_keys, precision_shape, recall_shape, scores_shape, backend
    ):
        """Test that extended stats are computed correctly."""
        metric = MeanAveragePrecision(extended_summary=True, backend=backend)
        for p, t in zip(preds, target):
            metric.update(p, t)
        result = metric.compute()

        ious = result["ious"]

        assert isinstance(ious, dict)
        assert len(ious) == expected_iou_len
        for key in ious:
            assert key in iou_keys

        precision = result["precision"]
        assert isinstance(precision, Tensor)
        assert precision.shape == precision_shape

        recall = result["recall"]
        assert isinstance(recall, Tensor)
        assert recall.shape == recall_shape

        scores = result["scores"]
        assert isinstance(scores, Tensor)
        assert scores.shape == scores_shape

    @pytest.mark.parametrize("class_metrics", [False, True])
    def test_average_argument(self, class_metrics, backend):
        """Test that average argument works.

        Calculating macro on inputs that only have one label should be the same as micro. Calculating class metrics
        should be the same regardless of average argument.

        """
        if class_metrics:
            _preds = _inputs["preds"]
            _target = _inputs["target"]
        else:
            _preds = apply_to_collection(deepcopy(_inputs["preds"]), IntTensor, lambda x: torch.ones_like(x))
            _target = apply_to_collection(deepcopy(_inputs["target"]), IntTensor, lambda x: torch.ones_like(x))

        metric_micro = MeanAveragePrecision(average="micro", class_metrics=class_metrics, backend=backend)
        metric_micro.update(deepcopy(_inputs["preds"][0]), deepcopy(_inputs["target"][0]))
        metric_micro.update(deepcopy(_inputs["preds"][1]), deepcopy(_inputs["target"][1]))
        result_micro = metric_micro.compute()

        metric_macro = MeanAveragePrecision(average="macro", class_metrics=class_metrics, backend=backend)
        metric_macro.update(_preds[0], _target[0])
        metric_macro.update(_preds[1], _target[1])
        result_macro = metric_macro.compute()

        if class_metrics:
            print(result_macro["map_per_class"], result_micro["map_per_class"])
            assert torch.allclose(result_macro["map_per_class"], result_micro["map_per_class"])
            assert torch.allclose(result_macro["mar_100_per_class"], result_micro["mar_100_per_class"])
        else:
            for key in result_macro:
                if key == "classes":
                    continue
                assert torch.allclose(result_macro[key], result_micro[key])

    def test_many_detection_thresholds(self, backend):
        """Test how metric behaves when there are many detection thresholds.

        Known to fail with the default pycocotools backend.
        See issue: https://github.com/Lightning-AI/torchmetrics/issues/1153

        """
        preds = [
            {
                "boxes": torch.tensor([[258.0, 41.0, 606.0, 285.0]]),
                "scores": torch.tensor([0.536]),
                "labels": torch.tensor([0]),
            }
        ]
        target = [
            {
                "boxes": torch.tensor([[214.0, 41.0, 562.0, 285.0]]),
                "labels": torch.tensor([0]),
            }
        ]
        metric = MeanAveragePrecision(max_detection_thresholds=[1, 10, 1000], backend=backend)
        res = metric(preds, target)

        if backend == "pycocotools":
            assert round(res["map"].item(), 5) != 0.6
        else:
            assert round(res["map"].item(), 5) == 0.6

        assert "mar_1" in res
        assert "mar_10" in res
        assert "mar_1000" in res

    @pytest.mark.parametrize("max_detection_thresholds", [[1, 10], [1, 10, 50, 100]])
    def test_with_more_and_less_detection_thresholds(self, max_detection_thresholds, backend):
        """Test how metric is working when list of max detection thresholds is not 3.

        This is a known limitation of the pycocotools where values are hardcoded to expect at least 3 elements
        https://github.com/ppwwyyxx/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py#L461

        """
        with pytest.raises(
            ValueError, match="When providing a list of max detection thresholds it should have length 3.*"
        ):
            MeanAveragePrecision(max_detection_thresholds=max_detection_thresholds, backend=backend)
