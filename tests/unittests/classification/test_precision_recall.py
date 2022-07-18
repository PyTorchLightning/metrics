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
from functools import partial
from typing import Callable, Optional

import numpy as np
import pytest
import torch
from scipy.special import expit as sigmoid
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
from sklearn.metrics import precision_score as sk_precision_score
from sklearn.metrics import recall_score as sk_recall_score
from torch import Tensor, tensor

from torchmetrics.classification.precision_recall import (
    BinaryPrecision,
    BinaryRecall,
    MulticlassPrecision,
    MulticlassRecall,
    MultilabelPrecision,
    MultilabelRecall,
)
from torchmetrics.functional.classification.precision_recall import (
    binary_precision,
    binary_recall,
    multiclass_precision,
    multiclass_recall,
    multilabel_precision,
    multilabel_recall,
)
from torchmetrics.utilities.compute import _safe_divide
from torchmetrics.utilities.imports import _TORCH_GREATER_EQUAL_1_6
from unittests.classification.inputs import _binary_cases, _multiclass_cases, _multilabel_cases
from unittests.helpers import seed_all
from unittests.helpers.testers import (
    NUM_BATCHES,
    NUM_CLASSES,
    THRESHOLD,
    MetricTester,
    inject_ignore_index,
    remove_ignore_index,
)

seed_all(42)


def _sk_precision_recall_binary(preds, target, sk_fn, ignore_index, multidim_average):
    if multidim_average == "global":
        preds = preds.view(-1).numpy()
        target = target.view(-1).numpy()
    else:
        preds = preds.numpy()
        target = target.numpy()

    if np.issubdtype(preds.dtype, np.floating):
        if not ((0 < preds) & (preds < 1)).all():
            preds = sigmoid(preds)
        preds = (preds >= THRESHOLD).astype(np.uint8)

    if multidim_average == "global":
        target, preds = remove_ignore_index(target, preds, ignore_index)
        return sk_fn(target, preds)
    else:
        res = []
        for pred, true in zip(preds, target):
            pred = pred.flatten()
            true = true.flatten()
            true, pred = remove_ignore_index(true, pred, ignore_index)
            res.append(sk_fn(true, pred))
        return np.stack(res)


@pytest.mark.parametrize("input", _binary_cases)
@pytest.mark.parametrize(
    "module, functional, compare",
    [
        (BinaryPrecision, binary_precision, sk_precision_score),
        (BinaryRecall, binary_recall, sk_recall_score),
    ],
    ids=["precision", "recall"],
)
class TestBinaryPrecisionRecall(MetricTester):
    @pytest.mark.parametrize("ignore_index", [None, 0, -1])
    @pytest.mark.parametrize("multidim_average", ["global", "samplewise"])
    @pytest.mark.parametrize("ddp", [False, True])
    def test_binary_precision_recall(self, ddp, input, module, functional, compare, ignore_index, multidim_average):
        preds, target = input
        if ignore_index == -1:
            target = inject_ignore_index(target, ignore_index)
        if multidim_average == "samplewise" and preds.ndim < 3:
            pytest.skip("samplewise and non-multidim arrays are not valid")
        if multidim_average == "samplewise" and ddp:
            pytest.skip("samplewise and ddp give different order than non ddp")

        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=module,
            sk_metric=partial(
                _sk_precision_recall_binary, sk_fn=compare, ignore_index=ignore_index, multidim_average=multidim_average
            ),
            metric_args={"threshold": THRESHOLD, "ignore_index": ignore_index, "multidim_average": multidim_average},
        )

    @pytest.mark.parametrize("ignore_index", [None, 0, -1])
    @pytest.mark.parametrize("multidim_average", ["global", "samplewise"])
    def test_binary_precision_recall_functional(
        self, input, module, functional, compare, ignore_index, multidim_average
    ):
        preds, target = input
        if ignore_index == -1:
            target = inject_ignore_index(target, ignore_index)
        if multidim_average == "samplewise" and preds.ndim < 3:
            pytest.skip("samplewise and non-multidim arrays are not valid")

        self.run_functional_metric_test(
            preds=preds,
            target=target,
            metric_functional=functional,
            sk_metric=partial(
                _sk_precision_recall_binary, sk_fn=compare, ignore_index=ignore_index, multidim_average=multidim_average
            ),
            metric_args={
                "threshold": THRESHOLD,
                "ignore_index": ignore_index,
                "multidim_average": multidim_average,
            },
        )

    def test_binary_precision_recall_differentiability(self, input, module, functional, compare):
        preds, target = input
        self.run_differentiability_test(
            preds=preds,
            target=target,
            metric_module=module,
            metric_functional=functional,
            metric_args={"threshold": THRESHOLD},
        )

    @pytest.mark.parametrize("dtype", [torch.half, torch.double])
    def test_binary_precision_recall_half_cpu(self, input, module, functional, compare, dtype):
        preds, target = input
        if dtype == torch.half and not _TORCH_GREATER_EQUAL_1_6:
            pytest.xfail(reason="half support of core ops not support before pytorch v1.6")
        if (preds < 0).any() and dtype == torch.half:
            pytest.xfail(reason="torch.sigmoid in metric does not support cpu + half precision")
        self.run_precision_test_cpu(
            preds=preds,
            target=target,
            metric_module=module,
            metric_functional=functional,
            metric_args={"threshold": THRESHOLD},
            dtype=dtype,
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
    @pytest.mark.parametrize("dtype", [torch.half, torch.double])
    def test_binary_precision_recall_half_gpu(self, input, module, functional, compare, dtype):
        preds, target = input
        self.run_precision_test_gpu(
            preds=preds,
            target=target,
            metric_module=module,
            metric_functional=functional,
            metric_args={"threshold": THRESHOLD},
            dtype=dtype,
        )


def _sk_precision_recall_multiclass(preds, target, sk_fn, ignore_index, multidim_average, average):
    if preds.ndim == target.ndim + 1:
        preds = torch.argmax(preds, 1)
    if multidim_average == "global":
        preds = preds.numpy().flatten()
        target = target.numpy().flatten()
        target, preds = remove_ignore_index(target, preds, ignore_index)
        return sk_fn(target, preds, average=average)
    else:
        preds = preds.numpy()
        target = target.numpy()
        res = []
        for pred, true in zip(preds, target):
            pred = pred.flatten()
            true = true.flatten()
            true, pred = remove_ignore_index(true, pred, ignore_index)
            res.append(sk_fn(true, pred, average=average, labels=list(range(NUM_CLASSES))))
        return np.stack(res, 0)


@pytest.mark.parametrize("input", _multiclass_cases)
@pytest.mark.parametrize(
    "module, functional, compare",
    [
        (MulticlassPrecision, multiclass_precision, sk_precision_score),
        (MulticlassRecall, multiclass_recall, sk_recall_score),
    ],
    ids=["precision", "recall"],
)
class TestMulticlassPrecisionRecall(MetricTester):
    @pytest.mark.parametrize("ignore_index", [None, 0, -1])
    @pytest.mark.parametrize("multidim_average", ["global", "samplewise"])
    @pytest.mark.parametrize("average", ["micro", "macro", "weighted", None])
    @pytest.mark.parametrize("ddp", [True, False])
    def test_multiclass_precision_recall(
        self, ddp, input, module, functional, compare, ignore_index, multidim_average, average
    ):
        preds, target = input
        if ignore_index == -1:
            target = inject_ignore_index(target, ignore_index)
        if multidim_average == "samplewise" and target.ndim < 3:
            pytest.skip("samplewise and non-multidim arrays are not valid")
        if multidim_average == "samplewise" and ddp:
            pytest.skip("samplewise and ddp give different order than non ddp")

        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=module,
            sk_metric=partial(
                _sk_precision_recall_multiclass,
                sk_fn=compare,
                ignore_index=ignore_index,
                multidim_average=multidim_average,
                average=average,
            ),
            metric_args={
                "ignore_index": ignore_index,
                "multidim_average": multidim_average,
                "average": average,
                "num_classes": NUM_CLASSES,
            },
        )

    @pytest.mark.parametrize("ignore_index", [None, 0, -1])
    @pytest.mark.parametrize("multidim_average", ["global", "samplewise"])
    @pytest.mark.parametrize("average", ["micro", "macro", "weighted", None])
    def test_multiclass_precision_recall_functional(
        self, input, module, functional, compare, ignore_index, multidim_average, average
    ):
        preds, target = input
        if ignore_index == -1:
            target = inject_ignore_index(target, ignore_index)
        if multidim_average == "samplewise" and target.ndim < 3:
            pytest.skip("samplewise and non-multidim arrays are not valid")

        self.run_functional_metric_test(
            preds=preds,
            target=target,
            metric_functional=functional,
            sk_metric=partial(
                _sk_precision_recall_multiclass,
                sk_fn=compare,
                ignore_index=ignore_index,
                multidim_average=multidim_average,
                average=average,
            ),
            metric_args={
                "ignore_index": ignore_index,
                "multidim_average": multidim_average,
                "average": average,
                "num_classes": NUM_CLASSES,
            },
        )

    def test_multiclass_precision_recall_differentiability(self, input, module, functional, compare):
        preds, target = input
        self.run_differentiability_test(
            preds=preds,
            target=target,
            metric_module=module,
            metric_functional=functional,
            metric_args={"num_classes": NUM_CLASSES},
        )

    @pytest.mark.parametrize("dtype", [torch.half, torch.double])
    def test_multiclass_precision_recall_half_cpu(self, input, module, functional, compare, dtype):
        preds, target = input
        if dtype == torch.half and not _TORCH_GREATER_EQUAL_1_6:
            pytest.xfail(reason="half support of core ops not support before pytorch v1.6")
        if (preds < 0).any() and dtype == torch.half:
            pytest.xfail(reason="torch.sigmoid in metric does not support cpu + half precision")
        self.run_precision_test_cpu(
            preds=preds,
            target=target,
            metric_module=module,
            metric_functional=functional,
            metric_args={"num_classes": NUM_CLASSES},
            dtype=dtype,
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
    @pytest.mark.parametrize("dtype", [torch.half, torch.double])
    def test_multiclass_precision_recall_half_gpu(self, input, module, functional, compare, dtype):
        preds, target = input
        self.run_precision_test_gpu(
            preds=preds,
            target=target,
            metric_module=module,
            metric_functional=functional,
            metric_args={"num_classes": NUM_CLASSES},
            dtype=dtype,
        )


_mc_k_target = tensor([0, 1, 2])
_mc_k_preds = tensor([[0.35, 0.4, 0.25], [0.1, 0.5, 0.4], [0.2, 0.1, 0.7]])


@pytest.mark.parametrize(
    "metric_class, metric_fn", [(MulticlassPrecision, multiclass_precision), (MulticlassRecall, multiclass_recall)]
)
@pytest.mark.parametrize(
    "k, preds, target, average, expected_prec, expected_recall",
    [
        (1, _mc_k_preds, _mc_k_target, "micro", tensor(2 / 3), tensor(2 / 3)),
        (2, _mc_k_preds, _mc_k_target, "micro", tensor(1 / 2), tensor(1.0)),
    ],
)
def test_top_k(
    metric_class,
    metric_fn,
    k: int,
    preds: Tensor,
    target: Tensor,
    average: str,
    expected_prec: Tensor,
    expected_recall: Tensor,
):
    """A simple test to check that top_k works as expected."""

    class_metric = metric_class(top_k=k, average=average, num_classes=3)
    class_metric.update(preds, target)

    if metric_class.__name__ == "MulticlassPrecision":
        result = expected_prec
    else:
        result = expected_recall

    assert torch.equal(class_metric.compute(), result)
    assert torch.equal(metric_fn(preds, target, top_k=k, average=average, num_classes=3), result)


def _sk_precision_recall_multilabel(preds, target, sk_fn, ignore_index, multidim_average, average):
    preds = preds.numpy()
    target = target.numpy()
    if np.issubdtype(preds.dtype, np.floating):
        if not ((0 < preds) & (preds < 1)).all():
            preds = sigmoid(preds)
        preds = (preds >= THRESHOLD).astype(np.uint8)
    preds = preds.reshape(*preds.shape[:2], -1)
    target = target.reshape(*target.shape[:2], -1)
    if ignore_index is None and multidim_average == "global":
        return sk_fn(
            target.transpose(0, 2, 1).reshape(-1, NUM_CLASSES),
            preds.transpose(0, 2, 1).reshape(-1, NUM_CLASSES),
            average=average,
        )
    elif multidim_average == "global":
        if average == "micro":
            preds = preds.flatten()
            target = target.flatten()
            target, preds = remove_ignore_index(target, preds, ignore_index)
            return sk_fn(target, preds)

        precision_recall, weights = [], []
        for i in range(preds.shape[1]):
            pred, true = preds[:, i].flatten(), target[:, i].flatten()
            true, pred = remove_ignore_index(true, pred, ignore_index)
            precision_recall.append(sk_fn(true, pred))
            confmat = sk_confusion_matrix(true, pred, labels=[0, 1])
            weights.append(confmat[1, 1] + confmat[1, 0])
        res = np.stack(precision_recall, axis=0)

        if average == "macro":
            return res.mean(0)
        elif average == "weighted":
            weights = np.stack(weights, 0)
            return _safe_divide(weights * res, weights.sum(-1, keepdims=True)).sum(-1)
        elif average is None or average == "none":
            return res
    else:
        precision_recall, weights = [], []
        for i in range(preds.shape[0]):
            if average == "micro":
                pred, true = preds[i].flatten(), target[i].flatten()
                true, pred = remove_ignore_index(true, pred, ignore_index)
                precision_recall.append(sk_fn(true, pred))
                confmat = sk_confusion_matrix(true, pred, labels=[0, 1])
                weights.append(confmat[1, 1] + confmat[1, 0])
            else:
                scores, w = [], []
                for j in range(preds.shape[1]):
                    pred, true = preds[i, j], target[i, j]
                    true, pred = remove_ignore_index(true, pred, ignore_index)
                    scores.append(sk_fn(true, pred))
                    confmat = sk_confusion_matrix(true, pred, labels=[0, 1])
                    w.append(confmat[1, 1] + confmat[1, 0])
                precision_recall.append(np.stack(scores))
                weights.append(np.stack(w))
        if average == "micro":
            return np.array(precision_recall)
        res = np.stack(precision_recall, 0)
        if average == "macro":
            return res.mean(-1)
        elif average == "weighted":
            weights = np.stack(weights, 0)
            return _safe_divide(weights * res, weights.sum(-1, keepdims=True)).sum(-1)
        elif average is None or average == "none":
            return res


@pytest.mark.parametrize("input", _multilabel_cases)
@pytest.mark.parametrize(
    "module, functional, compare",
    [
        (MultilabelPrecision, multilabel_precision, sk_precision_score),
        (MultilabelRecall, multilabel_recall, sk_recall_score),
    ],
    ids=["precision", "recall"],
)
class TestMultilabelPrecisionRecall(MetricTester):
    @pytest.mark.parametrize("ddp", [True, False])
    @pytest.mark.parametrize("ignore_index", [None, 0, -1])
    @pytest.mark.parametrize("multidim_average", ["global", "samplewise"])
    @pytest.mark.parametrize("average", ["micro", "macro", "weighted", None])
    def test_multilabel_precision_recall(
        self, ddp, input, module, functional, compare, ignore_index, multidim_average, average
    ):
        preds, target = input
        if ignore_index == -1:
            target = inject_ignore_index(target, ignore_index)
        if multidim_average == "samplewise" and preds.ndim < 4:
            pytest.skip("samplewise and non-multidim arrays are not valid")
        if multidim_average == "samplewise" and ddp:
            pytest.skip("samplewise and ddp give different order than non ddp")

        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=module,
            sk_metric=partial(
                _sk_precision_recall_multilabel,
                sk_fn=compare,
                ignore_index=ignore_index,
                multidim_average=multidim_average,
                average=average,
            ),
            metric_args={
                "num_labels": NUM_CLASSES,
                "threshold": THRESHOLD,
                "ignore_index": ignore_index,
                "multidim_average": multidim_average,
                "average": average,
            },
        )

    @pytest.mark.parametrize("ignore_index", [None, 0, -1])
    @pytest.mark.parametrize("multidim_average", ["global", "samplewise"])
    @pytest.mark.parametrize("average", ["micro", "macro", "weighted", None])
    def test_multilabel_precision_recall_functional(
        self, input, module, functional, compare, ignore_index, multidim_average, average
    ):
        preds, target = input
        if ignore_index == -1:
            target = inject_ignore_index(target, ignore_index)
        if multidim_average == "samplewise" and preds.ndim < 4:
            pytest.skip("samplewise and non-multidim arrays are not valid")

        self.run_functional_metric_test(
            preds=preds,
            target=target,
            metric_functional=functional,
            sk_metric=partial(
                _sk_precision_recall_multilabel,
                sk_fn=compare,
                ignore_index=ignore_index,
                multidim_average=multidim_average,
                average=average,
            ),
            metric_args={
                "num_labels": NUM_CLASSES,
                "threshold": THRESHOLD,
                "ignore_index": ignore_index,
                "multidim_average": multidim_average,
                "average": average,
            },
        )

    def test_multilabel_precision_recall_differentiability(self, input, module, functional, compare):
        preds, target = input
        self.run_differentiability_test(
            preds=preds,
            target=target,
            metric_module=module,
            metric_functional=functional,
            metric_args={"num_labels": NUM_CLASSES, "threshold": THRESHOLD},
        )

    @pytest.mark.parametrize("dtype", [torch.half, torch.double])
    def test_multilabel_precision_recall_half_cpu(self, input, module, functional, compare, dtype):
        preds, target = input
        if dtype == torch.half and not _TORCH_GREATER_EQUAL_1_6:
            pytest.xfail(reason="half support of core ops not support before pytorch v1.6")
        if (preds < 0).any() and dtype == torch.half:
            pytest.xfail(reason="torch.sigmoid in metric does not support cpu + half precision")
        self.run_precision_test_cpu(
            preds=preds,
            target=target,
            metric_module=module,
            metric_functional=functional,
            metric_args={"num_labels": NUM_CLASSES, "threshold": THRESHOLD},
            dtype=dtype,
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
    @pytest.mark.parametrize("dtype", [torch.half, torch.double])
    def test_multilabel_precision_recall_half_gpu(self, input, module, functional, compare, dtype):
        preds, target = input
        self.run_precision_test_gpu(
            preds=preds,
            target=target,
            metric_module=module,
            metric_functional=functional,
            metric_args={"num_labels": NUM_CLASSES, "threshold": THRESHOLD},
            dtype=dtype,
        )


# -------------------------- Old stuff --------------------------

# def _sk_prec_recall(preds, target, sk_fn, num_classes, average, multiclass, ignore_index, mdmc_average=None):
#     # todo: `mdmc_average` is unused
#     if average == "none":
#         average = None
#     if num_classes == 1:
#         average = "binary"

#     labels = list(range(num_classes))
#     try:
#         labels.remove(ignore_index)
#     except ValueError:
#         pass

#     sk_preds, sk_target, _ = _input_format_classification(
#         preds, target, THRESHOLD, num_classes=num_classes, multiclass=multiclass
#     )
#     sk_preds, sk_target = sk_preds.numpy(), sk_target.numpy()

#     sk_scores = sk_fn(sk_target, sk_preds, average=average, zero_division=0, labels=labels)

#     if len(labels) != num_classes and not average:
#         sk_scores = np.insert(sk_scores, ignore_index, np.nan)

#     return sk_scores


# def _sk_prec_recall_multidim_multiclass(
#     preds, target, sk_fn, num_classes, average, multiclass, ignore_index, mdmc_average
# ):
#     preds, target, _ = _input_format_classification(
#         preds, target, threshold=THRESHOLD, num_classes=num_classes, multiclass=multiclass
#     )

#     if mdmc_average == "global":
#         preds = torch.transpose(preds, 1, 2).reshape(-1, preds.shape[1])
#         target = torch.transpose(target, 1, 2).reshape(-1, target.shape[1])

#         return _sk_prec_recall(preds, target, sk_fn, num_classes, average, False, ignore_index)
#     if mdmc_average == "samplewise":
#         scores = []

#         for i in range(preds.shape[0]):
#             pred_i = preds[i, ...].T
#             target_i = target[i, ...].T
#             scores_i = _sk_prec_recall(pred_i, target_i, sk_fn, num_classes, average, False, ignore_index)

#             scores.append(np.expand_dims(scores_i, 0))

#         return np.concatenate(scores).mean(axis=0)


# @pytest.mark.parametrize("metric, fn_metric", [(Precision, precision), (Recall, recall)])
# @pytest.mark.parametrize(
#     "average, mdmc_average, num_classes, ignore_index, match_str",
#     [
#         ("wrong", None, None, None, "`average`"),
#         ("micro", "wrong", None, None, "`mdmc"),
#         ("macro", None, None, None, "number of classes"),
#         ("macro", None, 1, 0, "ignore_index"),
#     ],
# )
# def test_wrong_params(metric, fn_metric, average, mdmc_average, num_classes, ignore_index, match_str):
#     with pytest.raises(ValueError, match=match_str):
#         metric(
#             average=average,
#             mdmc_average=mdmc_average,
#             num_classes=num_classes,
#             ignore_index=ignore_index,
#         )

#     with pytest.raises(ValueError, match=match_str):
#         fn_metric(
#             _input_binary.preds[0],
#             _input_binary.target[0],
#             average=average,
#             mdmc_average=mdmc_average,
#             num_classes=num_classes,
#             ignore_index=ignore_index,
#         )

#     with pytest.raises(ValueError, match=match_str):
#         precision_recall(
#             _input_binary.preds[0],
#             _input_binary.target[0],
#             average=average,
#             mdmc_average=mdmc_average,
#             num_classes=num_classes,
#             ignore_index=ignore_index,
#         )


# @pytest.mark.parametrize("metric_class, metric_fn", [(Recall, recall), (Precision, precision)])
# def test_zero_division(metric_class, metric_fn):
#     """Test that zero_division works correctly (currently should just set to 0)."""

#     preds = tensor([0, 2, 1, 1])
#     target = tensor([2, 1, 2, 1])

#     cl_metric = metric_class(average="none", num_classes=3)
#     cl_metric(preds, target)

#     result_cl = cl_metric.compute()
#     result_fn = metric_fn(preds, target, average="none", num_classes=3)

#     assert result_cl[0] == result_fn[0] == 0


# @pytest.mark.parametrize("metric_class, metric_fn", [(Recall, recall), (Precision, precision)])
# def test_no_support(metric_class, metric_fn):
#     """This tests a rare edge case, where there is only one class present.

#     in target, and ignore_index is set to exactly that class - and the
#     average method is equal to 'weighted'.

#     This would mean that the sum of weights equals zero, and would, without
#     taking care of this case, return NaN. However, the reduction function
#     should catch that and set the metric to equal the value of zero_division
#     in this case (zero_division is for now not configurable and equals 0).
#     """

#     preds = tensor([1, 1, 0, 0])
#     target = tensor([0, 0, 0, 0])

#     cl_metric = metric_class(average="weighted", num_classes=2, ignore_index=0)
#     cl_metric(preds, target)

#     result_cl = cl_metric.compute()
#     result_fn = metric_fn(preds, target, average="weighted", num_classes=2, ignore_index=0)

#     assert result_cl == result_fn == 0


# @pytest.mark.parametrize(
#     "metric_class, metric_fn, sk_fn", [(Recall, recall, recall_score), (Precision, precision, precision_score)]
# )
# @pytest.mark.parametrize("average", ["micro", "macro", None, "weighted", "samples"])
# @pytest.mark.parametrize("ignore_index", [None, 0])
# @pytest.mark.parametrize(
#     "preds, target, num_classes, multiclass, mdmc_average, sk_wrapper",
#     [
#         (_input_binary_logits.preds, _input_binary_logits.target, 1, None, None, _sk_prec_recall),
#         (_input_binary_prob.preds, _input_binary_prob.target, 1, None, None, _sk_prec_recall),
#         (_input_binary.preds, _input_binary.target, 1, False, None, _sk_prec_recall),
#         (_input_mlb_logits.preds, _input_mlb_logits.target, NUM_CLASSES, None, None, _sk_prec_recall),
#         (_input_mlb_prob.preds, _input_mlb_prob.target, NUM_CLASSES, None, None, _sk_prec_recall),
#         (_input_mlb.preds, _input_mlb.target, NUM_CLASSES, False, None, _sk_prec_recall),
#         (_input_mcls_logits.preds, _input_mcls_logits.target, NUM_CLASSES, None, None, _sk_prec_recall),
#         (_input_mcls_prob.preds, _input_mcls_prob.target, NUM_CLASSES, None, None, _sk_prec_recall),
#         (_input_mcls.preds, _input_mcls.target, NUM_CLASSES, None, None, _sk_prec_recall),
#         (_input_mdmc.preds, _input_mdmc.target, NUM_CLASSES, None, "global", _sk_prec_recall_multidim_multiclass),
#         (
#             _input_mdmc_prob.preds,
#             _input_mdmc_prob.target,
#             NUM_CLASSES,
#             None,
#             "global",
#             _sk_prec_recall_multidim_multiclass,
#         ),
#         (_input_mdmc.preds, _input_mdmc.target, NUM_CLASSES, None, "samplewise", _sk_prec_recall_multidim_multiclass),
#         (
#             _input_mdmc_prob.preds,
#             _input_mdmc_prob.target,
#             NUM_CLASSES,
#             None,
#             "samplewise",
#             _sk_prec_recall_multidim_multiclass,
#         ),
#     ],
# )
# class TestPrecisionRecall(MetricTester):
#     @pytest.mark.parametrize("ddp", [False, True])
#     @pytest.mark.parametrize("dist_sync_on_step", [False])
#     def test_precision_recall_class(
#         self,
#         ddp: bool,
#         dist_sync_on_step: bool,
#         preds: Tensor,
#         target: Tensor,
#         sk_wrapper: Callable,
#         metric_class: Metric,
#         metric_fn: Callable,
#         sk_fn: Callable,
#         multiclass: Optional[bool],
#         num_classes: Optional[int],
#         average: str,
#         mdmc_average: Optional[str],
#         ignore_index: Optional[int],
#     ):
#         # todo: `metric_fn` is unused
#         if num_classes == 1 and average != "micro":
#             pytest.skip("Only test binary data for 'micro' avg (equivalent of 'binary' in sklearn)")

#         if ignore_index is not None and preds.ndim == 2:
#             pytest.skip("Skipping ignore_index test with binary inputs.")

#         if average == "weighted" and ignore_index is not None and mdmc_average is not None:
#             pytest.skip("Ignore special case where we are ignoring entire sample for 'weighted' average")

#         self.run_class_metric_test(
#             ddp=ddp,
#             preds=preds,
#             target=target,
#             metric_class=metric_class,
#             sk_metric=partial(
#                 sk_wrapper,
#                 sk_fn=sk_fn,
#                 average=average,
#                 num_classes=num_classes,
#                 multiclass=multiclass,
#                 ignore_index=ignore_index,
#                 mdmc_average=mdmc_average,
#             ),
#             dist_sync_on_step=dist_sync_on_step,
#             metric_args={
#                 "num_classes": num_classes,
#                 "average": average,
#                 "threshold": THRESHOLD,
#                 "multiclass": multiclass,
#                 "ignore_index": ignore_index,
#                 "mdmc_average": mdmc_average,
#             },
#         )

#     def test_precision_recall_fn(
#         self,
#         preds: Tensor,
#         target: Tensor,
#         sk_wrapper: Callable,
#         metric_class: Metric,
#         metric_fn: Callable,
#         sk_fn: Callable,
#         multiclass: Optional[bool],
#         num_classes: Optional[int],
#         average: str,
#         mdmc_average: Optional[str],
#         ignore_index: Optional[int],
#     ):
#         # todo: `metric_class` is unused
#         if num_classes == 1 and average != "micro":
#             pytest.skip("Only test binary data for 'micro' avg (equivalent of 'binary' in sklearn)")

#         if ignore_index is not None and preds.ndim == 2:
#             pytest.skip("Skipping ignore_index test with binary inputs.")

#         if average == "weighted" and ignore_index is not None and mdmc_average is not None:
#             pytest.skip("Ignore special case where we are ignoring entire sample for 'weighted' average")

#         self.run_functional_metric_test(
#             preds,
#             target,
#             metric_functional=metric_fn,
#             sk_metric=partial(
#                 sk_wrapper,
#                 sk_fn=sk_fn,
#                 average=average,
#                 num_classes=num_classes,
#                 multiclass=multiclass,
#                 ignore_index=ignore_index,
#                 mdmc_average=mdmc_average,
#             ),
#             metric_args={
#                 "num_classes": num_classes,
#                 "average": average,
#                 "threshold": THRESHOLD,
#                 "multiclass": multiclass,
#                 "ignore_index": ignore_index,
#                 "mdmc_average": mdmc_average,
#             },
#         )

#     def test_precision_recall_differentiability(
#         self,
#         preds: Tensor,
#         target: Tensor,
#         sk_wrapper: Callable,
#         metric_class: Metric,
#         metric_fn: Callable,
#         sk_fn: Callable,
#         multiclass: Optional[bool],
#         num_classes: Optional[int],
#         average: str,
#         mdmc_average: Optional[str],
#         ignore_index: Optional[int],
#     ):
#         # todo: `metric_class` is unused
#         if num_classes == 1 and average != "micro":
#             pytest.skip("Only test binary data for 'micro' avg (equivalent of 'binary' in sklearn)")

#         if ignore_index is not None and preds.ndim == 2:
#             pytest.skip("Skipping ignore_index test with binary inputs.")

#         if average == "weighted" and ignore_index is not None and mdmc_average is not None:
#             pytest.skip("Ignore special case where we are ignoring entire sample for 'weighted' average")

#         self.run_differentiability_test(
#             preds=preds,
#             target=target,
#             metric_module=metric_class,
#             metric_functional=metric_fn,
#             metric_args={
#                 "num_classes": num_classes,
#                 "average": average,
#                 "threshold": THRESHOLD,
#                 "multiclass": multiclass,
#                 "ignore_index": ignore_index,
#                 "mdmc_average": mdmc_average,
#             },
#         )


# @pytest.mark.parametrize("average", ["micro", "macro", None, "weighted", "samples"])
# def test_precision_recall_joint(average):
#     """A simple test of the joint precision_recall metric.

#     No need to test this thorougly, as it is just a combination of precision and recall, which are already tested
#     thoroughly.
#     """

#     precision_result = precision(
#         _input_mcls_prob.preds[0], _input_mcls_prob.target[0], average=average, num_classes=NUM_CLASSES
#     )
#     recall_result = recall(
#         _input_mcls_prob.preds[0], _input_mcls_prob.target[0], average=average, num_classes=NUM_CLASSES
#     )

#     prec_recall_result = precision_recall(
#         _input_mcls_prob.preds[0], _input_mcls_prob.target[0], average=average, num_classes=NUM_CLASSES
#     )

#     assert torch.equal(precision_result, prec_recall_result[0])
#     assert torch.equal(recall_result, prec_recall_result[1])


# _mc_k_target = tensor([0, 1, 2])
# _mc_k_preds = tensor([[0.35, 0.4, 0.25], [0.1, 0.5, 0.4], [0.2, 0.1, 0.7]])
# _ml_k_target = tensor([[0, 1, 0], [1, 1, 0], [0, 0, 0]])
# _ml_k_preds = tensor([[0.9, 0.2, 0.75], [0.1, 0.7, 0.8], [0.6, 0.1, 0.7]])


# @pytest.mark.parametrize("metric_class, metric_fn", [(Recall, recall), (Precision, precision)])
# @pytest.mark.parametrize(
#     "k, preds, target, average, expected_prec, expected_recall",
#     [
#         (1, _mc_k_preds, _mc_k_target, "micro", tensor(2 / 3), tensor(2 / 3)),
#         (2, _mc_k_preds, _mc_k_target, "micro", tensor(1 / 2), tensor(1.0)),
#         (1, _ml_k_preds, _ml_k_target, "micro", tensor(0.0), tensor(0.0)),
#         (2, _ml_k_preds, _ml_k_target, "micro", tensor(1 / 6), tensor(1 / 3)),
#     ],
# )
# def test_top_k(
#     metric_class,
#     metric_fn,
#     k: int,
#     preds: Tensor,
#     target: Tensor,
#     average: str,
#     expected_prec: Tensor,
#     expected_recall: Tensor,
# ):
#     """A simple test to check that top_k works as expected.

#     Just a sanity check, the tests in StatScores should already guarantee the correctness of results.
#     """

#     class_metric = metric_class(top_k=k, average=average, num_classes=3)
#     class_metric.update(preds, target)

#     if metric_class.__name__ == "Precision":
#         result = expected_prec
#     else:
#         result = expected_recall

#     assert torch.equal(class_metric.compute(), result)
#     assert torch.equal(metric_fn(preds, target, top_k=k, average=average, num_classes=3), result)


# @pytest.mark.parametrize("metric_class, metric_fn", [(Precision, precision), (Recall, recall)])
# @pytest.mark.parametrize(
#     "ignore_index, expected", [(None, torch.tensor([1.0, np.nan])), (0, torch.tensor([np.nan, np.nan]))]
# )
# def test_class_not_present(metric_class, metric_fn, ignore_index, expected):
#     """This tests that when metric is computed per class and a given class is not present in both the `preds` and
#     `target`, the resulting score is `nan`."""
#     preds = torch.tensor([0, 0, 0])
#     target = torch.tensor([0, 0, 0])
#     num_classes = 2

#     # test functional
#     result_fn = metric_fn(preds, target, average=AverageMethod.NONE, num_classes=num_classes, ignore_index=ignore_index)
#     assert torch.allclose(expected, result_fn, equal_nan=True)

#     # test class
#     cl_metric = metric_class(average=AverageMethod.NONE, num_classes=num_classes, ignore_index=ignore_index)
#     cl_metric(preds, target)
#     result_cl = cl_metric.compute()
#     assert torch.allclose(expected, result_cl, equal_nan=True)


# @pytest.mark.parametrize("average", ["micro", "macro", "weighted"])
# @pytest.mark.parametrize(
#     "metric_class, metric_functional, sk_fn", [(Precision, precision, precision_score), (Recall, recall, recall_score)]
# )
# def test_same_input(metric_class, metric_functional, sk_fn, average):
#     preds = _input_miss_class.preds
#     target = _input_miss_class.target
#     preds_flat = torch.cat(list(preds), dim=0)
#     target_flat = torch.cat(list(target), dim=0)

#     mc = metric_class(num_classes=NUM_CLASSES, average=average)
#     for i in range(NUM_BATCHES):
#         mc.update(preds[i], target[i])
#     class_res = mc.compute()
#     func_res = metric_functional(preds_flat, target_flat, num_classes=NUM_CLASSES, average=average)
#     sk_res = sk_fn(target_flat, preds_flat, average=average, zero_division=1)

#     assert torch.allclose(class_res, torch.tensor(sk_res).float())
#     assert torch.allclose(func_res, torch.tensor(sk_res).float())


# @pytest.mark.parametrize("metric_cls", [Precision, Recall])
# def test_noneavg(metric_cls, noneavg=_negmetric_noneavg):
#     prec = MetricWrapper(metric_cls(average="none", num_classes=noneavg["pred1"].shape[1]))
#     result1 = prec(noneavg["pred1"], noneavg["target1"])
#     assert torch.allclose(noneavg["res1"], result1, equal_nan=True)
#     result2 = prec(noneavg["pred2"], noneavg["target2"])
#     assert torch.allclose(noneavg["res2"], result2, equal_nan=True)
