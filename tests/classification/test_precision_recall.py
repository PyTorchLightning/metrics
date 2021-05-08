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
from sklearn.metrics import precision_score, recall_score
from torch import Tensor, tensor

from tests.classification.inputs import _input_binary, _input_binary_logits, _input_binary_prob
from tests.classification.inputs import _input_multiclass as _input_mcls
from tests.classification.inputs import _input_multiclass_logits as _input_mcls_logits
from tests.classification.inputs import _input_multiclass_prob as _input_mcls_prob
from tests.classification.inputs import _input_multidim_multiclass as _input_mdmc
from tests.classification.inputs import _input_multidim_multiclass_prob as _input_mdmc_prob
from tests.classification.inputs import _input_multilabel as _input_mlb
from tests.classification.inputs import _input_multilabel_logits as _input_mlb_logits
from tests.classification.inputs import _input_multilabel_prob as _input_mlb_prob
from tests.helpers import seed_all
from tests.helpers.testers import NUM_CLASSES, THRESHOLD, MetricTester
from torchmetrics import Metric, Precision, Recall
from torchmetrics.functional import precision, precision_recall, recall
from torchmetrics.utilities.checks import _input_format_classification

seed_all(42)


def _sk_prec_recall(preds, target, sk_fn, num_classes, average, multiclass, ignore_index, mdmc_average=None):
    # todo: `mdmc_average` is unused
    if average == "none":
        average = None
    if num_classes == 1:
        average = "binary"

    labels = list(range(num_classes))
    try:
        labels.remove(ignore_index)
    except ValueError:
        pass

    sk_preds, sk_target, _ = _input_format_classification(
        preds, target, THRESHOLD, num_classes=num_classes, multiclass=multiclass
    )
    sk_preds, sk_target = sk_preds.numpy(), sk_target.numpy()

    sk_scores = sk_fn(sk_target, sk_preds, average=average, zero_division=0, labels=labels)

    if len(labels) != num_classes and not average:
        sk_scores = np.insert(sk_scores, ignore_index, np.nan)

    return sk_scores


def _sk_prec_recall_multidim_multiclass(
    preds, target, sk_fn, num_classes, average, multiclass, ignore_index, mdmc_average
):
    preds, target, _ = _input_format_classification(
        preds, target, threshold=THRESHOLD, num_classes=num_classes, multiclass=multiclass
    )

    if mdmc_average == "global":
        preds = torch.transpose(preds, 1, 2).reshape(-1, preds.shape[1])
        target = torch.transpose(target, 1, 2).reshape(-1, target.shape[1])

        return _sk_prec_recall(preds, target, sk_fn, num_classes, average, False, ignore_index)
    elif mdmc_average == "samplewise":
        scores = []

        for i in range(preds.shape[0]):
            pred_i = preds[i, ...].T
            target_i = target[i, ...].T
            scores_i = _sk_prec_recall(pred_i, target_i, sk_fn, num_classes, average, False, ignore_index)

            scores.append(np.expand_dims(scores_i, 0))

        return np.concatenate(scores).mean(axis=0)


@pytest.mark.parametrize("metric, fn_metric", [(Precision, precision), (Recall, recall)])
@pytest.mark.parametrize(
    "average, mdmc_average, num_classes, ignore_index, match_str",
    [
        ("wrong", None, None, None, "`average`"),
        ("micro", "wrong", None, None, "`mdmc"),
        ("macro", None, None, None, "number of classes"),
        ("macro", None, 1, 0, "ignore_index"),
    ],
)
def test_wrong_params(metric, fn_metric, average, mdmc_average, num_classes, ignore_index, match_str):
    with pytest.raises(ValueError, match=match_str):
        metric(
            average=average,
            mdmc_average=mdmc_average,
            num_classes=num_classes,
            ignore_index=ignore_index,
        )

    with pytest.raises(ValueError, match=match_str):
        fn_metric(
            _input_binary.preds[0],
            _input_binary.target[0],
            average=average,
            mdmc_average=mdmc_average,
            num_classes=num_classes,
            ignore_index=ignore_index,
        )

    with pytest.raises(ValueError, match=match_str):
        precision_recall(
            _input_binary.preds[0],
            _input_binary.target[0],
            average=average,
            mdmc_average=mdmc_average,
            num_classes=num_classes,
            ignore_index=ignore_index,
        )


@pytest.mark.parametrize("metric_class, metric_fn", [(Recall, recall), (Precision, precision)])
def test_zero_division(metric_class, metric_fn):
    """ Test that zero_division works correctly (currently should just set to 0). """

    preds = tensor([1, 2, 1, 1])
    target = tensor([2, 1, 2, 1])

    cl_metric = metric_class(average="none", num_classes=3)
    cl_metric(preds, target)

    result_cl = cl_metric.compute()
    result_fn = metric_fn(preds, target, average="none", num_classes=3)

    assert result_cl[0] == result_fn[0] == 0


@pytest.mark.parametrize("metric_class, metric_fn", [(Recall, recall), (Precision, precision)])
def test_no_support(metric_class, metric_fn):
    """This tests a rare edge case, where there is only one class present
    in target, and ignore_index is set to exactly that class - and the
    average method is equal to 'weighted'.

    This would mean that the sum of weights equals zero, and would, without
    taking care of this case, return NaN. However, the reduction function
    should catch that and set the metric to equal the value of zero_division
    in this case (zero_division is for now not configurable and equals 0).
    """

    preds = tensor([1, 1, 0, 0])
    target = tensor([0, 0, 0, 0])

    cl_metric = metric_class(average="weighted", num_classes=2, ignore_index=0)
    cl_metric(preds, target)

    result_cl = cl_metric.compute()
    result_fn = metric_fn(preds, target, average="weighted", num_classes=2, ignore_index=0)

    assert result_cl == result_fn == 0


@pytest.mark.parametrize(
    "metric_class, metric_fn, sk_fn", [(Recall, recall, recall_score), (Precision, precision, precision_score)]
)
@pytest.mark.parametrize("average", ["micro", "macro", None, "weighted", "samples"])
@pytest.mark.parametrize("ignore_index", [None, 0])
@pytest.mark.parametrize(
    "preds, target, num_classes, multiclass, mdmc_average, sk_wrapper",
    [
        (_input_binary_logits.preds, _input_binary_logits.target, 1, None, None, _sk_prec_recall),
        (_input_binary_prob.preds, _input_binary_prob.target, 1, None, None, _sk_prec_recall),
        (_input_binary.preds, _input_binary.target, 1, False, None, _sk_prec_recall),
        (_input_mlb_logits.preds, _input_mlb_logits.target, NUM_CLASSES, None, None, _sk_prec_recall),
        (_input_mlb_prob.preds, _input_mlb_prob.target, NUM_CLASSES, None, None, _sk_prec_recall),
        (_input_mlb.preds, _input_mlb.target, NUM_CLASSES, False, None, _sk_prec_recall),
        (_input_mcls_logits.preds, _input_mcls_logits.target, NUM_CLASSES, None, None, _sk_prec_recall),
        (_input_mcls_prob.preds, _input_mcls_prob.target, NUM_CLASSES, None, None, _sk_prec_recall),
        (_input_mcls.preds, _input_mcls.target, NUM_CLASSES, None, None, _sk_prec_recall),
        (_input_mdmc.preds, _input_mdmc.target, NUM_CLASSES, None, "global", _sk_prec_recall_multidim_multiclass),
        (
            _input_mdmc_prob.preds, _input_mdmc_prob.target, NUM_CLASSES, None, "global",
            _sk_prec_recall_multidim_multiclass
        ),
        (_input_mdmc.preds, _input_mdmc.target, NUM_CLASSES, None, "samplewise", _sk_prec_recall_multidim_multiclass),
        (
            _input_mdmc_prob.preds, _input_mdmc_prob.target, NUM_CLASSES, None, "samplewise",
            _sk_prec_recall_multidim_multiclass
        ),
    ],
)
class TestPrecisionRecall(MetricTester):

    @pytest.mark.parametrize("ddp", [False, True])
    @pytest.mark.parametrize("dist_sync_on_step", [True, False])
    def test_precision_recall_class(
        self,
        ddp: bool,
        dist_sync_on_step: bool,
        preds: Tensor,
        target: Tensor,
        sk_wrapper: Callable,
        metric_class: Metric,
        metric_fn: Callable,
        sk_fn: Callable,
        multiclass: Optional[bool],
        num_classes: Optional[int],
        average: str,
        mdmc_average: Optional[str],
        ignore_index: Optional[int],
    ):
        # todo: `metric_fn` is unused
        if num_classes == 1 and average != "micro":
            pytest.skip("Only test binary data for 'micro' avg (equivalent of 'binary' in sklearn)")

        if ignore_index is not None and preds.ndim == 2:
            pytest.skip("Skipping ignore_index test with binary inputs.")

        if average == "weighted" and ignore_index is not None and mdmc_average is not None:
            pytest.skip("Ignore special case where we are ignoring entire sample for 'weighted' average")

        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=metric_class,
            sk_metric=partial(
                sk_wrapper,
                sk_fn=sk_fn,
                average=average,
                num_classes=num_classes,
                multiclass=multiclass,
                ignore_index=ignore_index,
                mdmc_average=mdmc_average,
            ),
            dist_sync_on_step=dist_sync_on_step,
            metric_args={
                "num_classes": num_classes,
                "average": average,
                "threshold": THRESHOLD,
                "multiclass": multiclass,
                "ignore_index": ignore_index,
                "mdmc_average": mdmc_average,
            },
            check_dist_sync_on_step=True,
            check_batch=True,
        )

    def test_precision_recall_fn(
        self,
        preds: Tensor,
        target: Tensor,
        sk_wrapper: Callable,
        metric_class: Metric,
        metric_fn: Callable,
        sk_fn: Callable,
        multiclass: Optional[bool],
        num_classes: Optional[int],
        average: str,
        mdmc_average: Optional[str],
        ignore_index: Optional[int],
    ):
        # todo: `metric_class` is unused
        if num_classes == 1 and average != "micro":
            pytest.skip("Only test binary data for 'micro' avg (equivalent of 'binary' in sklearn)")

        if ignore_index is not None and preds.ndim == 2:
            pytest.skip("Skipping ignore_index test with binary inputs.")

        if average == "weighted" and ignore_index is not None and mdmc_average is not None:
            pytest.skip("Ignore special case where we are ignoring entire sample for 'weighted' average")

        self.run_functional_metric_test(
            preds,
            target,
            metric_functional=metric_fn,
            sk_metric=partial(
                sk_wrapper,
                sk_fn=sk_fn,
                average=average,
                num_classes=num_classes,
                multiclass=multiclass,
                ignore_index=ignore_index,
                mdmc_average=mdmc_average,
            ),
            metric_args={
                "num_classes": num_classes,
                "average": average,
                "threshold": THRESHOLD,
                "multiclass": multiclass,
                "ignore_index": ignore_index,
                "mdmc_average": mdmc_average,
            },
        )


@pytest.mark.parametrize("average", ["micro", "macro", None, "weighted", "samples"])
def test_precision_recall_joint(average):
    """A simple test of the joint precision_recall metric.

    No need to test this thorougly, as it is just a combination of precision and recall,
    which are already tested thoroughly.
    """

    precision_result = precision(
        _input_mcls_prob.preds[0], _input_mcls_prob.target[0], average=average, num_classes=NUM_CLASSES
    )
    recall_result = recall(
        _input_mcls_prob.preds[0], _input_mcls_prob.target[0], average=average, num_classes=NUM_CLASSES
    )

    prec_recall_result = precision_recall(
        _input_mcls_prob.preds[0], _input_mcls_prob.target[0], average=average, num_classes=NUM_CLASSES
    )

    assert torch.equal(precision_result, prec_recall_result[0])
    assert torch.equal(recall_result, prec_recall_result[1])


_mc_k_target = tensor([0, 1, 2])
_mc_k_preds = tensor([[0.35, 0.4, 0.25], [0.1, 0.5, 0.4], [0.2, 0.1, 0.7]])
_ml_k_target = tensor([[0, 1, 0], [1, 1, 0], [0, 0, 0]])
_ml_k_preds = tensor([[0.9, 0.2, 0.75], [0.1, 0.7, 0.8], [0.6, 0.1, 0.7]])


@pytest.mark.parametrize("metric_class, metric_fn", [(Recall, recall), (Precision, precision)])
@pytest.mark.parametrize(
    "k, preds, target, average, expected_prec, expected_recall",
    [
        (1, _mc_k_preds, _mc_k_target, "micro", tensor(2 / 3), tensor(2 / 3)),
        (2, _mc_k_preds, _mc_k_target, "micro", tensor(1 / 2), tensor(1.0)),
        (1, _ml_k_preds, _ml_k_target, "micro", tensor(0.0), tensor(0.0)),
        (2, _ml_k_preds, _ml_k_target, "micro", tensor(1 / 6), tensor(1 / 3)),
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
    """A simple test to check that top_k works as expected.

    Just a sanity check, the tests in StatScores should already guarantee the correctness of results.
    """

    class_metric = metric_class(top_k=k, average=average, num_classes=3)
    class_metric.update(preds, target)

    if metric_class.__name__ == "Precision":
        result = expected_prec
    else:
        result = expected_recall

    assert torch.equal(class_metric.compute(), result)
    assert torch.equal(metric_fn(preds, target, top_k=k, average=average, num_classes=3), result)
