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

import numpy as np
import pytest
from sklearn.metrics import accuracy_score as sk_accuracy
from torch import tensor

from tests.classification.inputs import _input_binary, _input_binary_logits, _input_binary_prob
from tests.classification.inputs import _input_multiclass as _input_mcls
from tests.classification.inputs import _input_multiclass_logits as _input_mcls_logits
from tests.classification.inputs import _input_multiclass_prob as _input_mcls_prob
from tests.classification.inputs import _input_multidim_multiclass as _input_mdmc
from tests.classification.inputs import _input_multidim_multiclass_prob as _input_mdmc_prob
from tests.classification.inputs import _input_multilabel as _input_mlb
from tests.classification.inputs import _input_multilabel_logits as _input_mlb_logits
from tests.classification.inputs import _input_multilabel_multidim as _input_mlmd
from tests.classification.inputs import _input_multilabel_multidim_prob as _input_mlmd_prob
from tests.classification.inputs import _input_multilabel_prob as _input_mlb_prob
from tests.helpers import seed_all
from tests.helpers.testers import NUM_CLASSES, THRESHOLD, MetricTester
from torchmetrics import Accuracy
from torchmetrics.functional import accuracy
from torchmetrics.utilities.checks import _input_format_classification
from torchmetrics.utilities.enums import DataType

seed_all(42)


def _sk_accuracy(preds, target, subset_accuracy):
    sk_preds, sk_target, mode = _input_format_classification(preds, target, threshold=THRESHOLD)
    sk_preds, sk_target = sk_preds.numpy(), sk_target.numpy()

    if mode == DataType.MULTIDIM_MULTICLASS and not subset_accuracy:
        sk_preds, sk_target = np.transpose(sk_preds, (0, 2, 1)), np.transpose(sk_target, (0, 2, 1))
        sk_preds, sk_target = sk_preds.reshape(-1, sk_preds.shape[2]), sk_target.reshape(-1, sk_target.shape[2])
    elif mode == DataType.MULTIDIM_MULTICLASS and subset_accuracy:
        return np.all(sk_preds == sk_target, axis=(1, 2)).mean()
    elif mode == DataType.MULTILABEL and not subset_accuracy:
        sk_preds, sk_target = sk_preds.reshape(-1), sk_target.reshape(-1)

    return sk_accuracy(y_true=sk_target, y_pred=sk_preds)


@pytest.mark.parametrize(
    "preds, target, subset_accuracy",
    [
        (_input_binary_logits.preds, _input_binary_logits.target, False),
        (_input_binary_prob.preds, _input_binary_prob.target, False),
        (_input_binary.preds, _input_binary.target, False),
        (_input_mlb_prob.preds, _input_mlb_prob.target, True),
        (_input_mlb_logits.preds, _input_mlb_logits.target, False),
        (_input_mlb_prob.preds, _input_mlb_prob.target, False),
        (_input_mlb.preds, _input_mlb.target, True),
        (_input_mlb.preds, _input_mlb.target, False),
        (_input_mcls_prob.preds, _input_mcls_prob.target, False),
        (_input_mcls_logits.preds, _input_mcls_logits.target, False),
        (_input_mcls.preds, _input_mcls.target, False),
        (_input_mdmc_prob.preds, _input_mdmc_prob.target, False),
        (_input_mdmc_prob.preds, _input_mdmc_prob.target, True),
        (_input_mdmc.preds, _input_mdmc.target, False),
        (_input_mdmc.preds, _input_mdmc.target, True),
        (_input_mlmd_prob.preds, _input_mlmd_prob.target, True),
        (_input_mlmd_prob.preds, _input_mlmd_prob.target, False),
        (_input_mlmd.preds, _input_mlmd.target, True),
        (_input_mlmd.preds, _input_mlmd.target, False),
    ],
)
class TestAccuracies(MetricTester):

    @pytest.mark.parametrize("ddp", [False, True])
    @pytest.mark.parametrize("dist_sync_on_step", [False, True])
    def test_accuracy_class(self, ddp, dist_sync_on_step, preds, target, subset_accuracy):
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=Accuracy,
            sk_metric=partial(_sk_accuracy, subset_accuracy=subset_accuracy),
            dist_sync_on_step=dist_sync_on_step,
            metric_args={
                "threshold": THRESHOLD,
                "subset_accuracy": subset_accuracy
            },
        )

    def test_accuracy_fn(self, preds, target, subset_accuracy):
        self.run_functional_metric_test(
            preds,
            target,
            metric_functional=accuracy,
            sk_metric=partial(_sk_accuracy, subset_accuracy=subset_accuracy),
            metric_args={
                "threshold": THRESHOLD,
                "subset_accuracy": subset_accuracy
            },
        )

    def test_accuracy_differentiability(self, preds, target, subset_accuracy):
        self.run_differentiability_test(
            preds=preds,
            target=target,
            metric_module=Accuracy,
            metric_functional=accuracy,
            metric_args={
                "threshold": THRESHOLD,
                "subset_accuracy": subset_accuracy
            }
        )


_l1to4 = [0.1, 0.2, 0.3, 0.4]
_l1to4t3 = np.array([_l1to4, _l1to4, _l1to4])
_l1to4t3_mcls = [_l1to4t3.T, _l1to4t3.T, _l1to4t3.T]

# The preds in these examples always put highest probability on class 3, second highest on class 2,
# third highest on class 1, and lowest on class 0
_topk_preds_mcls = tensor([_l1to4t3, _l1to4t3]).float()
_topk_target_mcls = tensor([[1, 2, 3], [2, 1, 0]])

# This is like for MC case, but one sample in each batch is sabotaged with 0 class prediction :)
_topk_preds_mdmc = tensor([_l1to4t3_mcls, _l1to4t3_mcls]).float()
_topk_target_mdmc = tensor([[[1, 1, 0], [2, 2, 2], [3, 3, 3]], [[2, 2, 0], [1, 1, 1], [0, 0, 0]]])

# Multilabel
_ml_t1 = [.8, .2, .8, .2]
_ml_t2 = [_ml_t1, _ml_t1]
_ml_ta2 = [[1, 0, 1, 1], [0, 1, 1, 0]]
_av_preds_ml = tensor([_ml_t2, _ml_t2]).float()
_av_target_ml = tensor([_ml_ta2, _ml_ta2])


# Replace with a proper sk_metric test once sklearn 0.24 hits :)
@pytest.mark.parametrize(
    "preds, target, exp_result, k, subset_accuracy",
    [
        (_topk_preds_mcls, _topk_target_mcls, 1 / 6, 1, False),
        (_topk_preds_mcls, _topk_target_mcls, 3 / 6, 2, False),
        (_topk_preds_mcls, _topk_target_mcls, 5 / 6, 3, False),
        (_topk_preds_mcls, _topk_target_mcls, 1 / 6, 1, True),
        (_topk_preds_mcls, _topk_target_mcls, 3 / 6, 2, True),
        (_topk_preds_mcls, _topk_target_mcls, 5 / 6, 3, True),
        (_topk_preds_mdmc, _topk_target_mdmc, 1 / 6, 1, False),
        (_topk_preds_mdmc, _topk_target_mdmc, 8 / 18, 2, False),
        (_topk_preds_mdmc, _topk_target_mdmc, 13 / 18, 3, False),
        (_topk_preds_mdmc, _topk_target_mdmc, 1 / 6, 1, True),
        (_topk_preds_mdmc, _topk_target_mdmc, 2 / 6, 2, True),
        (_topk_preds_mdmc, _topk_target_mdmc, 3 / 6, 3, True),
        (_av_preds_ml, _av_target_ml, 5 / 8, None, False),
        (_av_preds_ml, _av_target_ml, 0, None, True),
    ],
)
def test_topk_accuracy(preds, target, exp_result, k, subset_accuracy):
    topk = Accuracy(top_k=k, subset_accuracy=subset_accuracy)

    for batch in range(preds.shape[0]):
        topk(preds[batch], target[batch])

    assert topk.compute() == exp_result

    # Test functional
    total_samples = target.shape[0] * target.shape[1]

    preds = preds.view(total_samples, 4, -1)
    target = target.view(total_samples, -1)

    assert accuracy(preds, target, top_k=k, subset_accuracy=subset_accuracy) == exp_result


# Only MC and MDMC with probs input type should be accepted for top_k
@pytest.mark.parametrize(
    "preds, target",
    [
        (_input_binary_prob.preds, _input_binary_prob.target),
        (_input_binary.preds, _input_binary.target),
        (_input_mlb_prob.preds, _input_mlb_prob.target),
        (_input_mlb.preds, _input_mlb.target),
        (_input_mcls.preds, _input_mcls.target),
        (_input_mdmc.preds, _input_mdmc.target),
        (_input_mlmd_prob.preds, _input_mlmd_prob.target),
        (_input_mlmd.preds, _input_mlmd.target),
    ],
)
def test_topk_accuracy_wrong_input_types(preds, target):
    topk = Accuracy(top_k=1)

    with pytest.raises(ValueError):
        topk(preds[0], target[0])

    with pytest.raises(ValueError):
        accuracy(preds[0], target[0], top_k=1)


@pytest.mark.parametrize(
    "average, mdmc_average, num_classes, inputs, ignore_index, top_k, threshold",
    [
        ("unknown", None, None, _input_binary, None, None, 0.5),
        ("micro", "unknown", None, _input_binary, None, None, 0.5),
        ("macro", None, None, _input_binary, None, None, 0.5),
        ("micro", None, None, _input_mdmc_prob, None, None, 0.5),
        ("micro", None, None, _input_binary_prob, 0, None, 0.5),
        ("micro", None, None, _input_mcls_prob, NUM_CLASSES, None, 0.5),
        ("micro", None, NUM_CLASSES, _input_mcls_prob, NUM_CLASSES, None, 0.5),
        (None, None, None, _input_mcls_prob, None, 0, 0.5),
        (None, None, None, _input_mcls_prob, None, None, 1.5),
    ],
)
def test_wrong_params(average, mdmc_average, num_classes, inputs, ignore_index, top_k, threshold):
    preds, target = inputs.preds, inputs.target

    with pytest.raises(ValueError):
        acc = Accuracy(
            average=average,
            mdmc_average=mdmc_average,
            num_classes=num_classes,
            ignore_index=ignore_index,
            threshold=threshold,
            top_k=top_k
        )
        acc(preds[0], target[0])
        acc.compute()

    with pytest.raises(ValueError):
        accuracy(
            preds[0],
            target[0],
            average=average,
            mdmc_average=mdmc_average,
            num_classes=num_classes,
            ignore_index=ignore_index,
            threshold=threshold,
            top_k=top_k
        )


@pytest.mark.parametrize(
    "preds_mc, target_mc, preds_ml, target_ml",
    [(
        tensor([0, 1, 1, 1]),
        tensor([2, 2, 1, 1]),
        tensor([[0.8, 0.2, 0.8, 0.7], [0.6, 0.4, 0.6, 0.5]]),
        tensor([[1, 0, 1, 1], [0, 0, 1, 0]]),
    )],
)
def test_different_modes(preds_mc, target_mc, preds_ml, target_ml):
    acc = Accuracy()
    acc(preds_mc, target_mc)
    with pytest.raises(ValueError, match="^[You cannot use]"):
        acc(preds_ml, target_ml)


_bin_t1 = [0.7, 0.6, 0.2, 0.1]
_av_preds_bin = tensor([_bin_t1, _bin_t1]).float()
_av_target_bin = tensor([[1, 0, 0, 0], [0, 1, 1, 0]])


@pytest.mark.parametrize(
    "preds, target, num_classes, exp_result, average, mdmc_average",
    [
        (_topk_preds_mcls, _topk_target_mcls, 4, 1 / 4, "macro", None),
        (_topk_preds_mcls, _topk_target_mcls, 4, 1 / 6, "weighted", None),
        (_topk_preds_mcls, _topk_target_mcls, 4, [0., 0., 0., 1.], "none", None),
        (_topk_preds_mcls, _topk_target_mcls, 4, 1 / 6, "samples", None),
        (_topk_preds_mdmc, _topk_target_mdmc, 4, 1 / 24, "macro", "samplewise"),
        (_topk_preds_mdmc, _topk_target_mdmc, 4, 1 / 6, "weighted", "samplewise"),
        (_topk_preds_mdmc, _topk_target_mdmc, 4, [0., 0., 0., 1 / 6], "none", "samplewise"),
        (_topk_preds_mdmc, _topk_target_mdmc, 4, 1 / 6, "samples", "samplewise"),
        (_topk_preds_mdmc, _topk_target_mdmc, 4, 1 / 6, "samples", "global"),
        (_av_preds_ml, _av_target_ml, 4, 5 / 8, "macro", None),
        (_av_preds_ml, _av_target_ml, 4, 0.70000005, "weighted", None),
        (_av_preds_ml, _av_target_ml, 4, [1 / 2, 1 / 2, 1., 1 / 2], "none", None),
        (_av_preds_ml, _av_target_ml, 4, 5 / 8, "samples", None),
    ],
)
def test_average_accuracy(preds, target, num_classes, exp_result, average, mdmc_average):
    acc = Accuracy(num_classes=num_classes, average=average, mdmc_average=mdmc_average)

    for batch in range(preds.shape[0]):
        acc(preds[batch], target[batch])

    assert (acc.compute() == tensor(exp_result)).all()

    # Test functional
    total_samples = target.shape[0] * target.shape[1]

    preds = preds.view(total_samples, num_classes, -1)
    target = target.view(total_samples, -1)

    acc_score = accuracy(preds, target, num_classes=num_classes, average=average, mdmc_average=mdmc_average)
    assert (acc_score == tensor(exp_result)).all()


@pytest.mark.parametrize(
    "preds, target, num_classes, exp_result, average, multiclass",
    [
        (_av_preds_bin, _av_target_bin, 2, 19 / 30, "macro", True),
        (_av_preds_bin, _av_target_bin, 2, 5 / 8, "weighted", True),
        (_av_preds_bin, _av_target_bin, 2, [3 / 5, 2 / 3], "none", True),
        (_av_preds_bin, _av_target_bin, 2, 5 / 8, "samples", True),
    ],
)
def test_average_accuracy_bin(preds, target, num_classes, exp_result, average, multiclass):
    acc = Accuracy(num_classes=num_classes, average=average, multiclass=multiclass)

    for batch in range(preds.shape[0]):
        acc(preds[batch], target[batch])

    assert (acc.compute() == tensor(exp_result)).all()

    # Test functional
    total_samples = target.shape[0] * target.shape[1]

    preds = preds.view(total_samples, -1)
    target = target.view(total_samples, -1)
    acc_score = accuracy(preds, target, num_classes=num_classes, average=average, multiclass=multiclass)
    assert (acc_score == tensor(exp_result)).all()
