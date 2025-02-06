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
import pickle
from copy import deepcopy
from typing import Any

import pytest
import torch

from torchmetrics import ClasswiseWrapper, Metric, MetricCollection
from torchmetrics.classification import (
    BinaryAccuracy,
    MulticlassAccuracy,
    MulticlassAUROC,
    MulticlassAveragePrecision,
    MulticlassCohenKappa,
    MulticlassConfusionMatrix,
    MulticlassF1Score,
    MulticlassMatthewsCorrCoef,
    MulticlassPrecision,
    MulticlassRecall,
    MultilabelAUROC,
    MultilabelAveragePrecision,
)
from torchmetrics.utilities.checks import _allclose_recursive
from unittests._helpers import seed_all
from unittests._helpers.testers import DummyMetricDiff, DummyMetricMultiOutputDict, DummyMetricSum

seed_all(42)


def test_metric_collection_jit_script():
    """Test that the MetricCollection can be scripted and jitted."""
    m1 = DummyMetricSum()
    m2 = DummyMetricDiff()
    metric_collection = MetricCollection([m1, m2])
    scripted = torch.jit.script(metric_collection)
    assert isinstance(scripted, torch.jit.ScriptModule)


def test_metric_collection(tmpdir):
    """Test that updating the metric collection is equal to individually updating metrics in the collection."""
    m1 = DummyMetricSum()
    m2 = DummyMetricDiff()

    metric_collection = MetricCollection([m1, m2])

    # Test correct dict structure
    assert len(metric_collection) == 2
    assert metric_collection["DummyMetricSum"] == m1
    assert metric_collection["DummyMetricDiff"] == m2

    # Test correct initialization
    for name, metric in metric_collection.items():
        assert metric.x == 0, f"Metric {name} not initialized correctly"

    # Test every metric gets updated
    metric_collection.update(5)
    for name, metric in metric_collection.items():
        assert metric.x.abs() == 5, f"Metric {name} not updated correctly"

    # Test compute on each metric
    metric_collection.update(-5)
    metric_vals = metric_collection.compute()
    assert len(metric_vals) == 2
    for name, metric_val in metric_vals.items():
        assert metric_val == 0, f"Metric {name}.compute not called correctly"

    # Test that everything is reset
    for name, metric in metric_collection.items():
        assert metric.x == 0, f"Metric {name} not reset correctly"

    # Test pickable
    metric_pickled = pickle.dumps(metric_collection)
    metric_loaded = pickle.loads(metric_pickled)
    assert isinstance(metric_loaded, MetricCollection)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires GPU.")
def test_device_and_dtype_transfer_metriccollection(tmpdir):
    """Test that metrics in the collection correctly gets updated their dtype and device."""
    m1 = DummyMetricSum()
    m2 = DummyMetricDiff()

    metric_collection = MetricCollection([m1, m2])
    for metric in metric_collection.values():
        assert metric.x.is_cuda is False
        assert metric.x.dtype == torch.float32

    metric_collection = metric_collection.to(device="cuda")
    for metric in metric_collection.values():
        assert metric.x.is_cuda

    metric_collection = metric_collection.set_dtype(torch.double)
    for metric in metric_collection.values():
        assert metric.x.dtype == torch.float64

    metric_collection = metric_collection.set_dtype(torch.half)
    for metric in metric_collection.values():
        assert metric.x.dtype == torch.float16


def test_metric_collection_wrong_input(tmpdir):
    """Check that errors are raised on wrong input."""
    dms = DummyMetricSum()

    # Not all input are metrics (list)
    with pytest.raises(ValueError, match="Input .* to `MetricCollection` is not a instance of .*"):
        _ = MetricCollection([dms, 5])

    # Not all input are metrics (dict)
    with pytest.raises(ValueError, match="Value .* belonging to key .* is not an instance of .*"):
        _ = MetricCollection({"metric1": dms, "metric2": 5})

    # Same metric passed in multiple times
    with pytest.raises(ValueError, match="Encountered two metrics both named *."):
        _ = MetricCollection([dms, dms])

    # Not a list or dict passed in
    with pytest.warns(Warning, match=" which are not `Metric` so they will be ignored."):
        _ = MetricCollection(dms, [dms])


def test_metric_collection_args_kwargs(tmpdir):
    """Check that args and kwargs gets passed correctly in metric collection, checks both update and forward."""
    m1 = DummyMetricSum()
    m2 = DummyMetricDiff()

    metric_collection = MetricCollection([m1, m2])

    # args gets passed to all metrics
    metric_collection.update(5)
    assert metric_collection["DummyMetricSum"].x == 5
    assert metric_collection["DummyMetricDiff"].x == -5
    metric_collection.reset()
    _ = metric_collection(5)
    assert metric_collection["DummyMetricSum"].x == 5
    assert metric_collection["DummyMetricDiff"].x == -5
    metric_collection.reset()

    # kwargs gets only passed to metrics that it matches
    metric_collection.update(x=10, y=20)
    assert metric_collection["DummyMetricSum"].x == 10
    assert metric_collection["DummyMetricDiff"].x == -20
    metric_collection.reset()
    _ = metric_collection(x=10, y=20)
    assert metric_collection["DummyMetricSum"].x == 10
    assert metric_collection["DummyMetricDiff"].x == -20


@pytest.mark.parametrize(
    ("prefix", "postfix"),
    [
        (None, None),
        ("prefix_", None),
        (None, "_postfix"),
        ("prefix_", "_postfix"),
    ],
)
def test_metric_collection_prefix_postfix_args(prefix, postfix):
    """Test that the prefix arg alters the keywords in the output."""
    m1 = DummyMetricSum()
    m2 = DummyMetricDiff()
    names = ["DummyMetricSum", "DummyMetricDiff"]
    names = [prefix + n if prefix is not None else n for n in names]
    names = [n + postfix if postfix is not None else n for n in names]

    metric_collection = MetricCollection([m1, m2], prefix=prefix, postfix=postfix)

    # test forward
    out = metric_collection(5)
    for name in names:
        assert name in out, "prefix or postfix argument not working as intended with forward method"

    # test compute
    out = metric_collection.compute()
    for name in names:
        assert name in out, "prefix or postfix argument not working as intended with compute method"

    # test clone
    new_metric_collection = metric_collection.clone(prefix="new_prefix_")
    out = new_metric_collection(5)
    names = [n[len(prefix) :] if prefix is not None else n for n in names]  # strip away old prefix
    for name in names:
        assert f"new_prefix_{name}" in out, "prefix argument not working as intended with clone method"

    for k in new_metric_collection:
        assert "new_prefix_" in k

    for k in new_metric_collection.keys(keep_base=False):
        assert "new_prefix_" in k

    for k in new_metric_collection.keys(keep_base=True):
        assert "new_prefix_" not in k

    assert isinstance(new_metric_collection.keys(keep_base=True), type(new_metric_collection.keys(keep_base=False)))
    assert isinstance(new_metric_collection.items(keep_base=True), type(new_metric_collection.items(keep_base=False)))

    new_metric_collection = new_metric_collection.clone(postfix="_new_postfix")
    out = new_metric_collection(5)
    names = [n[: -len(postfix)] if postfix is not None else n for n in names]  # strip away old postfix
    for name in names:
        assert f"new_prefix_{name}_new_postfix" in out, "postfix argument not working as intended with clone method"

    keys = list(new_metric_collection.keys())
    for k in keys:
        assert new_metric_collection[k]  # check that the keys are valid even with prefix and postfix


def test_metric_collection_repr():
    """Test MetricCollection."""

    class A(DummyMetricSum):
        pass

    class B(DummyMetricDiff):
        pass

    m1 = A()
    m2 = B()
    metric_collection = MetricCollection([m1, m2], prefix=None, postfix=None)

    expected = "MetricCollection(\n  (A): A()\n  (B): B()\n)"
    assert metric_collection.__repr__() == expected

    metric_collection = MetricCollection([m1, m2], prefix="a", postfix=None)

    expected = "MetricCollection(\n  (A): A()\n  (B): B(),\n  prefix=a\n)"
    assert metric_collection.__repr__() == expected

    metric_collection = MetricCollection([m1, m2], prefix=None, postfix="a")
    expected = "MetricCollection(\n  (A): A()\n  (B): B(),\n  postfix=a\n)"
    assert metric_collection.__repr__() == expected

    metric_collection = MetricCollection([m1, m2], prefix="a", postfix="b")
    expected = "MetricCollection(\n  (A): A()\n  (B): B(),\n  prefix=a,\n  postfix=b\n)"
    assert metric_collection.__repr__() == expected


def test_metric_collection_same_order():
    """Test that metrics are stored internally in the same order, regardless of input order."""
    m1 = DummyMetricSum()
    m2 = DummyMetricDiff()
    col1 = MetricCollection({"a": m1, "b": m2})
    col2 = MetricCollection({"b": m2, "a": m1})
    for k1, k2 in zip(col1.keys(), col2.keys()):
        assert k1 == k2


def test_collection_add_metrics():
    """Test that `add_metrics` function called multiple times works as expected."""
    m1 = DummyMetricSum()
    m2 = DummyMetricDiff()

    collection = MetricCollection([m1])
    collection.add_metrics({"m1_": DummyMetricSum()})
    collection.add_metrics(m2)

    collection.update(5)
    results = collection.compute()
    assert results["DummyMetricSum"] == results["m1_"]
    assert results["m1_"] == 5
    assert results["DummyMetricDiff"] == -5


def test_collection_check_arg():
    """Test that the `_check_arg` method works as expected."""
    assert MetricCollection._check_arg(None, "prefix") is None
    assert MetricCollection._check_arg("sample", "prefix") == "sample"

    with pytest.raises(ValueError, match="Expected input `postfix` to be a string, but got"):
        MetricCollection._check_arg(1, "postfix")


def test_collection_filtering():
    """Test that collections works with the kwargs argument."""

    class DummyMetric(Metric):
        full_state_update = True

        def __init__(self) -> None:
            super().__init__()

        def update(self, *args: Any, kwarg: Any):
            pass

        def compute(self):
            return

    class MyAccuracy(Metric):
        full_state_update = True

        def __init__(self) -> None:
            super().__init__()

        def update(self, preds, target, kwarg2):
            pass

        def compute(self):
            return

    mc = MetricCollection([BinaryAccuracy(), DummyMetric()])
    mc2 = MetricCollection([MyAccuracy(), DummyMetric()])
    mc(torch.tensor([0, 1]), torch.tensor([0, 1]), kwarg="kwarg")
    mc2(torch.tensor([0, 1]), torch.tensor([0, 1]), kwarg="kwarg", kwarg2="kwarg2")


# function for generating
_mc_preds = torch.randn(10, 3, 2).softmax(dim=1)
_mc_target = torch.randint(3, (10, 2))
_ml_preds = torch.rand(10, 3)
_ml_target = torch.randint(2, (10, 3))


@pytest.mark.parametrize(
    "metrics, expected, preds, target",
    [
        # single metric forms its own compute group
        (MulticlassAccuracy(num_classes=3), {0: ["MulticlassAccuracy"]}, _mc_preds, _mc_target),
        # two metrics of same class forms a compute group
        (
            {"acc0": MulticlassAccuracy(num_classes=3), "acc1": MulticlassAccuracy(num_classes=3)},
            {0: ["acc0", "acc1"]},
            _mc_preds,
            _mc_target,
        ),
        # two metrics from registry forms a compute group
        (
            [MulticlassPrecision(num_classes=3), MulticlassRecall(num_classes=3)],
            {0: ["MulticlassPrecision", "MulticlassRecall"]},
            _mc_preds,
            _mc_target,
        ),
        # two metrics from different classes gives two compute groups
        (
            [MulticlassConfusionMatrix(num_classes=3), MulticlassRecall(num_classes=3)],
            {0: ["MulticlassConfusionMatrix"], 1: ["MulticlassRecall"]},
            _mc_preds,
            _mc_target,
        ),
        # multi group multi metric
        (
            [
                MulticlassConfusionMatrix(num_classes=3),
                MulticlassCohenKappa(num_classes=3),
                MulticlassRecall(num_classes=3),
                MulticlassPrecision(num_classes=3),
            ],
            {0: ["MulticlassConfusionMatrix", "MulticlassCohenKappa"], 1: ["MulticlassRecall", "MulticlassPrecision"]},
            _mc_preds,
            _mc_target,
        ),
        # Complex example
        (
            {
                "acc": MulticlassAccuracy(num_classes=3),
                "acc2": MulticlassAccuracy(num_classes=3),
                "acc3": MulticlassAccuracy(num_classes=3, multidim_average="samplewise"),
                "f1": MulticlassF1Score(num_classes=3),
                "recall": MulticlassRecall(num_classes=3),
                "confmat": MulticlassConfusionMatrix(num_classes=3),
            },
            {0: ["acc", "acc2", "f1", "recall"], 1: ["acc3"], 2: ["confmat"]},
            _mc_preds,
            _mc_target,
        ),
        # With list states
        (
            [
                MulticlassAUROC(num_classes=3, average="macro"),
                MulticlassAveragePrecision(num_classes=3, average="macro"),
            ],
            {0: ["MulticlassAUROC", "MulticlassAveragePrecision"]},
            _mc_preds,
            _mc_target,
        ),
        # Nested collections
        (
            [
                MetricCollection(
                    MultilabelAUROC(num_labels=3, average="micro"),
                    MultilabelAveragePrecision(num_labels=3, average="micro"),
                    postfix="_micro",
                ),
                MetricCollection(
                    MultilabelAUROC(num_labels=3, average="macro"),
                    MultilabelAveragePrecision(num_labels=3, average="macro"),
                    postfix="_macro",
                ),
            ],
            {
                0: [
                    "MultilabelAUROC_micro",
                    "MultilabelAveragePrecision_micro",
                    "MultilabelAUROC_macro",
                    "MultilabelAveragePrecision_macro",
                ]
            },
            _ml_preds,
            _ml_target,
        ),
    ],
)
class TestComputeGroups:
    """Test class for testing groups computation."""

    @pytest.mark.parametrize(
        ("prefix", "postfix"),
        [
            (None, None),
            ("prefix_", None),
            (None, "_postfix"),
            ("prefix_", "_postfix"),
        ],
    )
    @pytest.mark.parametrize("with_reset", [True, False])
    def test_check_compute_groups_correctness(self, metrics, expected, preds, target, prefix, postfix, with_reset):
        """Check that compute groups are formed after initialization and that metrics are correctly computed."""
        if isinstance(metrics, MetricCollection):
            prefix, postfix = None, None  # disable for nested collections
        m = MetricCollection(deepcopy(metrics), prefix=prefix, postfix=postfix, compute_groups=True)
        # Construct without for comparison
        m2 = MetricCollection(deepcopy(metrics), prefix=prefix, postfix=postfix, compute_groups=False)

        assert len(m.compute_groups) == len(m)
        assert m2.compute_groups == {}

        for _ in range(2):  # repeat to emulate effect of multiple epochs
            m.update(preds, target)
            m2.update(preds, target)

            for member in m.values():
                assert member.update_called

            assert m.compute_groups == expected
            assert m2.compute_groups == {}

            # compute groups should kick in here
            m.update(preds, target)
            m2.update(preds, target)

            for member in m.values():
                assert member.update_called

            # compare results for correctness
            res_cg = m.compute()
            res_without_cg = m2.compute()
            for key in res_cg:
                assert torch.allclose(res_cg[key], res_without_cg[key])

            # Check if second compute is the same
            res_cg2 = m.compute()
            for key in res_cg2:
                assert torch.allclose(res_cg[key], res_cg2[key])

            if with_reset:
                m.reset()
                m2.reset()

        # Test if a second compute without a reset is the same
        m.reset()
        m.update(preds, target)
        res_cg = m.compute()
        # Simulate different preds by simply inversing them
        m.update(1 - preds, target)
        res_cg2 = m.compute()
        # Now check if the results from the first compute are different from the second
        for key in res_cg:
            # A different shape is okay, therefore skip (this happens for multidim_average="samplewise")
            if res_cg[key].shape != res_cg2[key].shape:
                continue
            assert not torch.all(res_cg[key] == res_cg2[key])

    @pytest.mark.parametrize("method", ["items", "values", "keys"])
    def test_check_compute_groups_items_and_values(self, metrics, expected, preds, target, method):
        """Check states are copied instead of passed by ref when a single metric in the collection is access."""
        m = MetricCollection(deepcopy(metrics), compute_groups=True)
        m2 = MetricCollection(deepcopy(metrics), compute_groups=False)

        for _ in range(2):  # repeat to emulate effect of multiple epochs
            for _ in range(2):  # repeat to emulate effect of multiple batches
                m.update(preds, target)
                m2.update(preds, target)

            def _compare(m1, m2):
                for state in m1._defaults:
                    assert _allclose_recursive(getattr(m1, state), getattr(m2, state))
                # if states are still by reference the reset will make following metrics fail
                m1.reset()
                m2.reset()

            if method == "items":
                for (name_cg, metric_cg), (name_no_cg, metric_no_cg) in zip(m.items(), m2.items()):
                    assert name_cg == name_no_cg
                    _compare(metric_cg, metric_no_cg)
            if method == "values":
                for metric_cg, metric_no_cg in zip(m.values(), m2.values()):
                    _compare(metric_cg, metric_no_cg)
            if method == "keys":
                for key in m:
                    metric_cg, metric_no_cg = m[key], m2[key]
                    _compare(metric_cg, metric_no_cg)


# TODO: test is flaky
# @pytest.mark.parametrize(
#     "metrics",
#     [
#         {"acc0": MulticlassAccuracy(3), "acc1": MulticlassAccuracy(3)},
#         [MulticlassPrecision(3), MulticlassRecall(3)],
#         [MulticlassConfusionMatrix(3), MulticlassCohenKappa(3), MulticlassRecall(3), MulticlassPrecision(3)],
#         {
#             "acc": MulticlassAccuracy(3),
#             "acc2": MulticlassAccuracy(3),
#             "acc3": MulticlassAccuracy(num_classes=3, average="macro"),
#             "f1": MulticlassF1Score(3),
#             "recall": MulticlassRecall(3),
#             "confmat": MulticlassConfusionMatrix(3),
#         },
#     ],
# )
# @pytest.mark.parametrize("steps", [1000])
# def test_check_compute_groups_is_faster(metrics, steps):
#     """Check that compute groups are formed after initialization."""
#     m = MetricCollection(deepcopy(metrics), compute_groups=True)
#     # Construct without for comparison
#     m2 = MetricCollection(deepcopy(metrics), compute_groups=False)

#     preds = torch.randn(10, 3).softmax(dim=-1)
#     target = torch.randint(3, (10,))

#     start = time.time()
#     for _ in range(steps):
#         m.update(preds, target)
#     time_cg = time.time() - start

#     start = time.time()
#     for _ in range(steps):
#         m2.update(preds, target)
#     time_no_cg = time.time() - start

#     assert time_cg < time_no_cg, "using compute groups were not faster"


def test_compute_group_define_by_user():
    """Check that user can provide compute groups."""
    m = MetricCollection(
        MulticlassConfusionMatrix(3),
        MulticlassRecall(3),
        MulticlassPrecision(3),
        compute_groups=[["MulticlassConfusionMatrix"], ["MulticlassRecall", "MulticlassPrecision"]],
    )

    # Check that we are not going to check the groups in the first update
    assert m._groups_checked
    assert m.compute_groups == {0: ["MulticlassConfusionMatrix"], 1: ["MulticlassRecall", "MulticlassPrecision"]}

    preds = torch.randn(10, 3).softmax(dim=-1)
    target = torch.randint(3, (10,))
    m.update(preds, target)
    assert m.compute()


def test_classwise_wrapper_compute_group():
    """Check that user can provide compute groups."""
    classwise_accuracy = ClasswiseWrapper(MulticlassAccuracy(num_classes=3, average=None), prefix="accuracy")
    classwise_recall = ClasswiseWrapper(MulticlassRecall(num_classes=3, average=None), prefix="recall")
    classwise_precision = ClasswiseWrapper(MulticlassPrecision(num_classes=3, average=None), prefix="precision")

    m = MetricCollection(
        {
            "accuracy": ClasswiseWrapper(MulticlassAccuracy(num_classes=3, average=None), prefix="accuracy"),
            "recall": ClasswiseWrapper(MulticlassRecall(num_classes=3, average=None), prefix="recall"),
            "precision": ClasswiseWrapper(MulticlassPrecision(num_classes=3, average=None), prefix="precision"),
        },
        compute_groups=[["accuracy", "recall", "precision"]],
    )

    # Check that we are not going to check the groups in the first update
    assert m._groups_checked
    assert m.compute_groups == {0: ["accuracy", "recall", "precision"]}

    preds = torch.randn(10, 3).softmax(dim=-1)
    target = torch.randint(3, (10,))

    expected = {
        **classwise_accuracy(preds, target),
        **classwise_recall(preds, target),
        **classwise_precision(preds, target),
    }

    m.update(preds, target)
    res = m.compute()

    for key in expected:
        assert torch.allclose(res[key], expected[key])

    # check metric state_dict
    m.state_dict()


def test_compute_on_different_dtype():
    """Check that extraction of compute groups are robust towards difference in dtype."""
    m = MetricCollection([
        MulticlassConfusionMatrix(num_classes=3),
        MulticlassMatthewsCorrCoef(num_classes=3),
    ])
    assert not m._groups_checked
    assert m.compute_groups == {0: ["MulticlassConfusionMatrix"], 1: ["MulticlassMatthewsCorrCoef"]}
    preds = torch.randn(10, 3).softmax(dim=-1)
    target = torch.randint(3, (10,))
    for _ in range(2):
        m.update(preds, target)
    assert m.compute_groups == {0: ["MulticlassConfusionMatrix", "MulticlassMatthewsCorrCoef"]}
    assert m.compute()


def test_error_on_wrong_specified_compute_groups():
    """Test that error is raised if user miss-specify the compute groups."""
    with pytest.raises(ValueError, match="Input MulticlassAccuracy in `compute_groups`.*"):
        MetricCollection(
            MulticlassConfusionMatrix(3),
            MulticlassRecall(3),
            MulticlassPrecision(3),
            compute_groups=[["MulticlassConfusionMatrix"], ["MulticlassRecall", "MulticlassAccuracy"]],
        )


@pytest.mark.parametrize(
    "input_collections",
    [
        [
            MetricCollection(
                [
                    MulticlassAccuracy(num_classes=3, average="macro"),
                    MulticlassPrecision(num_classes=3, average="macro"),
                ],
                prefix="macro_",
            ),
            MetricCollection(
                [
                    MulticlassAccuracy(num_classes=3, average="micro"),
                    MulticlassPrecision(num_classes=3, average="micro"),
                ],
                prefix="micro_",
            ),
        ],
        {
            "macro": MetricCollection([
                MulticlassAccuracy(num_classes=3, average="macro"),
                MulticlassPrecision(num_classes=3, average="macro"),
            ]),
            "micro": MetricCollection([
                MulticlassAccuracy(num_classes=3, average="micro"),
                MulticlassPrecision(num_classes=3, average="micro"),
            ]),
        },
    ],
)
def test_nested_collections(input_collections):
    """Test that nested collections gets flattened to a single collection."""
    metrics = MetricCollection(input_collections, prefix="valmetrics/")
    preds = torch.randn(10, 3).softmax(dim=-1)
    target = torch.randint(3, (10,))
    val = metrics(preds, target)
    assert "valmetrics/macro_MulticlassAccuracy" in val
    assert "valmetrics/macro_MulticlassPrecision" in val
    assert "valmetrics/micro_MulticlassAccuracy" in val
    assert "valmetrics/micro_MulticlassPrecision" in val


@pytest.mark.parametrize(
    ("base_metrics", "expected"),
    [
        (
            DummyMetricMultiOutputDict(),
            (
                "prefix2_prefix1_output1_postfix1_postfix2",
                "prefix2_prefix1_output2_postfix1_postfix2",
            ),
        ),
        (
            {"metric1": DummyMetricMultiOutputDict(), "metric2": DummyMetricMultiOutputDict()},
            (
                "prefix2_prefix1_metric1_output1_postfix1_postfix2",
                "prefix2_prefix1_metric1_output2_postfix1_postfix2",
                "prefix2_prefix1_metric2_output1_postfix1_postfix2",
                "prefix2_prefix1_metric2_output2_postfix1_postfix2",
            ),
        ),
    ],
)
def test_double_nested_collections(base_metrics, expected):
    """Test that double nested collections gets flattened to a single collection."""
    collection1 = MetricCollection(base_metrics, prefix="prefix1_", postfix="_postfix1")
    collection2 = MetricCollection([collection1], prefix="prefix2_", postfix="_postfix2")
    x = torch.randn(10).sum()
    val = collection2(x)

    for key in val:
        assert key in expected


def test_with_custom_prefix_postfix():
    """Test that metric collection does not clash with custom prefix and postfix in users metrics.

    See issue: https://github.com/Lightning-AI/torchmetrics/issues/2065

    """

    class CustomAccuracy(MulticlassAccuracy):
        prefix = "my_prefix"
        postfix = "my_postfix"

        def compute(self):
            value = super().compute()
            return {f"{self.prefix}/accuracy/{self.postfix}": value}

    class CustomPrecision(MulticlassAccuracy):
        prefix = "my_prefix"
        postfix = "my_postfix"

        def compute(self):
            value = super().compute()
            return {f"{self.prefix}/precision/{self.postfix}": value}

    metrics = MetricCollection([CustomAccuracy(num_classes=2), CustomPrecision(num_classes=2)])

    # Update metrics with current batch
    res = metrics(torch.tensor([1, 0, 0, 1]), torch.tensor([1, 0, 0, 0]))

    # Print the calculated metrics
    assert "my_prefix/accuracy/my_postfix" in res
    assert "my_prefix/precision/my_postfix" in res


def test_collection_update():
    """Test that metric collection updates metrics.

    See issue: https://github.com/Lightning-AI/torchmetrics/issues/2916

    """
    from torchmetrics.text import BLEUScore

    scores = MetricCollection({
        "bleu-1": BLEUScore(1),
        "bleu-2": BLEUScore(2),
        "bleu-3": BLEUScore(3),
        "bleu-4": BLEUScore(4),
    })

    preds = ["the cat is on the mat"]
    target = [["there is a cat on the mat", "a cat is on the mat"]]

    scores.update(preds, target)
    actual = scores.compute()

    expected = {
        "bleu-1": torch.tensor(0.8333),
        "bleu-2": torch.tensor(0.8165),
        "bleu-3": torch.tensor(0.7937),
        "bleu-4": torch.tensor(0.7598),
    }

    for k, v in expected.items():
        torch.testing.assert_close(actual=actual.get(k), expected=v, rtol=1e-4, atol=1e-4)
