from typing import Callable, List, Union

import pytest

from torchmetrics.functional.text.cer import char_error_rate
from torchmetrics.text.cer import CharErrorRate
from torchmetrics.utilities.imports import _JIWER_AVAILABLE
from unittests.text.helpers import TextTester
from unittests.text.inputs import _inputs_error_rate_batch_size_1, _inputs_error_rate_batch_size_2

if _JIWER_AVAILABLE:
    from jiwer import cer

else:
    compute_measures = Callable


def _compare_fn(preds: Union[str, List[str]], target: Union[str, List[str]]):
    return cer(target, preds)


@pytest.mark.skipif(not _JIWER_AVAILABLE, reason="test requires jiwer")
@pytest.mark.parametrize(
    ["preds", "targets"],
    [
        (_inputs_error_rate_batch_size_1.preds, _inputs_error_rate_batch_size_1.targets),
        (_inputs_error_rate_batch_size_2.preds, _inputs_error_rate_batch_size_2.targets),
    ],
)
class TestCharErrorRate(TextTester):
    """Test class for character error rate."""

    @pytest.mark.parametrize("ddp", [False, True])
    def test_cer_class(self, ddp, preds, targets):
        """Test modular version of cer."""
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            targets=targets,
            metric_class=CharErrorRate,
            reference_metric=_compare_fn,
        )

    def test_cer_functional(self, preds, targets):
        """Test functional version of cer."""
        self.run_functional_metric_test(
            preds,
            targets,
            metric_functional=char_error_rate,
            reference_metric=_compare_fn,
        )

    def test_cer_differentiability(self, preds, targets):
        """Test differentiability of cer metric."""
        self.run_differentiability_test(
            preds=preds,
            targets=targets,
            metric_module=CharErrorRate,
            metric_functional=char_error_rate,
        )
