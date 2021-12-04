from functools import partial
from typing import Sequence

import pytest
from torch import Tensor, tensor

from tests.text.helpers import INPUT_ORDER, TextTester
from tests.text.inputs import _input_multiple_references, _inputs_single_sentence_multiple_references
from torchmetrics.functional.text.chrf import chrf_score
from torchmetrics.text.chrf import CHRFScore
from torchmetrics.utilities.imports import _SACREBLEU_AVAILABLE

if _SACREBLEU_AVAILABLE:
    from sacrebleu.metrics import CHRF



def sacrebleu_chrf_fn(
    targets: Sequence[Sequence[str]],
    preds: Sequence[str],
    char_order: int,
    word_order: int,
    lowercase: bool,
    whitespace: bool,
) -> Tensor:
    sacrebleu_chrf = CHRF(
        char_order=char_order, word_order=word_order, lowercase=lowercase, whitespace=whitespace, eps_smoothing=True
    )
    # Sacrebleu CHRF expects different format of input
    targets = [[target[i] for target in targets] for i in range(len(targets[0]))]
    sacrebleu_chrf = sacrebleu_chrf.corpus_score(preds, targets).score / 100
    return tensor(sacrebleu_chrf)


@pytest.mark.parametrize(
    ["char_order", "word_order", "lowercase", "whitespace"],
    [
        pytest.param(6, 2, False, False),
        pytest.param(6, 2, False, True),
        pytest.param(4, 2, True, False),
        pytest.param(6, 0, True, False),
        pytest.param(6, 0, True, True),
        pytest.param(4, 0, False, True),
    ],
)
@pytest.mark.parametrize(
    ["preds", "targets"],
    [
        pytest.param(_input_multiple_references.preds, _input_multiple_references.target),
    ],
)
@pytest.mark.skipif(not _SACREBLEU_AVAILABLE, reason="test requires sacrebleu")
class TestCHRFScore(TextTester):
    @pytest.mark.parametrize("ddp", [False, True])
    @pytest.mark.parametrize("dist_sync_on_step", [False, True])
    def test_chrf_score_class(
        self, ddp, dist_sync_on_step, preds, targets, char_order, word_order, lowercase, whitespace
    ):
        metric_args = {
            "n_char_order": char_order,
            "n_word_order": word_order,
            "lowercase": lowercase,
            "whitespace": whitespace,
        }
        nltk_metric = partial(
            sacrebleu_chrf_fn, char_order=char_order, word_order=word_order, lowercase=lowercase, whitespace=whitespace
        )

        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            targets=targets,
            metric_class=CHRFScore,
            sk_metric=nltk_metric,
            dist_sync_on_step=dist_sync_on_step,
            metric_args=metric_args,
            input_order=INPUT_ORDER.TARGETS_FIRST,
        )

    def test_chrf_score_functional(self, preds, targets, char_order, word_order, lowercase, whitespace):
        metric_args = {
            "n_char_order": char_order,
            "n_word_order": word_order,
            "lowercase": lowercase,
            "whitespace": whitespace,
        }
        nltk_metric = partial(
            sacrebleu_chrf_fn, char_order=char_order, word_order=word_order, lowercase=lowercase, whitespace=whitespace
        )

        self.run_functional_metric_test(
            preds,
            targets,
            metric_functional=chrf_score,
            sk_metric=nltk_metric,
            metric_args=metric_args,
            input_order=INPUT_ORDER.TARGETS_FIRST,
        )

    def test_chrf_score_differentiability(self, preds, targets, char_order, word_order, lowercase, whitespace):
        metric_args = {
            "n_char_order": char_order,
            "n_word_order": word_order,
            "lowercase": lowercase,
            "whitespace": whitespace,
        }

        self.run_differentiability_test(
            preds=preds,
            targets=targets,
            metric_module=CHRFScore,
            metric_functional=chrf_score,
            metric_args=metric_args,
            input_order=INPUT_ORDER.TARGETS_FIRST,
        )


def test_chrf_empty_functional():
    hyp = []
    ref = [[]]
    assert chrf_score(ref, hyp) == tensor(0.0)


def test_chrf_empty_class():
    chrf = CHRFScore()
    hyp = []
    ref = [[]]
    assert chrf(ref, hyp) == tensor(0.0)


def test_chrf_return_sentence_level_score_functional():
    hyp = _inputs_single_sentence_multiple_references.preds
    ref =_inputs_single_sentence_multiple_references.target
    _, chrf_sentence_score = chrf_score(ref, hyp, return_sentence_level_score=True)
    isinstance(chrf_sentence_score, Tensor)


def test_chrf_return_sentence_level_class():
    chrf = CHRFScore(return_sentence_level_score=True)
    hyp = _inputs_single_sentence_multiple_references.preds
    ref =_inputs_single_sentence_multiple_references.target
    _, chrf_sentence_score = chrf(ref, hyp)
    isinstance(chrf_sentence_score, Tensor)
