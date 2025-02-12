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
import torchvision.transforms as transforms
from torch import Tensor

from torchmetrics.functional.image.arniqa import arniqa
from torchmetrics.image.arniqa import ARNIQA
from torchmetrics.utilities.imports import _TORCHVISION_AVAILABLE
from unittests._helpers import seed_all
from unittests._helpers.testers import MetricTester

seed_all(42)


_input_img = torch.rand(4, 2, 3, 224, 224)


class ARNIQATesterClass(ARNIQA):
    """Tester class for `ARNIQA` metric overriding its update method."""

    def update(self, preds, target):
        """Override the update method to support two input arguments."""
        super().update(preds)

    def compute(self):
        """Override the compute method."""
        return super().compute().sum()


def _arniqa_wrapped(preds, target, regressor_dataset="koniq10k", normalize=True):
    """Tester function for `arniqa` that supports two input arguments."""
    return arniqa(preds, regressor_dataset, normalize=normalize)


def _reference_arniqa(img: Tensor, target: Tensor, regressor_dataset: str, reduction: str = "mean") -> Tensor:
    """Comparison function (based on ARNIQA official repo (https://github.com/miccunifi/ARNIQA)) for tm implementation."""
    model = torch.hub.load(
        repo_or_dir="miccunifi/ARNIQA", source="github", model="ARNIQA", regressor_dataset=regressor_dataset
    )
    model.eval()

    h, w = img.shape[-2:]
    img_ds = transforms.Resize((h // 2, w // 2))(img)

    img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)
    img_ds = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img_ds)

    res = model(img, img_ds).detach().cpu().numpy()

    if reduction == "mean":
        return res.mean()
    return res.sum()


@pytest.mark.skipif(not _TORCHVISION_AVAILABLE, reason="test requires that torchvision is installed")
class TestARNIQA(MetricTester):
    """Test class for `ARNIQA` metric."""

    atol: float = 1e-6

    @pytest.mark.parametrize("regressor_dataset", ["kadid10k", "koniq10k"])
    @pytest.mark.parametrize("ddp", [pytest.param(True, marks=pytest.mark.DDP), False])
    def test_arniqa(self, regressor_dataset, ddp):
        """Test class implementation of metric."""
        self.run_class_metric_test(
            ddp=ddp,
            preds=_input_img,
            target=_input_img,
            metric_class=ARNIQATesterClass,
            reference_metric=partial(_reference_arniqa, regressor_dataset=regressor_dataset),
            check_scriptable=False,
            check_state_dict=False,
            metric_args={"regressor_dataset": regressor_dataset, "normalize": True},
        )

    def test_arniqa_functional(self):
        """Test functional implementation of metric."""
        self.run_functional_metric_test(
            preds=_input_img,
            target=_input_img,
            metric_functional=_arniqa_wrapped,
            reference_metric=partial(_reference_arniqa, regressor_dataset="koniq10k"),
            metric_args={"regressor_dataset": "koniq10k", "normalize": True},
        )

    def test_arniqa_differentiability(self):
        """Test the differentiability of the metric, according to its `is_differentiable` attribute."""
        self.run_differentiability_test(preds=_input_img, target=_input_img, metric_module=ARNIQATesterClass)

    # ARNIQA half + cpu does not work due to missing support in torch.min for older version of torch
    def test_arniqa_half_cpu(self):
        """Test for half + cpu support."""
        self.run_precision_test_cpu(preds=_input_img, target=_input_img, metric_module=ARNIQATesterClass)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
    def test_arniqa_half_gpu(self):
        """Test dtype support of the metric on GPU."""
        self.run_precision_test_gpu(preds=_input_img, target=_input_img, metric_module=ARNIQATesterClass)


@pytest.mark.skipif(not _TORCHVISION_AVAILABLE, reason="test requires that torchvision is installed")
def test_normalize_arg():
    """Test that normalize argument works as expected."""
    _input_img_norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(_input_img[0])
    metric_nonorm = ARNIQA(regressor_dataset="koniq10k", normalize=False)
    metric_norm = ARNIQA(regressor_dataset="koniq10k", normalize=True)
    res = metric_nonorm(_input_img_norm)
    res2 = metric_norm(_input_img[0])
    assert torch.allclose(res, res2, atol=1e-6), f"Results differ: max difference {torch.max((res - res2).abs())}"


@pytest.mark.skipif(not _TORCHVISION_AVAILABLE, reason="test requires that torchvision is installed")
def test_error_on_wrong_init():
    """Test class raises the expected errors."""
    with pytest.raises(ValueError, match="Argument `regressor_dataset` must be one .*"):
        ARNIQA(regressor_dataset="spaq")

    with pytest.raises(ValueError, match="Argument `reduction` must be one .*"):
        ARNIQA(regressor_dataset="koniq10k", reduction=None)


@pytest.mark.skipif(not _TORCHVISION_AVAILABLE, reason="test requires that torchvision is installed")
def test_error_on_wrong_input_shape():
    """Test error is raised on wrong input shape to update method."""
    inp = torch.rand(1, 1, 224, 224)
    metric = ARNIQA()
    with pytest.raises(ValueError, match="Input image must have .*"):
        metric(inp)


@pytest.mark.skipif(not _TORCHVISION_AVAILABLE, reason="test requires that torchvision is installed")
def test_error_on_wrong_normalize_value():
    """Test error is raised on wrong normalize parameter value to update method."""
    inp = torch.randn(1, 3, 224, 224)
    metric = ARNIQA(normalize=True)
    with pytest.raises(ValueError, match="Input image values must be .*"):
        metric(inp)


@pytest.mark.skipif(not _TORCHVISION_AVAILABLE, reason="test requires that torchvision is installed")
def test_check_for_backprop():
    """Check that by default the metric supports propagation of gradients, but does not update its parameters."""
    metric = ARNIQA()
    assert not metric.model.encoder[0].weight.requires_grad
    assert not metric.model.regressor.weight.requires_grad
    preds = _input_img[0]
    preds.requires_grad = True
    loss = metric(preds)
    assert loss.requires_grad
    loss.backward()
    assert metric.model.encoder[0].weight.grad is None
    assert metric.model.regressor.weight.grad is None
