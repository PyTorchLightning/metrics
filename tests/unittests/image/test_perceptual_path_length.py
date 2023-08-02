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
import pytest
import torch
import torch_fidelity
from torch import nn
from torchmetrics.functional.image.perceptual_path_length import perceptual_path_length
from torchmetrics.image.perceptual_path_length import PerceptualPathLength
from torchmetrics.utilities.imports import _TORCH_FIDELITY_AVAILABLE


class DummyGenerator(torch.nn.Module):
    """From https://github.com/toshas/torch-fidelity/blob/master/examples/sngan_cifar10.py."""

    def __init__(self, z_size) -> None:
        super().__init__()
        self.z_size = z_size
        self.model = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(z_size, 512, 4, stride=1),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(512, 256, 4, stride=2, padding=(1, 1)),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(256, 128, 4, stride=2, padding=(1, 1)),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(128, 64, 4, stride=2, padding=(1, 1)),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(64, 3, 3, stride=1, padding=(1, 1)),
            torch.nn.Tanh(),
        )

    def forward(self, z):
        """Generate images from latent vectors."""
        fake = self.model(z.view(-1, self.z_size, 1, 1))
        if not self.training:
            fake = 255 * (fake.clamp(-1, 1) * 0.5 + 0.5)
            fake = fake.to(torch.uint8)
        return fake

    def sample(self, num_samples):
        """Sample latent vectors."""
        return torch.randn(num_samples, self.z_size)


@pytest.mark.skipif(not _TORCH_FIDELITY_AVAILABLE, reason="test requires torch_fidelity")
@pytest.mark.parametrize(
    ("argument", "match"),
    [
        ({"num_samples": 0}, "Argument `num_samples` must be a positive integer, but got 0."),
        ({"conditional": 2}, "Argument `conditional` must be a boolean, but got 2."),
        ({"batch_size": 0}, "Argument `batch_size` must be a positive integer, but got 0."),
        ({"interpolation_method": "wrong"}, "Argument `interpolation_method` must be one of.*"),
        ({"epsilon": 0}, "Argument `epsilon` must be a positive float, but got 0."),
        ({"resize": 0}, "Argument `resize` must be a positive integer or `None`, but got 0."),
        ({"lower_discard": -1}, "Argument `lower_discard` must be a float between 0 and 1 or `None`, but got -1"),
        ({"upper_discard": 2}, "Argument `upper_discard` must be a float between 0 and 1 or `None`, but got 2"),
    ],
)
def test_raises_error_on_wrong_arguments(argument, match):
    """Test that appropriate errors are raised on wrong arguments."""
    with pytest.raises(ValueError, match=match):
        perceptual_path_length(DummyGenerator(128), **argument)

    with pytest.raises(ValueError, match=match):
        PerceptualPathLength(**argument)


class _WrongGenerator1(nn.Module):
    pass


class _WrongGenerator2(nn.Module):
    sample = 1


class _WrongGenerator3(nn.Module):
    def sample(self, n):
        return torch.randn(n, 2)


class _WrongGenerator4(nn.Module):
    def sample(self, n):
        return torch.randn(n, 2)

    @property
    def num_classes(self):
        return [10, 10]


@pytest.mark.parametrize(
    ("generator", "errortype", "match"),
    [
        (_WrongGenerator1(), NotImplementedError, "The generator must have a `sample` method.*"),
        (_WrongGenerator2(), ValueError, "The generator's `sample` method must be callable."),
        (
            _WrongGenerator3(),
            AttributeError,
            "The generator must have a `num_classes` attribute when `conditional=True`.",
        ),
        (
            _WrongGenerator4(),
            ValueError,
            "The generator's `num_classes` attribute must be an integer when `conditional=True`.",
        ),
    ],
)
def test_raises_error_on_wrong_generator(generator, errortype, match):
    """Test that appropriate errors are raised on wrong generator."""
    with pytest.raises(errortype, match=match):
        perceptual_path_length(generator, conditional=True)

    ppl = PerceptualPathLength(conditional=True)
    with pytest.raises(errortype, match=match):
        ppl.update(generator=generator)


@pytest.mark.skipif(not _TORCH_FIDELITY_AVAILABLE, reason="test requires torch_fidelity")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires GPU machine")
def test_compare():
    """Test against torch_fidelity.

    Because it is a sample based metric, the results are not deterministic. Thus we need a large amount of samples to
    even get close to the reference value. Even then we are going to allow a 5% deviation on the mean and 5% deviation
    on the standard deviation.

    """
    generator = DummyGenerator(128)

    compare = torch_fidelity.calculate_metrics(
        input1=torch_fidelity.GenerativeModelModuleWrapper(generator, 128, "normal", 10),
        input1_model_num_samples=50000,
        ppl=True,
        ppl_reduction="none",
        input_model_num_classes=0,
        ppl_discard_percentile_lower=None,
        ppl_discard_percentile_higher=None,
    )
    compare = torch.tensor(compare["perceptual_path_length_raw"])

    result = perceptual_path_length(
        generator, num_samples=50000, conditional=False, lower_discard=None, upper_discard=None, device="cuda"
    )
    result = result[-1].cpu()

    assert 0.95 * result.mean() <= compare.mean() <= 1.05 * result.mean()
    assert 0.95 * result.std() <= compare.std() <= 1.05 * result.std()
