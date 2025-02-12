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
from torchmetrics.functional.image.arniqa import arniqa
from torchmetrics.functional.image.d_lambda import spectral_distortion_index
from torchmetrics.functional.image.d_s import spatial_distortion_index
from torchmetrics.functional.image.ergas import error_relative_global_dimensionless_synthesis
from torchmetrics.functional.image.gradients import image_gradients
from torchmetrics.functional.image.lpips import learned_perceptual_image_patch_similarity
from torchmetrics.functional.image.perceptual_path_length import perceptual_path_length
from torchmetrics.functional.image.psnr import peak_signal_noise_ratio
from torchmetrics.functional.image.psnrb import peak_signal_noise_ratio_with_blocked_effect
from torchmetrics.functional.image.qnr import quality_with_no_reference
from torchmetrics.functional.image.rase import relative_average_spectral_error
from torchmetrics.functional.image.rmse_sw import root_mean_squared_error_using_sliding_window
from torchmetrics.functional.image.sam import spectral_angle_mapper
from torchmetrics.functional.image.scc import spatial_correlation_coefficient
from torchmetrics.functional.image.ssim import (
    multiscale_structural_similarity_index_measure,
    structural_similarity_index_measure,
)
from torchmetrics.functional.image.tv import total_variation
from torchmetrics.functional.image.uqi import universal_image_quality_index
from torchmetrics.functional.image.vif import visual_information_fidelity

__all__ = [
    "arniqa",
    "error_relative_global_dimensionless_synthesis",
    "image_gradients",
    "learned_perceptual_image_patch_similarity",
    "multiscale_structural_similarity_index_measure",
    "peak_signal_noise_ratio",
    "peak_signal_noise_ratio_with_blocked_effect",
    "perceptual_path_length",
    "quality_with_no_reference",
    "relative_average_spectral_error",
    "root_mean_squared_error_using_sliding_window",
    "spatial_correlation_coefficient",
    "spatial_distortion_index",
    "spectral_angle_mapper",
    "spectral_distortion_index",
    "structural_similarity_index_measure",
    "total_variation",
    "universal_image_quality_index",
    "visual_information_fidelity",
]
