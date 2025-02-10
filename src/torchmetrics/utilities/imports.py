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
"""Import utilities."""

import shutil
import sys

from lightning_utilities.core.imports import RequirementCache

_PYTHON_VERSION = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
_TORCH_GREATER_EQUAL_2_1 = RequirementCache("torch>=2.1.0")
_TORCH_GREATER_EQUAL_2_2 = RequirementCache("torch>=2.2.0")
_TORCH_GREATER_EQUAL_2_5 = RequirementCache("torch>=2.5.0")
_TORCH_LESS_THAN_2_6 = RequirementCache("torch<2.6.0")
_TORCHMETRICS_GREATER_EQUAL_1_6 = RequirementCache("torchmetrics>=1.7.0")

_NLTK_AVAILABLE = RequirementCache("nltk")
_ROUGE_SCORE_AVAILABLE = RequirementCache("rouge_score")
_BERTSCORE_AVAILABLE = RequirementCache("bert_score")
_SCIPY_AVAILABLE = RequirementCache("scipy")
_SCIPY_GREATER_EQUAL_1_8 = RequirementCache("scipy>=1.8.0")
_TORCH_FIDELITY_AVAILABLE = RequirementCache("torch_fidelity")
_LPIPS_AVAILABLE = RequirementCache("lpips")
_PYCOCOTOOLS_AVAILABLE = RequirementCache("pycocotools")
_TORCHVISION_AVAILABLE = RequirementCache("torchvision")
_TQDM_AVAILABLE = RequirementCache("tqdm")
_TRANSFORMERS_AVAILABLE = RequirementCache("transformers")
_TRANSFORMERS_GREATER_EQUAL_4_4 = RequirementCache("transformers>=4.4.0")
_TRANSFORMERS_GREATER_EQUAL_4_10 = RequirementCache("transformers>=4.10.0")
_PESQ_AVAILABLE = RequirementCache("pesq")
_GAMMATONE_AVAILABLE = RequirementCache("gammatone")
_TORCHAUDIO_AVAILABLE = RequirementCache("torchaudio")
_REGEX_AVAILABLE = RequirementCache("regex")
_PYSTOI_AVAILABLE = RequirementCache("pystoi")
_REQUESTS_AVAILABLE = RequirementCache("requests")
_LIBROSA_AVAILABLE = RequirementCache("librosa")
_ONNXRUNTIME_AVAILABLE = RequirementCache("onnxruntime")
_FAST_BSS_EVAL_AVAILABLE = RequirementCache("fast_bss_eval")
_MATPLOTLIB_AVAILABLE = RequirementCache("matplotlib")
_SCIENCEPLOT_AVAILABLE = RequirementCache("scienceplots")
_MULTIPROCESSING_AVAILABLE = RequirementCache("multiprocessing")
_XLA_AVAILABLE = RequirementCache("torch_xla")
_PIQ_GREATER_EQUAL_0_8 = RequirementCache("piq>=0.8.0")
_FASTER_COCO_EVAL_AVAILABLE = RequirementCache("faster_coco_eval")
_MECAB_AVAILABLE = RequirementCache("MeCab")
_MECAB_KO_AVAILABLE = RequirementCache("mecab_ko")
_MECAB_KO_DIC_AVAILABLE = RequirementCache("mecab_ko_dic")
_IPADIC_AVAILABLE = RequirementCache("ipadic")
_SENTENCEPIECE_AVAILABLE = RequirementCache("sentencepiece")
_SCIPI_AVAILABLE = RequirementCache("scipy")
_SKLEARN_GREATER_EQUAL_1_3 = RequirementCache("scikit-learn>=1.3.0")
_TORCH_LINEAR_ASSIGNMENT_AVAILABLE = RequirementCache("torch_linear_assignment")
_PYTDC_AVAILABLE = RequirementCache("pyTDC")
_LATEX_AVAILABLE: bool = shutil.which("latex") is not None
