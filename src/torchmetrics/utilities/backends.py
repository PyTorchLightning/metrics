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
from types import ModuleType
from typing import Literal

from torchmetrics.utilities.imports import (
    _FASTER_COCO_EVAL_AVAILABLE,
    _PYCOCOTOOLS_AVAILABLE,
)


def _load_coco_backend_tools(backend: Literal["pycocotools", "faster_coco_eval"]) -> tuple[object, object, ModuleType]:
    """Load the backend tools for the given backend."""
    if backend == "pycocotools":
        if not _PYCOCOTOOLS_AVAILABLE:
            raise ModuleNotFoundError(
                "Backend `pycocotools` in metric `MeanAveragePrecision`  metric requires that `pycocotools` is"
                " installed. Please install with `pip install pycocotools` or `pip install torchmetrics[detection]`"
            )
        import pycocotools.mask as mask_utils
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval

        return COCO, COCOeval, mask_utils

    if not _FASTER_COCO_EVAL_AVAILABLE:
        raise ModuleNotFoundError(
            "Backend `faster_coco_eval` in metric `MeanAveragePrecision`  metric requires that `faster-coco-eval` is"
            " installed. Please install with `pip install faster-coco-eval`."
        )
    from faster_coco_eval import COCO
    from faster_coco_eval import COCOeval_faster as COCOeval
    from faster_coco_eval.core import mask as mask_utils

    return COCO, COCOeval, mask_utils
