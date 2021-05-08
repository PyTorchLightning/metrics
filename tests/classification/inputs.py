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
from collections import namedtuple

import torch

from tests.helpers.testers import BATCH_SIZE, EXTRA_DIM, NUM_BATCHES, NUM_CLASSES

Input = namedtuple('Input', ["preds", "target"])

_input_binary_prob = Input(
    preds=torch.rand(NUM_BATCHES, BATCH_SIZE), target=torch.randint(high=2, size=(NUM_BATCHES, BATCH_SIZE))
)

_input_binary = Input(
    preds=torch.randint(high=2, size=(NUM_BATCHES, BATCH_SIZE)),
    target=torch.randint(high=2, size=(NUM_BATCHES, BATCH_SIZE))
)

_input_binary_logits = Input(
    preds=torch.randn(NUM_BATCHES, BATCH_SIZE), target=torch.randint(high=2, size=(NUM_BATCHES, BATCH_SIZE))
)

_input_multilabel_prob = Input(
    preds=torch.rand(NUM_BATCHES, BATCH_SIZE, NUM_CLASSES),
    target=torch.randint(high=2, size=(NUM_BATCHES, BATCH_SIZE, NUM_CLASSES))
)

_input_multilabel_multidim_prob = Input(
    preds=torch.rand(NUM_BATCHES, BATCH_SIZE, NUM_CLASSES, EXTRA_DIM),
    target=torch.randint(high=2, size=(NUM_BATCHES, BATCH_SIZE, NUM_CLASSES, EXTRA_DIM))
)

_input_multilabel_logits = Input(
    preds=torch.randn(NUM_BATCHES, BATCH_SIZE, NUM_CLASSES),
    target=torch.randint(high=2, size=(NUM_BATCHES, BATCH_SIZE, NUM_CLASSES))
)

_input_multilabel = Input(
    preds=torch.randint(high=2, size=(NUM_BATCHES, BATCH_SIZE, NUM_CLASSES)),
    target=torch.randint(high=2, size=(NUM_BATCHES, BATCH_SIZE, NUM_CLASSES))
)


_input_multilabel_multidim = Input(
    preds=torch.randint(high=2, size=(NUM_BATCHES, BATCH_SIZE, NUM_CLASSES, EXTRA_DIM)),
    target=torch.randint(high=2, size=(NUM_BATCHES, BATCH_SIZE, NUM_CLASSES, EXTRA_DIM))
)

# Generate edge multilabel edge case, where nothing matches (scores are undefined)
__temp_preds = torch.randint(high=2, size=(NUM_BATCHES, BATCH_SIZE, NUM_CLASSES))
__temp_target = abs(__temp_preds - 1)

_input_multilabel_no_match = Input(preds=__temp_preds, target=__temp_target)

__mc_prob_logits = torch.randn(NUM_BATCHES, BATCH_SIZE, NUM_CLASSES)
__mc_prob_preds = __mc_prob_logits.abs() / __mc_prob_logits.abs().sum(dim=2, keepdim=True)

_input_multiclass_prob = Input(
    preds=__mc_prob_preds, target=torch.randint(high=NUM_CLASSES, size=(NUM_BATCHES, BATCH_SIZE))
)

_input_multiclass_logits = Input(
    preds=__mc_prob_logits, target=torch.randint(high=NUM_CLASSES, size=(NUM_BATCHES, BATCH_SIZE))
)

_input_multiclass = Input(
    preds=torch.randint(high=NUM_CLASSES, size=(NUM_BATCHES, BATCH_SIZE)),
    target=torch.randint(high=NUM_CLASSES, size=(NUM_BATCHES, BATCH_SIZE))
)

__mdmc_prob_preds = torch.rand(NUM_BATCHES, BATCH_SIZE, NUM_CLASSES, EXTRA_DIM)
__mdmc_prob_preds = __mdmc_prob_preds / __mdmc_prob_preds.sum(dim=2, keepdim=True)

_input_multidim_multiclass_prob = Input(
    preds=__mdmc_prob_preds, target=torch.randint(high=NUM_CLASSES, size=(NUM_BATCHES, BATCH_SIZE, EXTRA_DIM))
)

_input_multidim_multiclass = Input(
    preds=torch.randint(high=NUM_CLASSES, size=(NUM_BATCHES, BATCH_SIZE, EXTRA_DIM)),
    target=torch.randint(high=NUM_CLASSES, size=(NUM_BATCHES, BATCH_SIZE, EXTRA_DIM))
)


# Generate plausible-looking inputs
def generate_plausible_inputs_multilabel(num_classes=NUM_CLASSES, num_batches=NUM_BATCHES, batch_size=BATCH_SIZE):
    correct_targets = torch.randint(high=num_classes, size=(num_batches, batch_size))
    preds = torch.rand(num_batches, batch_size, num_classes)
    targets = torch.zeros_like(preds, dtype=torch.long)
    for i in range(preds.shape[0]):
        for j in range(preds.shape[1]):
            targets[i, j, correct_targets[i, j]] = 1
    preds += torch.rand(num_batches, batch_size, num_classes) * targets / 3

    preds = preds / preds.sum(dim=2, keepdim=True)

    return Input(preds=preds, target=targets)


def generate_plausible_inputs_binary(num_batches=NUM_BATCHES, batch_size=BATCH_SIZE):
    targets = torch.randint(high=2, size=(num_batches, batch_size))
    preds = torch.rand(num_batches, batch_size) + torch.rand(num_batches, batch_size) * targets / 3
    return Input(preds=preds / (preds.max() + 0.01), target=targets)


_input_multilabel_prob_plausible = generate_plausible_inputs_multilabel()

_input_binary_prob_plausible = generate_plausible_inputs_binary()
