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

Input = namedtuple("Input", ["preds", "target"])

# example taken from
# https://www.nltk.org/api/nltk.translate.html?highlight=bleu%20score#nltk.translate.bleu_score.corpus_bleu and adjusted
# EXAMPLE 1
HYPOTHESIS_A = "It is a guide to action which ensures that the military always obeys the commands of the party"
REFERENCE_1A = "It is a guide to action that ensures that the military will forever heed Party commands"
REFERENCE_2A = "It is a guiding principle which makes the military forces always being under the command of the Party"

# EXAMPLE 2
HYPOTHESIS_B = "he read the book because he was interested in world history"
REFERENCE_1B = "he was interested in world history because he read the book"
REFERENCE_2B = "It is the practical guide for the army always to heed the directions of the party"

# EXAMPLE 3 (add intentionally whitespaces)
HYPOTHESIS_C = "the cat the   cat on the mat "
REFERENCE_1C = "the  cat is     on the mat "
REFERENCE_2C = "there is a   cat on the mat"

TUPLE_OF_REFERENCES = (
    ((REFERENCE_1A, REFERENCE_2A), (REFERENCE_1B, REFERENCE_2B)),
    ((REFERENCE_1B, REFERENCE_2B), (REFERENCE_1C, REFERENCE_2C)),
)
TUPLE_OF_HYPOTHESES = ((HYPOTHESIS_A, HYPOTHESIS_B), (HYPOTHESIS_B, HYPOTHESIS_C))

_inputs_single_sentence_multiple_references = Input(preds=[HYPOTHESIS_B], target=[[REFERENCE_1B, REFERENCE_2B]])

_input_multiple_references = Input(preds=TUPLE_OF_HYPOTHESES, target=TUPLE_OF_REFERENCES)

_input_single_reference_batch_size_1 = Input()

_input_single_reference_batch_size_2 = Input()