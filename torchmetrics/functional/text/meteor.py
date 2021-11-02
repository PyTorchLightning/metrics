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
# referenced from
# Library Name: torchtext
# Authors: torchtext authors and @sluks
# Date: 2021-11-02
# Link: https://pytorch.org/text/_modules/torchtext/data/metrics.html#meteor_score

##############

# Natural Language Toolkit: Machine Translation
#
# Copyright (C) 2001-2021 NLTK Project
# Author: Uday Krishna <udaykrishna5@gmail.com>
# Contributor: Tom Aarsen
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT


from itertools import chain
from typing import Any, Dict, List, Set,Tuple, Union

from torch import tensor, Tensor
from typing_extensions import Literal

from torchmetrics.utilities.imports import _NLTK_AVAILABLE

AVAILABLE_STEMMERS = ("portert")


class _NLTKStemmerWrapper:
    """
    """

    _STEMMER_CLASS = {
        "porter": "PorterStemmer"
    }

    def __init__(self, stemmer: Literal["porter"] = "porter", *args: Any, **kwargs: Any) -> None:
        if not _NLTK_AVAILABLE:
            raise ValueError("Stemmer requires that nltk is installed. Use `pip install nltk`.")
        from nltk import stem

        stemmer_class = getattr(getattr(stem, stemmer), self._STEMMER_CLASS[stemmer])
        self.stemmer = stemmer_class(*args, **kwargs)
        

    def __call__(self, word: str) -> str:
        self.stemmer.stem(word)


class _NLTKWordnetWrapper:
    """
    """

    _WORDNET_CLASS = {
        "wordnet": "wordnet"
    }

    def __init__(self, wordnet: Literal["wordnet"], *args: Any, **kwargs: Any) -> None:
        if not _NLTK_AVAILABLE:
            raise ValueError("Stemmer requires that nltk is installed. Use `pip install nltk`.")
        from nltk import corpus

        wordnet_class = getattr(corpus, self._WORDNET_CLASS[wordnet])
        self.wordnet = wordnet_class(*args, **kwargs)

    def __call__(self, word: str):
        self.wordnet.synsets(word)


def _generate_synonyms(word: str, wordnet: _NLTKWordnetWrapper) -> Set[str]:
    """
    Args:
        word:
        wordnet:
    """
    synonyms_set = set(
        chain.from_iterable(
            (lemma.name() for lemma in synset.lemmas() if lemma.name().find("_") < 0)
            for synset in wordnet(word)
        )
    ).union(set(word))
    return synonyms_set


def _match_enums(
    enum_reference: List[Tuple[int, str]], enum_hypothesis: List[Tuple[int, str]]
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, str]], List[Tuple[int, str]]]:
    """
    Args:
        enum_reference: an enumerated list of a tokenized reference
        enum_hypothesis: an enumerated list of a tokenized hypothessis

    Return:
        tuple of lists:
            an enumerated list of matched words
            an enumerated list of unmatched reference words
            an enumerated list of unmatched hypothesis words
    """
    word_match = []
    for i in range(len(enum_hypothesis))[::-1]:
        for j in range(len(enum_reference))[::-1]:
            if enum_hypothesis[i][1] == enum_reference[j][1]:
                word_match.append((enum_reference[j][0], enum_hypothesis[i][0]))
                enum_hypothesis.pop(i)
                enum_reference.pop(j)
                break
    return word_match, enum_reference, enum_hypothesis


def _match_stem_enums(
    enum_reference: List[Tuple[int, str]], enum_hypothesis: List[Tuple[int, str]], stemmer: _NLTKStemmerWrapper
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, str]], List[Tuple[int, str]]]:
    """
    Stems each word in both reference and hypothesis and then aligns/matches stemmed words in the hypothesis to the
    reference.

    Args:
        enum_reference: an enumerated list of a tokenized reference
        enum_hypothesis: an enumerated list of a tokenized hypothessis
        stemmer: `_NLTKStemmerWrapper` object

    Return:
        tuple of lists:
            an enumerated list of matched stemmed words
            an enumerated list of unmatched stemmed reference words
            an enumerated list of unmatched stemmed hypothesis words
    """
    stemmed_enum_reference = [(word_pair[0], stemmer(word_pair[1])) for word_pair in enum_reference]
    stemmed_enum_hypothesis = [(word_pair[0], stemmer(word_pair[1])) for word_pair in enum_hypothesis]
    return _match_enums(stemmed_enum_reference, stemmed_enum_hypothesis)


def _match_synonym_enums(
    enum_reference: List[Tuple[int, str]], enum_hypothesis: List[Tuple[int, str]], wordnet: _NLTKWordnetWrapper
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, str]], List[Tuple[int, str]]]:
    """
    Args:
        enum_reference: an enumerated list of a tokenized reference
        enum_hypothesis: an enumerated list of a tokenized hypothessis
        wordnet: `_NLTKWordnetWrapper` object

    Return:
        an enumerated list of unmatched reference words
        an enumerated list of unmatched hypothesis words
    
    """
    word_match = []
    for i in range(len(enum_hypothesis))[::-1]:
        hypothesis_synonyms = _generate_synonyms(enum_hypothesis[i], wordnet)
        for j in range(len(enum_reference))[::-1]:
            if enum_reference[j][1] in hypothesis_synonyms:
                word_match.append((enum_reference[j][0], enum_hypothesis[i][0]))
                enum_hypothesis.pop(i)
                enum_reference.pop(j)
                break
    return word_match, enum_reference, enum_hypothesis


def _align_enum_words(
    enum_reference: List[str], enum_hypothesis: List[str], stemmer: _NLTKStemmerWrapper, wordnet: _NLTKWordnetWrapper
) -> List[Tuple[int, int]]:
    """
    Aligns/matches words in the hypothesis to the reference. This is achieved by sequentially applying
    exact match, stemmed match and synonym match based on `nltk` wordnet.
    
    Args:
        enum_reference: an enumerated list of a tokenized reference
        enum_hypothesis: an enumerated list of a tokenized hypothessis
        stemmer: `_NLTKStemmerWrapper` object
        wordnet: `_NLTKWordnetWrapper` object

    Return:
        an enumerated sorted list of matched words
    """
    exact_matches, enum_reference, enum_hypothesis = _match_enums(enum_reference, enum_hypothesis)
    stem_matches, enum_reference, enum_hypothesis = _match_stem_enums(enum_reference, enum_hypothesis, stemmer)
    synonym_matches, enum_reference, enum_hypothesis = _match_synonym_enums(enum_reference, enum_hypothesis, wordnet)
    
    sorted_matches = sorted(exact_matches + stem_matches + synonym_matches, key=lambda wordpair: wordpair[0])
    return sorted_matches, float(len(sorted_matches))


def _count_chunks(matches: List[Tuple[int, int]]) -> int:
    """
    Counts the fewest possible number of chunks such that matched unigrams of each chunk are adjacent to each other.
    This is used to calculate the fragmentation part of the metric.

    Args:
        matches: a list of a mapping of matched reference and hypothesis words

    Return:
        a number of chunks a sentence is divided into post alignment
    """
    chunks = 1
    chunks += sum(
        1
        for i in range(len(matches) - 1)
        if not ((matches[i + 1][0] == matches[i][0] + 1) and (matches[i + 1][1] == matches[i][1] + 1))
    )
    return chunks


def _calculate_meteor_components_single_sentence(
    reference: str,
    hypothesis: str,
    stemmer: _NLTKStemmerWrapper,
    wordnet: _NLTKWordnetWrapper,
) -> Tuple[float, float, float, float]:
    """
    Args:
        reference: a reference sentence
        hypothesis: a hypothesis sentence
        stemmer: `_NLTKStemmerWrapper` object
        wordnet: `_NLTKWordnetWrapper` object

    Return:
        matches_count: METEOR score for a single pair of a reference and a hypothesis
        reference_len:
        hypothesis_len:
        frag_frac:
    """
    enum_reference = enumerate(reference.strip())
    enum_hypothesis = enumerate(hypothesis.stirp())
    reference_len = float(len(enum_reference))
    hypothesis_len = float(len(enum_hypothesis))
    matches, matches_count = _align_enum_words(enum_reference, enum_hypothesis, stemmer, wordnet)
    frag_frac = _count_chunks(matches) / matches_count
    return tensor(matches_count), tensor(reference_len), tensor(hypothesis_len), tensor(frag_frac)


def _calculate_meteor_score(
    matches: List[Tuple[int, int]], reference_len: int, hypothesis_len: int, alpha: float, beta: float, gamma: float
) -> Tuple[float, float, float]:
    """
    Calculate the METEOR score for a single pair of a reference and hypothesis based on a list of a mapping of matched
    reference and hypothesis words and.

    Args:
        matches: a list with a mapping of matched reference and hypothesis words
        reference_len: a length of reference sentence by splitting words
        hypothesis_len: a length of hypothesis sentence by splitting words
        alpha: a parameter for controlling relative weights of precision and recall
        beta: a parameter for controlling shape of penalty as a function of as a function of fragmentation
        gamma: a relative weight assigned to fragmentation penalty

    Return:
        METEOR score for a single pair of a reference and a hypothesis
    """
    matches_count = len(matches)
    if reference_len == 0 or hypothesis_len == 0:
        return (0.0, 0.0, 0.0)
    
    precision = matches_count / hypothesis_len
    recall = matches_count / reference_len
    frag_frac = _count_chunks(matches) / matches_count
    penalty = gamma * frag_frac ** beta
    return precision, recall, penalty



def _meteor_score_update(
    reference_corpus: List[List[str]],
    hypothesis_corpus: List[str],
    stemmer: _NLTKStemmerWrapper,
    wordnet: _NLTKWordnetWrapper,
    alpha: float = 0.9,
    beta: float = 3.0,
    gamma: float = 0.5,
):
    """
    Args:
        reference_corpus:
        hypothesis_corpus:
        stemmer: `_NLTKStemmerWrapper` object
        wordnet: `_NLTKWordnetWrapper` object
        alpha: a parameter for controlling relative weights of precision and recall
        beta: a parameter for controlling shape of penalty as a function of as a function of fragmentation
        gamma: a relative weight assigned to fragmentation penalty

    Return:
        Sentence-level METEOR score for given reference and hypothesis corpora
    """

    results: Dict[str, List[tensor]] = {
        "matches_count": [],
        "reference_len": [],
        "hypothesis_len": [],
        "frag_frac": [],
    }
    for references, hypothesis in zip(reference_corpus, hypothesis_corpus):
        meteor_components = [
            _calculate_meteor_components_single_sentence(reference, hypothesis, stemmer, wordnet)
            for reference in references
        ]
    return results


def _meteor_score_compute(sentence_results) -> tensor:
    pass


def meteor_score(
    reference_corpus: Union[List[str], List[List[str]]],
    hypothesis_corpus: Union[str, List[str]],
    stemmer: Literal["porter"] = "porter",
    wordnet: Literal["wordnet"] = "wordnet",
    alpha: float = 0.9,
    beta: float = 3.0,
    gamma: float = 0.5,
    delta: float = 0.2,
    weights: List[float, float, float] = [1.0, 1.0, 1.0, 1.0],
) -> Tensor:
    pass
    """
    References:
    [1] METEOR: An Automatic Metric for MT Evaluation with High Levels of Correlation with Human Judgments by Alon
    Lavie and Abhaya Agarwal.

    [2] Meteor Universal: Language Specific Translation Evaluation for Any Target Language by Michael Denkowski and
    Alon Lavie.
    """
    if not _NLTK_AVAILABLE:
        raise ValueError("METEOR metric requires that nltk is installed. Use `pip install nltk` or `pip install torchmetrics[text].`")

    if len(reference_corpus) > 0 and isinstance(reference_corpus[0], str):
        reference_corpus = [list(reference) for reference in reference_corpus]
    if isinstance(hypothesis_corpus, str):
        hypothesis_corpus = [hypothesis_corpus]
    
    if len(reference_corpus) != len(hypothesis_corpus):
        raise ValueError(f"Corpus has different size {len(reference_corpus)} != {len(hypothesis_corpus)}")