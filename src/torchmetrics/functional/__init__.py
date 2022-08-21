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
from torchmetrics.functional.audio.pit import permutation_invariant_training, pit_permutate
from torchmetrics.functional.audio.sdr import scale_invariant_signal_distortion_ratio, signal_distortion_ratio
from torchmetrics.functional.audio.snr import scale_invariant_signal_noise_ratio, signal_noise_ratio
from torchmetrics.functional.classification.accuracy import (
    accuracy,
    binary_accuracy,
    multiclass_accuracy,
    multilabel_accuracy,
)
from torchmetrics.functional.classification.auc import auc
from torchmetrics.functional.classification.auroc import auroc, binary_auroc, multiclass_auroc, multilabel_auroc
from torchmetrics.functional.classification.average_precision import (
    average_precision,
    binary_average_precision,
    multiclass_average_precision,
    multilabel_average_precision,
)
from torchmetrics.functional.classification.calibration_error import (
    binary_calibration_error,
    calibration_error,
    multiclass_calibration_error,
)
from torchmetrics.functional.classification.cohen_kappa import binary_cohen_kappa, cohen_kappa, multiclass_cohen_kappa
from torchmetrics.functional.classification.confusion_matrix import (
    binary_confusion_matrix,
    confusion_matrix,
    multiclass_confusion_matrix,
    multilabel_confusion_matrix,
)
from torchmetrics.functional.classification.dice import dice, dice_score
from torchmetrics.functional.classification.exact_match import multilabel_exact_match
from torchmetrics.functional.classification.f_beta import (
    binary_f1_score,
    binary_fbeta_score,
    f1_score,
    fbeta_score,
    multiclass_f1_score,
    multiclass_fbeta_score,
    multilabel_f1_score,
    multilabel_fbeta_score,
)
from torchmetrics.functional.classification.hamming import (
    binary_hamming_distance,
    hamming_distance,
    multiclass_hamming_distance,
    multilabel_hamming_distance,
)
from torchmetrics.functional.classification.hinge import binary_hinge_loss, hinge_loss, multiclass_hinge_loss
from torchmetrics.functional.classification.jaccard import (
    binary_jaccard_index,
    jaccard_index,
    multiclass_jaccard_index,
    multilabel_jaccard_index,
)
from torchmetrics.functional.classification.matthews_corrcoef import (
    binary_matthews_corrcoef,
    matthews_corrcoef,
    multiclass_matthews_corrcoef,
    multilabel_matthews_corrcoef,
)
from torchmetrics.functional.classification.precision_recall import (
    binary_precision,
    binary_recall,
    multiclass_precision,
    multiclass_recall,
    multilabel_precision,
    multilabel_recall,
    precision,
    precision_recall,
    recall,
)
from torchmetrics.functional.classification.precision_recall_curve import (
    binary_precision_recall_curve,
    multiclass_precision_recall_curve,
    multilabel_precision_recall_curve,
    precision_recall_curve,
)
from torchmetrics.functional.classification.ranking import (
    coverage_error,
    label_ranking_average_precision,
    label_ranking_loss,
    multilabel_coverage_error,
    multilabel_ranking_average_precision,
    multilabel_ranking_loss,
)
from torchmetrics.functional.classification.recall_at_fixed_precision import (
    binary_recall_at_fixed_precision,
    multiclass_recall_at_fixed_precision,
    multilabel_recall_at_fixed_precision,
)
from torchmetrics.functional.classification.roc import binary_roc, multiclass_roc, multilabel_roc, roc
from torchmetrics.functional.classification.specificity import (
    binary_specificity,
    multiclass_specificity,
    multilabel_specificity,
    specificity,
)
from torchmetrics.functional.classification.stat_scores import (
    binary_stat_scores,
    multiclass_stat_scores,
    multilabel_stat_scores,
    stat_scores,
)
from torchmetrics.functional.image.d_lambda import spectral_distortion_index
from torchmetrics.functional.image.ergas import error_relative_global_dimensionless_synthesis
from torchmetrics.functional.image.gradients import image_gradients
from torchmetrics.functional.image.psnr import peak_signal_noise_ratio
from torchmetrics.functional.image.sam import spectral_angle_mapper
from torchmetrics.functional.image.ssim import (
    multiscale_structural_similarity_index_measure,
    structural_similarity_index_measure,
)
from torchmetrics.functional.image.uqi import universal_image_quality_index
from torchmetrics.functional.pairwise.cosine import pairwise_cosine_similarity
from torchmetrics.functional.pairwise.euclidean import pairwise_euclidean_distance
from torchmetrics.functional.pairwise.linear import pairwise_linear_similarity
from torchmetrics.functional.pairwise.manhattan import pairwise_manhattan_distance
from torchmetrics.functional.regression.cosine_similarity import cosine_similarity
from torchmetrics.functional.regression.explained_variance import explained_variance
from torchmetrics.functional.regression.kl_divergence import kl_divergence
from torchmetrics.functional.regression.log_mse import mean_squared_log_error
from torchmetrics.functional.regression.mae import mean_absolute_error
from torchmetrics.functional.regression.mape import mean_absolute_percentage_error
from torchmetrics.functional.regression.mse import mean_squared_error
from torchmetrics.functional.regression.pearson import pearson_corrcoef
from torchmetrics.functional.regression.r2 import r2_score
from torchmetrics.functional.regression.spearman import spearman_corrcoef
from torchmetrics.functional.regression.symmetric_mape import symmetric_mean_absolute_percentage_error
from torchmetrics.functional.regression.tweedie_deviance import tweedie_deviance_score
from torchmetrics.functional.regression.wmape import weighted_mean_absolute_percentage_error
from torchmetrics.functional.retrieval.average_precision import retrieval_average_precision
from torchmetrics.functional.retrieval.fall_out import retrieval_fall_out
from torchmetrics.functional.retrieval.hit_rate import retrieval_hit_rate
from torchmetrics.functional.retrieval.ndcg import retrieval_normalized_dcg
from torchmetrics.functional.retrieval.precision import retrieval_precision
from torchmetrics.functional.retrieval.precision_recall_curve import retrieval_precision_recall_curve
from torchmetrics.functional.retrieval.r_precision import retrieval_r_precision
from torchmetrics.functional.retrieval.recall import retrieval_recall
from torchmetrics.functional.retrieval.reciprocal_rank import retrieval_reciprocal_rank
from torchmetrics.functional.text.bleu import bleu_score
from torchmetrics.functional.text.cer import char_error_rate
from torchmetrics.functional.text.chrf import chrf_score
from torchmetrics.functional.text.eed import extended_edit_distance
from torchmetrics.functional.text.mer import match_error_rate
from torchmetrics.functional.text.perplexity import perplexity
from torchmetrics.functional.text.rouge import rouge_score
from torchmetrics.functional.text.sacre_bleu import sacre_bleu_score
from torchmetrics.functional.text.squad import squad
from torchmetrics.functional.text.ter import translation_edit_rate
from torchmetrics.functional.text.wer import word_error_rate
from torchmetrics.functional.text.wil import word_information_lost
from torchmetrics.functional.text.wip import word_information_preserved
from torchmetrics.utilities.imports import _TRANSFORMERS_AVAILABLE

if _TRANSFORMERS_AVAILABLE:
    from torchmetrics.functional.text.bert import bert_score  # noqa: F401
    from torchmetrics.functional.text.infolm import infolm  # noqa: F401

__all__ = [
    "accuracy",
    "auc",
    "auroc",
    "average_precision",
    "bleu_score",
    "calibration_error",
    "char_error_rate",
    "chrf_score",
    "cohen_kappa",
    "confusion_matrix",
    "cosine_similarity",
    "coverage_error",
    "tweedie_deviance_score",
    "dice_score",
    "dice",
    "error_relative_global_dimensionless_synthesis",
    "explained_variance",
    "extended_edit_distance",
    "f1_score",
    "fbeta_score",
    "hamming_distance",
    "hinge_loss",
    "image_gradients",
    "jaccard_index",
    "kl_divergence",
    "label_ranking_average_precision",
    "label_ranking_loss",
    "match_error_rate",
    "matthews_corrcoef",
    "mean_absolute_error",
    "mean_absolute_percentage_error",
    "mean_squared_error",
    "mean_squared_log_error",
    "multiscale_structural_similarity_index_measure",
    "pairwise_cosine_similarity",
    "pairwise_euclidean_distance",
    "pairwise_linear_similarity",
    "pairwise_manhattan_distance",
    "pearson_corrcoef",
    "permutation_invariant_training",
    "perplexity",
    "pit_permutate",
    "precision",
    "precision_recall",
    "precision_recall_curve",
    "peak_signal_noise_ratio",
    "r2_score",
    "recall",
    "retrieval_average_precision",
    "retrieval_fall_out",
    "retrieval_hit_rate",
    "retrieval_normalized_dcg",
    "retrieval_precision",
    "retrieval_r_precision",
    "retrieval_recall",
    "retrieval_reciprocal_rank",
    "retrieval_precision_recall_curve",
    "roc",
    "rouge_score",
    "sacre_bleu_score",
    "signal_distortion_ratio",
    "scale_invariant_signal_distortion_ratio",
    "scale_invariant_signal_noise_ratio",
    "signal_noise_ratio",
    "spearman_corrcoef",
    "specificity",
    "spectral_distortion_index",
    "squad",
    "structural_similarity_index_measure",
    "stat_scores",
    "symmetric_mean_absolute_percentage_error",
    "translation_edit_rate",
    "universal_image_quality_index",
    "spectral_angle_mapper",
    "weighted_mean_absolute_percentage_error",
    "word_error_rate",
    "word_information_lost",
    "word_information_preserved",
] + [
    "binary_confusion_matrix",
    "multiclass_confusion_matrix",
    "multilabel_confusion_matrix",
    "binary_stat_scores",
    "multiclass_stat_scores",
    "multilabel_stat_scores",
    "binary_f1_score",
    "binary_fbeta_score",
    "multiclass_f1_score",
    "multiclass_fbeta_score",
    "multilabel_f1_score",
    "multilabel_fbeta_score",
    "binary_cohen_kappa",
    "multiclass_cohen_kappa",
    "binary_jaccard_index",
    "multiclass_jaccard_index",
    "multilabel_jaccard_index",
    "binary_matthews_corrcoef",
    "multiclass_matthews_corrcoef",
    "multilabel_matthews_corrcoef",
    "multilabel_coverage_error",
    "multilabel_ranking_average_precision",
    "multilabel_ranking_loss",
    "binary_accuracy",
    "multilabel_accuracy",
    "multiclass_accuracy",
    "binary_specificity",
    "multiclass_specificity",
    "multilabel_specificity",
    "binary_hamming_distance",
    "multiclass_hamming_distance",
    "multilabel_hamming_distance",
    "binary_precision",
    "multiclass_precision",
    "multilabel_precision",
    "binary_recall",
    "multiclass_recall",
    "multilabel_recall",
    "multilabel_exact_match",
    "binary_auroc",
    "multiclass_auroc",
    "multilabel_auroc",
    "binary_average_precision",
    "multiclass_average_precision",
    "multilabel_average_precision",
    "binary_precision_recall_curve",
    "multiclass_precision_recall_curve",
    "multilabel_precision_recall_curve",
    "binary_recall_at_fixed_precision",
    "multiclass_recall_at_fixed_precision",
    "multilabel_recall_at_fixed_precision",
    "binary_roc",
    "multiclass_roc",
    "multilabel_roc",
    "binary_calibration_error",
    "multiclass_calibration_error",
    "binary_hinge_loss",
    "multiclass_hinge_loss",
]
