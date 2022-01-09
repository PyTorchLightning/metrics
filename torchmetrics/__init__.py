r"""Root package info."""
import logging as __logging
import os

from torchmetrics.__about__ import *  # noqa: F401, F403

_logger = __logging.getLogger("torchmetrics")
_logger.addHandler(__logging.StreamHandler())
_logger.setLevel(__logging.INFO)

_PACKAGE_ROOT = os.path.dirname(__file__)
_PROJECT_ROOT = os.path.dirname(_PACKAGE_ROOT)

from torchmetrics import functional  # noqa: E402
from torchmetrics.aggregation import CatMetric, MaxMetric, MeanMetric, MinMetric, SumMetric  # noqa: E402
from torchmetrics.audio import (  # noqa: E402
    PIT,
    SDR,
    SI_SDR,
    SI_SNR,
    SNR,
    ScaleInvariantSignalDistortionRatio,
    ScaleInvariantSignalNoiseRatio,
    SignalDistortionRatio,
    SignalNoiseRatio,
)
from torchmetrics.classification import (  # noqa: E402, F401
    AUC,
    AUROC,
    F1Score,
    ROC,
    Accuracy,
    AveragePrecision,
    BinnedAveragePrecision,
    BinnedPrecisionRecallCurve,
    BinnedRecallAtFixedPrecision,
    CalibrationError,
    CohenKappa,
    ConfusionMatrix,
    FBeta,
    HammingDistance,
    Hinge,
    IoU,
    JaccardIndex,
    KLDivergence,
    MatthewsCorrcoef,
    MatthewsCorrCoef,
    Precision,
    PrecisionRecallCurve,
    Recall,
    Specificity,
    StatScores,
)
from torchmetrics.image import PSNR, SSIM  # noqa: E402
from torchmetrics.metric import Metric  # noqa: E402
from torchmetrics.metric_collections import MetricCollection  # noqa: E402
from torchmetrics.regression import (  # noqa: E402
    CosineSimilarity,
    ExplainedVariance,
    MeanAbsoluteError,
    MeanAbsolutePercentageError,
    MeanSquaredError,
    MeanSquaredLogError,
    PearsonCorrcoef,
    PearsonCorrCoef,
    R2Score,
    SpearmanCorrcoef,
    SpearmanCorrCoef,
    SymmetricMeanAbsolutePercentageError,
    TweedieDevianceScore,
)
from torchmetrics.retrieval import (  # noqa: E402
    RetrievalFallOut,
    RetrievalHitRate,
    RetrievalMAP,
    RetrievalMRR,
    RetrievalNormalizedDCG,
    RetrievalPrecision,
    RetrievalRecall,
    RetrievalRPrecision,
)
from torchmetrics.text import (  # noqa: E402
    WER,
    BLEUScore,
    CharErrorRate,
    CHRFScore,
    MatchErrorRate,
    SacreBLEUScore,
    SQuAD,
    TranslationEditRate,
    WordErrorRate,
    WordInfoLost,
    WordInfoPreserved,
)
from torchmetrics.wrappers import BootStrapper, MetricTracker, MinMaxMetric, MultioutputWrapper  # noqa: E402

__all__ = [
    "functional",
    "Accuracy",
    "AUC",
    "AUROC",
    "AveragePrecision",
    "BinnedAveragePrecision",
    "BinnedPrecisionRecallCurve",
    "BinnedRecallAtFixedPrecision",
    "BLEUScore",
    "BootStrapper",
    "CalibrationError",
    "CatMetric",
    "CHRFScore",
    "CohenKappa",
    "ConfusionMatrix",
    "CosineSimilarity",
    "TweedieDevianceScore",
    "ExplainedVariance",
    "F1Score",
    "FBeta",
    "HammingDistance",
    "Hinge",
    "JaccardIndex",
    "KLDivergence",
    "MatthewsCorrcoef",
    "MatthewsCorrCoef",
    "MaxMetric",
    "MeanAbsoluteError",
    "MeanAbsolutePercentageError",
    "MeanMetric",
    "MeanSquaredError",
    "MeanSquaredLogError",
    "Metric",
    "MetricCollection",
    "MetricTracker",
    "MinMaxMetric",
    "MinMetric",
    "MultioutputWrapper",
    "PearsonCorrcoef",
    "PearsonCorrCoef",
    "PIT",
    "Precision",
    "PrecisionRecallCurve",
    "PSNR",
    "R2Score",
    "Recall",
    "RetrievalFallOut",
    "RetrievalHitRate",
    "RetrievalMAP",
    "RetrievalMRR",
    "RetrievalNormalizedDCG",
    "RetrievalPrecision",
    "RetrievalRecall",
    "RetrievalRPrecision",
    "ROC",
    "SacreBLEUScore",
    "SDR",
    "SignalDistortionRatio",
    "ScaleInvariantSignalDistortionRatio",
    "SI_SDR",
    "SI_SNR",
    "ScaleInvariantSignalNoiseRatio",
    "SignalNoiseRatio",
    "SNR",
    "SpearmanCorrcoef",
    "SpearmanCorrCoef",
    "Specificity",
    "SQuAD",
    "SSIM",
    "StatScores",
    "SumMetric",
    "SymmetricMeanAbsolutePercentageError",
    "TranslationEditRate",
    "WER",
    "WordErrorRate",
    "CharErrorRate",
    "MatchErrorRate",
    "WordInfoLost",
    "WordInfoPreserved",
]
