# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

**Note: we move fast, but still we preserve 0.1 version (one feature release) back compatibility.**

---

## [UnReleased] - 2024-MM-DD

### Added

- Added `ClusterAccuracy` metric to cluster package ([#2777](https://github.com/Lightning-AI/torchmetrics/pull/2777))


-


### Changed

- Make `num_classes` optional for classification in case of micro averaging ([#2841](https://github.com/PyTorchLightning/metrics/pull/2841))


### Removed

-


### Fixed

-


---

## [1.6.1] - 2024-12-24

### Changed

- Enabled specifying weights path for FID ([#2867](https://github.com/PyTorchLightning/metrics/pull/2867))
- Delete `Device2Host` caused by comm with device and host ([#2840](https://github.com/PyTorchLightning/metrics/pull/2840))

### Fixed

- Fixed plotting of multilabel confusion matrix ([#2858](https://github.com/PyTorchLightning/metrics/pull/2858))
- Fixed issue with shared state in metric collection when using dice score ([#2848](https://github.com/PyTorchLightning/metrics/pull/2848))
- Fixed `top_k` for `multiclassf1score` with one-hot encoding ([#2839](https://github.com/Lightning-AI/torchmetrics/issues/2839))
- Fixed slow calculations of classification metrics with MPS ([#2876](https://github.com/Lightning-AI/torchmetrics/issues/2876))


## [1.6.0] - 2024-11-12

### Added

- Added audio metric `NISQA` ([#2792](https://github.com/PyTorchLightning/metrics/pull/2792))
- Added classification metric `LogAUC` ([#2377](https://github.com/Lightning-AI/torchmetrics/pull/2377))
- Added classification metric `NegativePredictiveValue` ([#2433](https://github.com/Lightning-AI/torchmetrics/pull/2433))
- Added regression metric `NormalizedRootMeanSquaredError` ([#2442](https://github.com/Lightning-AI/torchmetrics/pull/2442))
- Added segmentation metric `Dice` ([#2725](https://github.com/Lightning-AI/torchmetrics/pull/2725))
- Added method `merge_state` to `Metric` ([#2786](https://github.com/Lightning-AI/torchmetrics/pull/2786))
- Added support for propagation of the autograd graph in ddp setting ([#2754](https://github.com/Lightning-AI/torchmetrics/pull/2754))

### Changed

- Changed naming and input order arguments in `KLDivergence` ([#2800](https://github.com/Lightning-AI/torchmetrics/pull/2800))

### Deprecated

- Deprecated Dice from classification metrics ([#2725](https://github.com/Lightning-AI/torchmetrics/pull/2725))

### Removed

- Changed minimum supported Pytorch version to 2.0 ([#2671](https://github.com/Lightning-AI/torchmetrics/pull/2671))
- Dropped support for Python 3.8 ([#2827](https://github.com/Lightning-AI/torchmetrics/pull/2827))
- Removed `num_outputs` in `R2Score` ([#2800](https://github.com/Lightning-AI/torchmetrics/pull/2800))

### Fixed

- Fixed segmentation `Dice` + `GeneralizedDice` for 2d index tensors ([#2832](https://github.com/Lightning-AI/torchmetrics/pull/2832))
- Fixed mixed results of `rouge_score` with `accumulate='best'` ([#2830](https://github.com/Lightning-AI/torchmetrics/pull/2830))

---

## [1.5.2] - 2024-11-07

### Changed

- Re-adding `numpy` 2+ support ([#2804](https://github.com/Lightning-AI/torchmetrics/pull/2804))

### Fixed

- Fixed iou scores in detection for either empty predictions/targets leading to wrong scores ([#2805](https://github.com/Lightning-AI/torchmetrics/pull/2805))
- Fixed `MetricCollection` compatibility with `torch.jit.script` ([#2813](https://github.com/Lightning-AI/torchmetrics/pull/2813))
- Fixed assert in PIT ([#2811](https://github.com/Lightning-AI/torchmetrics/pull/2811))
- Patched `np.Inf` for `numpy` 2.0+ ([#2826](https://github.com/Lightning-AI/torchmetrics/pull/2826))


## [1.5.1] - 2024-10-22

### Fixed

- Changing `_modules` dict type in Pytorch 2.5 preventing to fail collections metrics ([#2793](https://github.com/Lightning-AI/torchmetrics/pull/2793))


## [1.5.0] - 2024-10-18

### Added

- Added segmentation metric `HausdorffDistance` ([#2122](https://github.com/Lightning-AI/torchmetrics/pull/2122))
- Added audio metric `DNSMOS` ([#2525](https://github.com/PyTorchLightning/metrics/pull/2525))
- Added shape metric `ProcrustesDistance` ([#2723](https://github.com/Lightning-AI/torchmetrics/pull/2723)
- Added `MetricInputTransformer` wrapper ([#2392](https://github.com/Lightning-AI/torchmetrics/pull/2392))
- Added `input_format` argument to segmentation metrics ([#2572](https://github.com/Lightning-AI/torchmetrics/pull/2572))
- Added `multi-output` support for MAE metric ([#2605](https://github.com/Lightning-AI/torchmetrics/pull/2605))
- Added `truncation` argument to `BERTScore` ([#2776](https://github.com/Lightning-AI/torchmetrics/pull/2776))

### Changed

- Tracker higher is better integration ([#2649](https://github.com/Lightning-AI/torchmetrics/pull/2649))
- Updated `InfoLM` class to dynamically set `higher_is_better` ([#2674](https://github.com/Lightning-AI/torchmetrics/pull/2674))

### Deprecated

- Deprecated `num_outputs` in `R2Score` ([#2705](https://github.com/Lightning-AI/torchmetrics/pull/2705))

### Fixed

- Fixed corner case in `IoU` metric for single empty prediction tensors ([#2780](https://github.com/Lightning-AI/torchmetrics/pull/2780))
- Fixed `PSNR` calculation for integer type input images ([#2788](https://github.com/Lightning-AI/torchmetrics/pull/2788))

---

## [1.4.3] - 2024-10-10

### Fixed

- Fixed for Pearson changes inputs ([#2765](https://github.com/Lightning-AI/torchmetrics/pull/2765))
- Fixed bug in `PESQ` metric where `NoUtterancesError` prevented calculating on a batch of data ([#2753](https://github.com/Lightning-AI/torchmetrics/pull/2753))
- Fixed corner case in `MatthewsCorrCoef` ([#2743](https://github.com/Lightning-AI/torchmetrics/pull/2743))


## [1.4.2] - 2022-09-12

### Added

- Re-adding `Chrf` implementation ([#2701](https://github.com/Lightning-AI/torchmetrics/pull/2701))

### Fixed

- Fixed wrong aggregation in `segmentation.MeanIoU` ([#2698](https://github.com/Lightning-AI/torchmetrics/pull/2698))
- Fixed handling zero division error in binary IoU (Jaccard index) calculation ([#2726](https://github.com/Lightning-AI/torchmetrics/pull/2726))
- Corrected the padding related calculation errors in SSIM ([#2721](https://github.com/Lightning-AI/torchmetrics/pull/2721))
- Fixed compatibility of audio domain with new `scipy` ([#2733](https://github.com/Lightning-AI/torchmetrics/pull/2733))
- Fixed how `prefix`/`postfix` works in `MultitaskWrapper` ([#2722](https://github.com/Lightning-AI/torchmetrics/pull/2722))
- Fixed flakiness in tests related to `torch.unique` with `dim=None` ([#2650](https://github.com/Lightning-AI/torchmetrics/pull/2650))


## [1.4.1] - 2024-08-02

### Changed

- Calculate text color of `ConfusionMatrix` plot based on luminance ([#2590](https://github.com/Lightning-AI/torchmetrics/pull/2590))
- Updated `_safe_divide` to allow `Accuracy` to run on the GPU ([#2640](https://github.com/Lightning-AI/torchmetrics/pull/2640))
- Improved error messages for intersection detection metrics for wrong user input ([#2577](https://github.com/Lightning-AI/torchmetrics/pull/2577))

### Removed

- Dropped `Chrf` implementation due to licensing issues with the upstream package ([#2668](https://github.com/Lightning-AI/torchmetrics/pull/2668))

### Fixed

- Fixed bug in `MetricCollection` when using compute groups and `compute` is called more than once ([#2571](https://github.com/Lightning-AI/torchmetrics/pull/2571))
- Fixed class order of `panoptic_quality(..., return_per_class=True)` output ([#2548](https://github.com/Lightning-AI/torchmetrics/pull/2548))
- Fixed `BootstrapWrapper` not being reset correctly ([#2574](https://github.com/Lightning-AI/torchmetrics/pull/2574))
- Fixed integration between `ClasswiseWrapper` and `MetricCollection` with custom `_filter_kwargs` method ([#2575](https://github.com/Lightning-AI/torchmetrics/pull/2575))
- Fixed BertScore calculation: pred target misalignment ([#2347](https://github.com/Lightning-AI/torchmetrics/pull/2347))
- Fixed `_cumsum` helper function in multi-gpu ([#2636](https://github.com/Lightning-AI/torchmetrics/pull/2636))
- Fixed bug in `MeanAveragePrecision.coco_to_tm` ([#2588](https://github.com/Lightning-AI/torchmetrics/pull/2588))
- Fixed missed f-strings in exceptions/warnings ([#2667](https://github.com/Lightning-AI/torchmetrics/pull/2667))


## [1.4.0] - 2024-05-03

### Added

- Added `SensitivityAtSpecificity` metric to classification subpackage ([#2217](https://github.com/Lightning-AI/torchmetrics/pull/2217))
- Added `QualityWithNoReference` metric to image subpackage ([#2288](https://github.com/Lightning-AI/torchmetrics/pull/2288))
- Added a new segmentation metric:
    - `MeanIoU` ([#1236](https://github.com/PyTorchLightning/metrics/pull/1236))
    - `GeneralizedDiceScore` ([#1090](https://github.com/Lightning-AI/metrics/pull/1090))
- Added support for calculating segmentation quality and recognition quality in `PanopticQuality` metric ([#2381](https://github.com/Lightning-AI/torchmetrics/pull/2381))
- Added `pretty-errors` for improving error prints ([#2431](https://github.com/Lightning-AI/torchmetrics/pull/2431))
- Added support for `torch.float` weighted networks for FID and KID calculations ([#2483](https://github.com/Lightning-AI/torchmetrics/pull/2483))
- Added `zero_division` argument to selected classification metrics ([#2198](https://github.com/Lightning-AI/torchmetrics/pull/2198))

### Changed

- Made `__getattr__` and `__setattr__` of `ClasswiseWrapper` more general ([#2424](https://github.com/Lightning-AI/torchmetrics/pull/2424))

### Fixed

- Fix getitem for metric collection when prefix/postfix is set ([#2430](https://github.com/Lightning-AI/torchmetrics/pull/2430))
- Fixed axis names with Precision-Recall curve ([#2462](https://github.com/Lightning-AI/torchmetrics/pull/2462))
- Fixed list synchronization with partly empty lists ([#2468](https://github.com/Lightning-AI/torchmetrics/pull/2468))
- Fixed memory leak in metrics using list states ([#2492](https://github.com/Lightning-AI/torchmetrics/pull/2492))
- Fixed bug in computation of `ERGAS` metric ([#2498](https://github.com/Lightning-AI/torchmetrics/pull/2498))
- Fixed `BootStrapper` wrapper not working with `kwargs` provided argument ([#2503](https://github.com/Lightning-AI/torchmetrics/pull/2503))
- Fixed warnings being suppressed in `MeanAveragePrecision` when requested ([#2501](https://github.com/Lightning-AI/torchmetrics/pull/2501))
- Fixed corner-case in `binary_average_precision` when only negative samples are provided ([#2507](https://github.com/Lightning-AI/torchmetrics/pull/2507))

---

## [1.3.2] - 2024-03-18

### Fixed

- Fixed negative variance estimates in certain image metrics ([#2378](https://github.com/Lightning-AI/torchmetrics/pull/2378))
- Fixed dtype being changed by deepspeed for certain regression metrics ([#2379](https://github.com/Lightning-AI/torchmetrics/pull/2379))
- Fixed plotting of metric collection when prefix/postfix is set ([#2429](https://github.com/Lightning-AI/torchmetrics/pull/2429))
- Fixed bug when `top_k>1` and `average="macro"` for classification metrics ([#2423](https://github.com/Lightning-AI/torchmetrics/pull/2423))
- Fixed case where label prediction tensors in classification metrics were not validated correctly ([#2427](https://github.com/Lightning-AI/torchmetrics/pull/2427))
- Fixed how auc scores are calculated in `PrecisionRecallCurve.plot` methods ([#2437](https://github.com/Lightning-AI/torchmetrics/pull/2437))


## [1.3.1] - 2024-02-12

### Fixed

- Fixed how backprop is handled in `LPIPS` metric ([#2326](https://github.com/Lightning-AI/torchmetrics/pull/2326))
- Fixed `MultitaskWrapper` not being able to be logged in lightning when using metric collections ([#2349](https://github.com/Lightning-AI/torchmetrics/pull/2349))
- Fixed high memory consumption in `Perplexity` metric ([#2346](https://github.com/Lightning-AI/torchmetrics/pull/2346))
- Fixed cached network in `FeatureShare` not being moved to the correct device ([#2348](https://github.com/Lightning-AI/torchmetrics/pull/2348))
- Fix naming of statistics in `MeanAveragePrecision` with custom max det thresholds ([#2367](https://github.com/Lightning-AI/torchmetrics/pull/2367))
- Fixed custom aggregation in retrieval metrics ([#2364](https://github.com/Lightning-AI/torchmetrics/pull/2364))
- Fixed initialize aggregation metrics with default floating type ([#2366](https://github.com/Lightning-AI/torchmetrics/pull/2366))
- Fixed plotting of confusion matrices ([#2358](https://github.com/Lightning-AI/torchmetrics/pull/2358))


## [1.3.0] - 2024-01-10

### Added

- Added more tokenizers for `SacreBLEU` metric ([#2068](https://github.com/Lightning-AI/torchmetrics/pull/2068))
- Added support for logging `MultiTaskWrapper` directly with lightnings `log_dict` method ([#2213](https://github.com/Lightning-AI/torchmetrics/pull/2213))
- Added `FeatureShare` wrapper to share submodules containing feature extractors between metrics ([#2120](https://github.com/Lightning-AI/torchmetrics/pull/2120))
- Added new metrics to image domain:
  * `SpatialDistortionIndex` ([#2260](https://github.com/Lightning-AI/torchmetrics/pull/2260))
  * Added `CriticalSuccessIndex` ([#2257](https://github.com/Lightning-AI/torchmetrics/pull/2257))
  * `Spatial Correlation Coefficient` ([#2248](https://github.com/Lightning-AI/torchmetrics/pull/2248))
- Added `average` argument to multiclass versions of `PrecisionRecallCurve` and `ROC` ([#2084](https://github.com/Lightning-AI/torchmetrics/pull/2084))
- Added confidence scores when `extended_summary=True` in `MeanAveragePrecision` ([#2212](https://github.com/Lightning-AI/torchmetrics/pull/2212))
- Added `RetrievalAUROC` metric ([#2251](https://github.com/Lightning-AI/torchmetrics/pull/2251))
- Added `aggregate` argument to retrieval metrics ([#2220](https://github.com/Lightning-AI/torchmetrics/pull/2220))
- Added utility functions in `segmentation.utils` for future segmentation metrics ([#2105](https://github.com/Lightning-AI/torchmetrics/pull/2105))


### Changed

- Changed minimum supported Pytorch version from 1.8 to 1.10 ([#2145](https://github.com/Lightning-AI/torchmetrics/pull/2145))
- Changed x-/y-axis order for `PrecisionRecallCurve` to be consistent with scikit-learn ([#2183](https://github.com/Lightning-AI/torchmetrics/pull/2183))

### Deprecated

- Deprecated `metric._update_called` ([#2141](https://github.com/Lightning-AI/torchmetrics/pull/2141))
- Deprecated `specicity_at_sensitivity` in favour of `specificity_at_sensitivity` ([#2199](https://github.com/Lightning-AI/torchmetrics/pull/2199))

### Fixed

- Fixed support for half precision + CPU in metrics requiring topk operator ([#2252](https://github.com/Lightning-AI/torchmetrics/pull/2252))
- Fixed warning incorrectly being raised in `Running` metrics ([#2256](https://github.com/Lightning-AI/torchmetrics/pull/2265))
- Fixed integration with custom feature extractor in `FID` metric ([#2277](https://github.com/Lightning-AI/torchmetrics/pull/2277))

---

## [1.2.1] - 2023-11-30

### Added

- Added error if `NoTrainInceptionV3` is being initialized without `torch-fidelity` not being installed ([#2143](https://github.com/Lightning-AI/torchmetrics/pull/2143))
- Added support for Pytorch v2.1 ([#2142](https://github.com/Lightning-AI/torchmetrics/pull/2142))

### Changed

- Change default state of `SpectralAngleMapper` and `UniversalImageQualityIndex` to be tensors ([#2089](https://github.com/Lightning-AI/torchmetrics/pull/2089))
- Use `torch` range func and repeat for deterministic bincount ([#2184](https://github.com/Lightning-AI/torchmetrics/pull/2184))

### Removed

- Removed unused `lpips` third-party package as dependency of `LearnedPerceptualImagePatchSimilarity` metric ([#2230](https://github.com/Lightning-AI/torchmetrics/pull/2230))

### Fixed

- Fixed numerical stability bug in `LearnedPerceptualImagePatchSimilarity` metric ([#2144](https://github.com/Lightning-AI/torchmetrics/pull/2144))
- Fixed numerical stability issue in `UniversalImageQualityIndex` metric ([#2222](https://github.com/Lightning-AI/torchmetrics/pull/2222))
- Fixed incompatibility for `MeanAveragePrecision` with `pycocotools` backend when too little `max_detection_thresholds` are provided ([#2219](https://github.com/Lightning-AI/torchmetrics/pull/2219))
- Fixed support for half precision in Perplexity metric ([#2235](https://github.com/Lightning-AI/torchmetrics/pull/2235))
- Fixed device and dtype for `LearnedPerceptualImagePatchSimilarity` functional metric ([#2234](https://github.com/Lightning-AI/torchmetrics/pull/2234))
- Fixed bug in `Metric._reduce_states(...)` when using `dist_sync_fn="cat"` ([#2226](https://github.com/Lightning-AI/torchmetrics/pull/2226))
- Fixed bug in `CosineSimilarity` where 2d is expected but 1d input was given ([#2241](https://github.com/Lightning-AI/torchmetrics/pull/2241))
- Fixed bug in `MetricCollection` when using compute groups and `compute` is called more than once ([#2211](https://github.com/Lightning-AI/torchmetrics/pull/2211))


## [1.2.0] - 2023-09-22

### Added

- Added metric to cluster package:
    - `MutualInformationScore` ([#2008](https://github.com/Lightning-AI/torchmetrics/pull/2008))
    - `RandScore` ([#2025](https://github.com/Lightning-AI/torchmetrics/pull/2025))
    - `NormalizedMutualInfoScore` ([#2029](https://github.com/Lightning-AI/torchmetrics/pull/2029))
    - `AdjustedRandScore` ([#2032](https://github.com/Lightning-AI/torchmetrics/pull/2032))
    - `CalinskiHarabaszScore` ([#2036](https://github.com/Lightning-AI/torchmetrics/pull/2036))
    - `DunnIndex` ([#2049](https://github.com/Lightning-AI/torchmetrics/pull/2049))
    - `HomogeneityScore` ([#2053](https://github.com/Lightning-AI/torchmetrics/pull/2053))
    - `CompletenessScore` ([#2053](https://github.com/Lightning-AI/torchmetrics/pull/2053))
    - `VMeasureScore` ([#2053](https://github.com/Lightning-AI/torchmetrics/pull/2053))
    - `FowlkesMallowsIndex` ([#2066](https://github.com/Lightning-AI/torchmetrics/pull/2066))
    - `AdjustedMutualInfoScore` ([#2058](https://github.com/Lightning-AI/torchmetrics/pull/2058))
    - `DaviesBouldinScore` ([#2071](https://github.com/Lightning-AI/torchmetrics/pull/2071))
- Added `backend` argument to `MeanAveragePrecision` ([#2034](https://github.com/Lightning-AI/torchmetrics/pull/2034))

---

## [1.1.2] - 2023-09-11

### Fixed

- Fixed tie breaking in ndcg metric ([#2031](https://github.com/Lightning-AI/torchmetrics/pull/2031))
- Fixed bug in `BootStrapper` when very few samples were evaluated that could lead to crash ([#2052](https://github.com/Lightning-AI/torchmetrics/pull/2052))
- Fixed bug when creating multiple plots that lead to not all plots being shown ([#2060](https://github.com/Lightning-AI/torchmetrics/pull/2060))
- Fixed performance issues in `RecallAtFixedPrecision` for large batch sizes ([#2042](https://github.com/Lightning-AI/torchmetrics/pull/2042))
- Fixed bug related to `MetricCollection` used with custom metrics have `prefix`/`postfix` attributes ([#2070](https://github.com/Lightning-AI/torchmetrics/pull/2070))


## [1.1.1] - 2023-08-29

### Added

- Added `average` argument to `MeanAveragePrecision` ([#2018](https://github.com/Lightning-AI/torchmetrics/pull/2018))

### Fixed

- Fixed bug in `PearsonCorrCoef` is updated on single samples at a time ([#2019](https://github.com/Lightning-AI/torchmetrics/pull/2019))
- Fixed support for pixel-wise MSE ([#2017](https://github.com/Lightning-AI/torchmetrics/pull/2017))
- Fixed bug in `MetricCollection` when used with multiple metrics that return dicts with same keys ([#2027](https://github.com/Lightning-AI/torchmetrics/pull/2027))
- Fixed bug in detection intersection metrics when `class_metrics=True` resulting in wrong values ([#1924](https://github.com/Lightning-AI/torchmetrics/pull/1924))
- Fixed missing attributes `higher_is_better`, `is_differentiable` for some metrics ([#2028](https://github.com/Lightning-AI/torchmetrics/pull/2028))


## [1.1.0] - 2023-08-22

### Added

- Added source aggregated signal-to-distortion ratio (SA-SDR) metric ([#1882](https://github.com/Lightning-AI/torchmetrics/pull/1882)
- Added `VisualInformationFidelity` to image package ([#1830](https://github.com/Lightning-AI/torchmetrics/pull/1830))
- Added `EditDistance` to text package ([#1906](https://github.com/Lightning-AI/torchmetrics/pull/1906))
- Added `top_k` argument to `RetrievalMRR` in retrieval package ([#1961](https://github.com/Lightning-AI/torchmetrics/pull/1961))
- Added support for evaluating `"segm"` and `"bbox"` detection in `MeanAveragePrecision` at the same time ([#1928](https://github.com/Lightning-AI/torchmetrics/pull/1928))
- Added `PerceptualPathLength` to image package ([#1939](https://github.com/Lightning-AI/torchmetrics/pull/1939))
- Added support for multioutput evaluation in `MeanSquaredError` ([#1937](https://github.com/Lightning-AI/torchmetrics/pull/1937))
- Added argument `extended_summary` to `MeanAveragePrecision` such that precision, recall, iou can be easily returned ([#1983](https://github.com/Lightning-AI/torchmetrics/pull/1983))
- Added warning to `ClipScore` if long captions are detected and truncate ([#2001](https://github.com/Lightning-AI/torchmetrics/pull/2001))
- Added `CLIPImageQualityAssessment` to multimodal package ([#1931](https://github.com/Lightning-AI/torchmetrics/pull/1931))
- Added new property `metric_state` to all metrics for users to investigate currently stored tensors in memory ([#2006](https://github.com/Lightning-AI/torchmetrics/pull/2006))

---

## [1.0.3] - 2023-08-08

### Added

- Added warning to `MeanAveragePrecision` if too many detections are observed ([#1978](https://github.com/Lightning-AI/torchmetrics/pull/1978))

### Fixed

- Fix support for int input for when `multidim_average="samplewise"` in classification metrics  ([#1977](https://github.com/Lightning-AI/torchmetrics/pull/1977))
- Fixed x/y labels when plotting confusion matrices ([#1976](https://github.com/Lightning-AI/torchmetrics/pull/1976))
- Fixed IOU compute in cuda ([#1982](https://github.com/Lightning-AI/torchmetrics/pull/1982))


## [1.0.2] - 2023-08-02

### Added

- Added warning to `PearsonCorrCoeff` if input has a very small variance for its given dtype ([#1926](https://github.com/Lightning-AI/torchmetrics/pull/1926))

### Changed

- Changed all non-task specific classification metrics to be true subtypes of `Metric` ([#1963](https://github.com/Lightning-AI/torchmetrics/pull/1963))

### Fixed

- Fixed bug in `CalibrationError` where calculations for double precision input was performed in float precision ([#1919](https://github.com/Lightning-AI/torchmetrics/pull/1919))
- Fixed bug related to the `prefix/postfix` arguments in `MetricCollection` and `ClasswiseWrapper` being duplicated ([#1918](https://github.com/Lightning-AI/torchmetrics/pull/1918))
- Fixed missing AUC score when plotting classification metrics that support the `score` argument ([#1948](https://github.com/Lightning-AI/torchmetrics/pull/1948))


## [1.0.1] - 2023-07-13

### Fixed
- Fixes corner case when using `MetricCollection` together with aggregation metrics ([#1896](https://github.com/Lightning-AI/torchmetrics/pull/1896))
- Fixed the use of `max_fpr` in `AUROC` metric when only one class is present ([#1895](https://github.com/Lightning-AI/torchmetrics/pull/1895))
- Fixed bug related to empty predictions for `IntersectionOverUnion` metric ([#1892](https://github.com/Lightning-AI/torchmetrics/pull/1892))
- Fixed bug related to `MeanMetric` and broadcasting of weights when Nans are present ([#1898](https://github.com/Lightning-AI/torchmetrics/pull/1898))
- Fixed bug related to expected input format of pycoco in `MeanAveragePrecision` ([#1913](https://github.com/Lightning-AI/torchmetrics/pull/1913))


## [1.0.0] - 2023-07-04

### Added

- Added `prefix` and `postfix` arguments to `ClasswiseWrapper` ([#1866](https://github.com/Lightning-AI/torchmetrics/pull/1866))
- Added speech-to-reverberation modulation energy ratio (SRMR) metric ([#1792](https://github.com/Lightning-AI/torchmetrics/pull/1792), [#1872](https://github.com/Lightning-AI/torchmetrics/pull/1872))
- Added new global arg `compute_with_cache` to control caching behaviour after `compute` method ([#1754](https://github.com/Lightning-AI/torchmetrics/pull/1754))
- Added `ComplexScaleInvariantSignalNoiseRatio` for audio package ([#1785](https://github.com/Lightning-AI/torchmetrics/pull/1785))
- Added `Running` wrapper for calculate running statistics ([#1752](https://github.com/Lightning-AI/torchmetrics/pull/1752))
- Added`RelativeAverageSpectralError` and `RootMeanSquaredErrorUsingSlidingWindow` to image package ([#816](https://github.com/PyTorchLightning/metrics/pull/816))
- Added support for `SpecificityAtSensitivity` Metric ([#1432](https://github.com/Lightning-AI/metrics/pull/1432))
- Added support for plotting of metrics through `.plot()` method (
    [#1328](https://github.com/Lightning-AI/metrics/pull/1328),
    [#1481](https://github.com/Lightning-AI/metrics/pull/1481),
    [#1480](https://github.com/Lightning-AI/metrics/pull/1480),
    [#1490](https://github.com/Lightning-AI/metrics/pull/1490),
    [#1581](https://github.com/Lightning-AI/metrics/pull/1581),
    [#1585](https://github.com/Lightning-AI/metrics/pull/1585),
    [#1593](https://github.com/Lightning-AI/metrics/pull/1593),
    [#1600](https://github.com/Lightning-AI/metrics/pull/1600),
    [#1605](https://github.com/Lightning-AI/metrics/pull/1605),
    [#1610](https://github.com/Lightning-AI/metrics/pull/1610),
    [#1609](https://github.com/Lightning-AI/metrics/pull/1609),
    [#1621](https://github.com/Lightning-AI/metrics/pull/1621),
    [#1624](https://github.com/Lightning-AI/metrics/pull/1624),
    [#1623](https://github.com/Lightning-AI/metrics/pull/1623),
    [#1638](https://github.com/Lightning-AI/metrics/pull/1638),
    [#1631](https://github.com/Lightning-AI/metrics/pull/1631),
    [#1650](https://github.com/Lightning-AI/metrics/pull/1650),
    [#1639](https://github.com/Lightning-AI/metrics/pull/1639),
    [#1660](https://github.com/Lightning-AI/metrics/pull/1660),
    [#1682](https://github.com/Lightning-AI/torchmetrics/pull/1682),
    [#1786](https://github.com/Lightning-AI/torchmetrics/pull/1786),
)
- Added support for plotting of audio metrics through `.plot()` method ([#1434](https://github.com/Lightning-AI/metrics/pull/1434))
- Added `classes` to output from `MAP` metric ([#1419](https://github.com/Lightning-AI/metrics/pull/1419))
- Added Binary group fairness metrics to classification package ([#1404](https://github.com/Lightning-AI/metrics/pull/1404))
- Added `MinkowskiDistance` to regression package ([#1362](https://github.com/Lightning-AI/metrics/pull/1362))
- Added `pairwise_minkowski_distance` to pairwise package ([#1362](https://github.com/Lightning-AI/metrics/pull/1362))
- Added new detection metric `PanopticQuality` (
    [#929](https://github.com/PyTorchLightning/metrics/pull/929),
    [#1527](https://github.com/PyTorchLightning/metrics/pull/1527),
)
- Added `PSNRB` metric ([#1421](https://github.com/Lightning-AI/metrics/pull/1421))
- Added `ClassificationTask` Enum and use in metrics ([#1479](https://github.com/Lightning-AI/metrics/pull/1479))
- Added `ignore_index` option to `exact_match` metric ([#1540](https://github.com/Lightning-AI/metrics/pull/1540))
- Add parameter `top_k` to `RetrievalMAP` ([#1501](https://github.com/Lightning-AI/metrics/pull/1501))
- Added support for deterministic evaluation on GPU for metrics that uses `torch.cumsum` operator ([#1499](https://github.com/Lightning-AI/metrics/pull/1499))
- Added support for plotting of aggregation metrics through `.plot()` method ([#1485](https://github.com/Lightning-AI/metrics/pull/1485))
- Added support for python 3.11 ([#1612](https://github.com/Lightning-AI/metrics/pull/1612))
- Added support for auto clamping of input for metrics that uses the `data_range` ([#1606](argument https://github.com/Lightning-AI/metrics/pull/1606))
- Added `ModifiedPanopticQuality` metric to detection package ([#1627](https://github.com/Lightning-AI/metrics/pull/1627))
- Added `PrecisionAtFixedRecall` metric to classification package ([#1683](https://github.com/Lightning-AI/torchmetrics/pull/1683))
- Added multiple metrics to detection package ([#1284](https://github.com/Lightning-AI/metrics/pull/1284))
  * `IntersectionOverUnion`
  * `GeneralizedIntersectionOverUnion`
  * `CompleteIntersectionOverUnion`
  * `DistanceIntersectionOverUnion`
- Added `MultitaskWrapper` to wrapper package ([#1762](https://github.com/Lightning-AI/torchmetrics/pull/1762))
- Added `RelativeSquaredError` metric to regression package ([#1765](https://github.com/Lightning-AI/torchmetrics/pull/1765))
- Added `MemorizationInformedFrechetInceptionDistance` metric to image package ([#1580](https://github.com/Lightning-AI/torchmetrics/pull/1580))


### Changed

- Changed `permutation_invariant_training` to allow using a `'permutation-wise'` metric function ([#1794](https://github.com/Lightning-AI/metrics/pull/1794))
- Changed `update_count` and `update_called` from private to public methods ([#1370](https://github.com/Lightning-AI/metrics/pull/1370))
- Raise exception for invalid kwargs in Metric base class ([#1427](https://github.com/Lightning-AI/metrics/pull/1427))
- Extend `EnumStr` raising `ValueError` for invalid value ([#1479](https://github.com/Lightning-AI/metrics/pull/1479))
- Improve speed and memory consumption of binned `PrecisionRecallCurve` with large number of samples ([#1493](https://github.com/Lightning-AI/metrics/pull/1493))
- Changed `__iter__` method from raising `NotImplementedError` to `TypeError` by setting to `None` ([#1538](https://github.com/Lightning-AI/metrics/pull/1538))
- `FID` metric will now raise an error if too few samples are provided ([#1655](https://github.com/Lightning-AI/metrics/pull/1655))
- Allowed FID with `torch.float64` ([#1628](https://github.com/Lightning-AI/metrics/pull/1628))
- Changed `LPIPS` implementation to no more rely on third-party package ([#1575](https://github.com/Lightning-AI/metrics/pull/1575))
- Changed FID matrix square root calculation from `scipy` to `torch` ([#1708](https://github.com/Lightning-AI/torchmetrics/pull/1708))
- Changed calculation in `PearsonCorrCoeff` to be more robust in certain cases  ([#1729](https://github.com/Lightning-AI/torchmetrics/pull/1729))
- Changed `MeanAveragePrecision` to `pycocotools` backend ([#1832](https://github.com/Lightning-AI/torchmetrics/pull/1832))


### Deprecated

- Deprecated domain metrics import from package root (
    [#1685](https://github.com/Lightning-AI/metrics/pull/1685),
    [#1694](https://github.com/Lightning-AI/metrics/pull/1694),
    [#1696](https://github.com/Lightning-AI/metrics/pull/1696),
    [#1699](https://github.com/Lightning-AI/metrics/pull/1699),
    [#1703](https://github.com/Lightning-AI/metrics/pull/1703),
)


### Removed

- Support for python 3.7 ([#1640](https://github.com/Lightning-AI/metrics/pull/1640))


### Fixed

- Fixed support in `MetricTracker` for `MultioutputWrapper` and nested structures ([#1608](https://github.com/Lightning-AI/metrics/pull/1608))
- Fixed restrictive check in `PearsonCorrCoef` ([#1649](https://github.com/Lightning-AI/metrics/pull/1649))
- Fixed integration with `jsonargparse` and `LightningCLI` ([#1651](https://github.com/Lightning-AI/metrics/pull/1651))
- Fixed corner case in calibration error for zero confidence input ([#1648](https://github.com/Lightning-AI/metrics/pull/1648))
- Fix precision-recall curve based computations for float target ([#1642](https://github.com/Lightning-AI/metrics/pull/1642))
- Fixed missing kwarg squeeze in `MultiOutputWrapper` ([#1675](https://github.com/Lightning-AI/torchmetrics/pull/1675))
- Fixed padding removal for 3d input in `MSSSIM` ([#1674](https://github.com/Lightning-AI/torchmetrics/pull/1674))
- Fixed `max_det_threshold` in MAP detection ([#1712](https://github.com/Lightning-AI/torchmetrics/pull/1712))
- Fixed states being saved in metrics that use `register_buffer` ([#1728](https://github.com/Lightning-AI/torchmetrics/pull/1728))
- Fixed states not being correctly synced and device transferred in `MeanAveragePrecision` for `iou_type="segm"` ([#1763](https://github.com/Lightning-AI/torchmetrics/pull/1763))
- Fixed use of `prefix` and `postfix` in nested `MetricCollection` ([#1773](https://github.com/Lightning-AI/torchmetrics/pull/1773))
- Fixed `ax` plotting logging in `MetricCollection ([#1783](https://github.com/Lightning-AI/torchmetrics/pull/1783))
- Fixed lookup for punkt sources being downloaded in `RougeScore` ([#1789](https://github.com/Lightning-AI/torchmetrics/pull/1789))
- Fixed integration with lightning for `CompositionalMetric` ([#1761](https://github.com/Lightning-AI/torchmetrics/pull/1761))
- Fixed several bugs in `SpectralDistortionIndex` metric ([#1808](https://github.com/Lightning-AI/torchmetrics/pull/1808))
- Fixed bug for corner cases in `MatthewsCorrCoef` (
    [#1812](https://github.com/Lightning-AI/torchmetrics/pull/1812),
    [#1863](https://github.com/Lightning-AI/torchmetrics/pull/1863)
)
- Fixed support for half precision in `PearsonCorrCoef` ([#1819](https://github.com/Lightning-AI/torchmetrics/pull/1819))
- Fixed number of bugs related to `average="macro"` in classification metrics ([#1821](https://github.com/Lightning-AI/torchmetrics/pull/1821))
- Fixed off-by-one issue when `ignore_index = num_classes + 1` in Multiclass-jaccard ([#1860](https://github.com/Lightning-AI/torchmetrics/pull/1860))

---

## [0.11.4] - 2023-03-10

### Fixed

- Fixed evaluation of `R2Score` with near constant target ([#1576](https://github.com/Lightning-AI/metrics/pull/1576))
- Fixed dtype conversion when metric is submodule ([#1583](https://github.com/Lightning-AI/metrics/pull/1583))
- Fixed bug related to `top_k>1` and `ignore_index!=None` in `StatScores` based metrics ([#1589](https://github.com/Lightning-AI/metrics/pull/1589))
- Fixed corner case for `PearsonCorrCoef` when running in ddp mode but only on single device ([#1587](https://github.com/Lightning-AI/metrics/pull/1587))
- Fixed overflow error for specific cases in `MAP` when big areas are calculated ([#1607](https://github.com/Lightning-AI/metrics/pull/1607))


## [0.11.3] - 2023-02-28

### Fixed

- Fixed classification metrics for `byte` input ([#1521](https://github.com/Lightning-AI/metrics/pull/1474))
- Fixed the use of `ignore_index` in `MulticlassJaccardIndex` ([#1386](https://github.com/Lightning-AI/metrics/pull/1386))


## [0.11.2] - 2023-02-21

### Fixed

- Fixed compatibility between XLA in `_bincount` function ([#1471](https://github.com/Lightning-AI/metrics/pull/1471))
- Fixed type hints in methods belonging to `MetricTracker` wrapper ([#1472](https://github.com/Lightning-AI/metrics/pull/1472))
- Fixed `multilabel` in `ExactMatch` ([#1474](https://github.com/Lightning-AI/metrics/pull/1474))


## [0.11.1] - 2023-01-30

### Fixed

- Fixed type checking on the `maximize` parameter at the initialization of `MetricTracker` ([#1428](https://github.com/Lightning-AI/metrics/issues/1428))
- Fixed mixed precision autocast for `SSIM` metric ([#1454](https://github.com/Lightning-AI/metrics/pull/1454))
- Fixed checking for `nltk.punkt` in `RougeScore` if a machine is not online ([#1456](https://github.com/Lightning-AI/metrics/pull/1456))
- Fixed wrongly reset method in `MultioutputWrapper` ([#1460](https://github.com/Lightning-AI/metrics/issues/1460))
- Fixed dtype checking in `PrecisionRecallCurve` for `target` tensor ([#1457](https://github.com/Lightning-AI/metrics/pull/1457))


## [0.11.0] - 2022-11-30

### Added

- Added `MulticlassExactMatch` to classification metrics ([#1343](https://github.com/Lightning-AI/metrics/pull/1343))
- Added `TotalVariation` to image package ([#978](https://github.com/Lightning-AI/metrics/pull/978))
- Added `CLIPScore` to new multimodal package ([#1314](https://github.com/Lightning-AI/metrics/pull/1314))
- Added regression metrics:
   * `KendallRankCorrCoef` ([#1271](https://github.com/Lightning-AI/metrics/pull/1271))
   * `LogCoshError` ([#1316](https://github.com/Lightning-AI/metrics/pull/1316))
- Added new nominal metrics:
  * `CramersV` ([#1298](https://github.com/Lightning-AI/metrics/pull/1298))
  * `PearsonsContingencyCoefficient` ([#1334](https://github.com/Lightning-AI/metrics/pull/1334))
  * `TschuprowsT` ([#1334](https://github.com/Lightning-AI/metrics/pull/1334))
  * `TheilsU` ([#1337](https://github.com/Lightning-AI/metrics/pull/1334))
- Added option to pass `distributed_available_fn` to metrics to allow checks for custom communication backend for making `dist_sync_fn` actually useful ([#1301](https://github.com/Lightning-AI/metrics/pull/1301))
- Added `normalize` argument to `Inception`, `FID`, `KID` metrics ([#1246](https://github.com/Lightning-AI/metrics/pull/1246))

### Changed

- Changed minimum Pytorch version to be 1.8 ([#1263](https://github.com/Lightning-AI/metrics/pull/1263))
- Changed interface for all functional and modular classification metrics after refactor ([#1252](https://github.com/Lightning-AI/metrics/pull/1252))

### Removed

- Removed deprecated `BinnedAveragePrecision`, `BinnedPrecisionRecallCurve`, `RecallAtFixedPrecision` ([#1251](https://github.com/Lightning-AI/metrics/pull/1251))
- Removed deprecated `LabelRankingAveragePrecision`, `LabelRankingLoss` and `CoverageError` ([#1251](https://github.com/Lightning-AI/metrics/pull/1251))
- Removed deprecated `KLDivergence` and `AUC` ([#1251](https://github.com/Lightning-AI/metrics/pull/1251))

### Fixed

- Fixed precision bug in `pairwise_euclidean_distance` ([#1352](https://github.com/Lightning-AI/metrics/pull/1352))

---

## [0.10.3] - 2022-11-16

### Fixed

- Fixed bug in `Metrictracker.best_metric` when `return_step=False` ([#1306](https://github.com/Lightning-AI/metrics/pull/1306))
- Fixed bug to prevent users from going into an infinite loop if trying to iterate of a single metric ([#1320](https://github.com/Lightning-AI/metrics/pull/1320))

## [0.10.2] - 2022-10-31

### Changed

- Changed in-place operation to out-of-place operation in `pairwise_cosine_similarity` ([#1288](https://github.com/Lightning-AI/metrics/pull/1288))

### Fixed

- Fixed high memory usage for certain classification metrics when `average='micro'` ([#1286](https://github.com/Lightning-AI/metrics/pull/1286))
- Fixed precision problems when `structural_similarity_index_measure` was used with autocast ([#1291](https://github.com/Lightning-AI/metrics/pull/1291))
- Fixed slow performance for confusion matrix based metrics ([#1302](https://github.com/Lightning-AI/metrics/pull/1302))
- Fixed restrictive dtype checking in `spearman_corrcoef` when used with autocast ([#1303](https://github.com/Lightning-AI/metrics/pull/1303))


## [0.10.1] - 2022-10-21

### Fixed

- Fixed broken clone method for classification metrics ([#1250](https://github.com/Lightning-AI/metrics/pull/1250))
- Fixed unintentional downloading of `nltk.punkt` when `lsum` not in `rouge_keys` ([#1258](https://github.com/Lightning-AI/metrics/pull/1258))
- Fixed type casting in `MAP` metric between `bool` and `float32` ([#1150](https://github.com/Lightning-AI/metrics/pull/1150))


## [0.10.0] - 2022-10-04

### Added

- Added a new NLP metric `InfoLM` ([#915](https://github.com/Lightning-AI/metrics/pull/915))
- Added `Perplexity` metric ([#922](https://github.com/Lightning-AI/metrics/pull/922))
- Added `ConcordanceCorrCoef` metric to regression package ([#1201](https://github.com/Lightning-AI/metrics/pull/1201))
- Added argument `normalize` to `LPIPS` metric ([#1216](https://github.com/Lightning-AI/metrics/pull/1216))
- Added support for multiprocessing of batches in `PESQ` metric ([#1227](https://github.com/Lightning-AI/metrics/pull/1227))
- Added support for multioutput in `PearsonCorrCoef` and `SpearmanCorrCoef` ([#1200](https://github.com/Lightning-AI/metrics/pull/1200))

### Changed

- Classification refactor (
    [#1054](https://github.com/Lightning-AI/metrics/pull/1054),
    [#1143](https://github.com/Lightning-AI/metrics/pull/1143),
    [#1145](https://github.com/Lightning-AI/metrics/pull/1145),
    [#1151](https://github.com/Lightning-AI/metrics/pull/1151),
    [#1159](https://github.com/Lightning-AI/metrics/pull/1159),
    [#1163](https://github.com/Lightning-AI/metrics/pull/1163),
    [#1167](https://github.com/Lightning-AI/metrics/pull/1167),
    [#1175](https://github.com/Lightning-AI/metrics/pull/1175),
    [#1189](https://github.com/Lightning-AI/metrics/pull/1189),
    [#1197](https://github.com/Lightning-AI/metrics/pull/1197),
    [#1215](https://github.com/Lightning-AI/metrics/pull/1215),
    [#1195](https://github.com/Lightning-AI/metrics/pull/1195)
)
- Changed update in `FID` metric to be done in online fashion to save memory ([#1199](https://github.com/Lightning-AI/metrics/pull/1199))
- Improved performance of retrieval metrics ([#1242](https://github.com/Lightning-AI/metrics/pull/1242))
- Changed `SSIM` and `MSSSIM` update to be online to reduce memory usage ([#1231](https://github.com/Lightning-AI/metrics/pull/1231))

### Deprecated

- Deprecated `BinnedAveragePrecision`, `BinnedPrecisionRecallCurve`, `BinnedRecallAtFixedPrecision` ([#1163](https://github.com/Lightning-AI/metrics/pull/1163))
  * `BinnedAveragePrecision` -> use `AveragePrecision` with `thresholds` arg
  * `BinnedPrecisionRecallCurve` -> use `AveragePrecisionRecallCurve` with `thresholds` arg
  * `BinnedRecallAtFixedPrecision` -> use `RecallAtFixedPrecision` with `thresholds` arg
- Renamed and refactored `LabelRankingAveragePrecision`, `LabelRankingLoss` and `CoverageError` ([#1167](https://github.com/Lightning-AI/metrics/pull/1167))
  * `LabelRankingAveragePrecision` -> `MultilabelRankingAveragePrecision`
  * `LabelRankingLoss` -> `MultilabelRankingLoss`
  * `CoverageError` -> `MultilabelCoverageError`
- Deprecated `KLDivergence` and `AUC` from classification package ([#1189](https://github.com/Lightning-AI/metrics/pull/1189))
  * `KLDivergence` moved to `regression` package
  * Instead of `AUC` use `torchmetrics.utils.compute.auc`

### Fixed

- Fixed a bug in `ssim` when `return_full_image=True` where the score was still reduced ([#1204](https://github.com/Lightning-AI/metrics/pull/1204))
- Fixed MPS support for:
  * MAE metric ([#1210](https://github.com/Lightning-AI/metrics/pull/1210))
  * Jaccard index ([#1205](https://github.com/Lightning-AI/metrics/pull/1205))
- Fixed bug in `ClasswiseWrapper` such that `compute` gave wrong result ([#1225](https://github.com/Lightning-AI/metrics/pull/1225))
- Fixed synchronization of empty list states ([#1219](https://github.com/Lightning-AI/metrics/pull/1219))

---

## [0.9.3] - 2022-08-22

### Added

- Added global option `sync_on_compute` to disable automatic synchronization when `compute` is called ([#1107](https://github.dev/Lightning-AI/metrics/pull/1107))

### Fixed

- Fixed missing reset in `ClasswiseWrapper` ([#1129](https://github.com/Lightning-AI/metrics/pull/1129))
- Fixed `JaccardIndex` multi-label compute ([#1125](https://github.com/Lightning-AI/metrics/pull/1125))
- Fix SSIM propagate device if `gaussian_kernel` is False, add test ([#1149](https://github.com/Lightning-AI/metrics/pull/1149))


## [0.9.2] - 2022-06-29

### Fixed

- Fixed mAP calculation for areas with 0 predictions ([#1080](https://github.com/Lightning-AI/metrics/pull/1080))
- Fixed bug where avg precision state and auroc state was not merge when using MetricCollections ([#1086](https://github.com/Lightning-AI/metrics/pull/1086))
- Skip box conversion if no boxes are present in `MeanAveragePrecision` ([#1097](https://github.com/Lightning-AI/metrics/pull/1097))
- Fixed inconsistency in docs and code when setting `average="none"` in `AveragePrecision` metric ([#1116](https://github.com/Lightning-AI/metrics/pull/1116))


## [0.9.1] - 2022-06-08

### Added

- Added specific `RuntimeError` when metric object is on the wrong device ([#1056](https://github.com/Lightning-AI/metrics/pull/1056))
- Added an option to specify own n-gram weights for `BLEUScore` and `SacreBLEUScore` instead of using uniform weights only. ([#1075](https://github.com/Lightning-AI/metrics/pull/1075))

### Fixed

- Fixed aggregation metrics when input only contains zero ([#1070](https://github.com/Lightning-AI/metrics/pull/1070))
- Fixed `TypeError` when providing superclass arguments as `kwargs` ([#1069](https://github.com/Lightning-AI/metrics/pull/1069))
- Fixed bug related to state reference in metric collection when using compute groups ([#1076](https://github.com/Lightning-AI/metrics/pull/1076))


## [0.9.0] - 2022-05-30

### Added

- Added `RetrievalPrecisionRecallCurve` and `RetrievalRecallAtFixedPrecision` to retrieval package ([#951](https://github.com/Lightning-AI/metrics/pull/951))
- Added class property `full_state_update` that determines `forward` should call `update` once or twice (
    [#984](https://github.com/Lightning-AI/metrics/pull/984),
    [#1033](https://github.com/Lightning-AI/metrics/pull/1033))
- Added support for nested metric collections ([#1003](https://github.com/Lightning-AI/metrics/pull/1003))
- Added `Dice` to classification package ([#1021](https://github.com/Lightning-AI/metrics/pull/1021))
- Added support to segmentation type `segm` as IOU for mean average precision ([#822](https://github.com/Lightning-AI/metrics/pull/822))

### Changed

- Renamed `reduction` argument to `average` in Jaccard score and added additional options ([#874](https://github.com/Lightning-AI/metrics/pull/874))

### Removed

- Removed deprecated `compute_on_step` argument (
    [#962](https://github.com/Lightning-AI/metrics/pull/962),
    [#967](https://github.com/Lightning-AI/metrics/pull/967),
    [#979](https://github.com/Lightning-AI/metrics/pull/979),
    [#990](https://github.com/Lightning-AI/metrics/pull/990),
    [#991](https://github.com/Lightning-AI/metrics/pull/991),
    [#993](https://github.com/Lightning-AI/metrics/pull/993),
    [#1005](https://github.com/Lightning-AI/metrics/pull/1005),
    [#1004](https://github.com/Lightning-AI/metrics/pull/1004),
    [#1007](https://github.com/Lightning-AI/metrics/pull/1007)
)

### Fixed

- Fixed non-empty state dict for a few metrics ([#1012](https://github.com/Lightning-AI/metrics/pull/1012))
- Fixed bug when comparing states while finding compute groups ([#1022](https://github.com/Lightning-AI/metrics/pull/1022))
- Fixed `torch.double` support in stat score metrics ([#1023](https://github.com/Lightning-AI/metrics/pull/1023))
- Fixed `FID` calculation for non-equal size real and fake input ([#1028](https://github.com/Lightning-AI/metrics/pull/1028))
- Fixed case where `KLDivergence` could output `Nan` ([#1030](https://github.com/Lightning-AI/metrics/pull/1030))
- Fixed deterministic for PyTorch<1.8 ([#1035](https://github.com/Lightning-AI/metrics/pull/1035))
- Fixed default value for `mdmc_average` in `Accuracy` ([#1036](https://github.com/Lightning-AI/metrics/pull/1036))
- Fixed missing copy of property when using compute groups in `MetricCollection` ([#1052](https://github.com/Lightning-AI/metrics/pull/1052))

---

## [0.8.2] - 2022-05-06


### Fixed

- Fixed multi device aggregation in `PearsonCorrCoef` ([#998](https://github.com/Lightning-AI/metrics/pull/998))
- Fixed MAP metric when using custom list of thresholds ([#995](https://github.com/Lightning-AI/metrics/pull/995))
- Fixed compatibility between compute groups in `MetricCollection` and prefix/postfix arg ([#1007](https://github.com/Lightning-AI/metrics/pull/1008))
- Fixed compatibility with future Pytorch 1.12 in `safe_matmul` ([#1011](https://github.com/Lightning-AI/metrics/pull/1011), [#1014](https://github.com/Lightning-AI/metrics/pull/1014))


## [0.8.1] - 2022-04-27

### Changed

- Reimplemented the `signal_distortion_ratio` metric, which removed the absolute requirement of `fast-bss-eval` ([#964](https://github.com/Lightning-AI/metrics/pull/964))

### Fixed

- Fixed "Sort currently does not support bool dtype on CUDA" error in MAP for empty preds ([#983](https://github.com/Lightning-AI/metrics/pull/983))
- Fixed `BinnedPrecisionRecallCurve` when `thresholds` argument is not provided ([#968](https://github.com/Lightning-AI/metrics/pull/968))
- Fixed `CalibrationError` to work on logit input ([#985](https://github.com/Lightning-AI/metrics/pull/985))


## [0.8.0] - 2022-04-14

### Added

- Added `WeightedMeanAbsolutePercentageError` to regression package ([#948](https://github.com/Lightning-AI/metrics/pull/948))
- Added new classification metrics:
  * `CoverageError` ([#787](https://github.com/Lightning-AI/metrics/pull/787))
  * `LabelRankingAveragePrecision` and `LabelRankingLoss` ([#787](https://github.com/Lightning-AI/metrics/pull/787))
- Added new image metric:
  * `SpectralAngleMapper` ([#885](https://github.com/Lightning-AI/metrics/pull/885))
  * `ErrorRelativeGlobalDimensionlessSynthesis` ([#894](https://github.com/Lightning-AI/metrics/pull/894))
  * `UniversalImageQualityIndex` ([#824](https://github.com/Lightning-AI/metrics/pull/824))
  * `SpectralDistortionIndex` ([#873](https://github.com/Lightning-AI/metrics/pull/873))
- Added support for `MetricCollection` in `MetricTracker` ([#718](https://github.com/Lightning-AI/metrics/pull/718))
- Added support for 3D image and uniform kernel in `StructuralSimilarityIndexMeasure` ([#818](https://github.com/Lightning-AI/metrics/pull/818))
- Added smart update of `MetricCollection` ([#709](https://github.com/Lightning-AI/metrics/pull/709))
- Added `ClasswiseWrapper` for better logging of classification metrics with multiple output values ([#832](https://github.com/Lightning-AI/metrics/pull/832))
- Added `**kwargs` argument for passing additional arguments to base class ([#833](https://github.com/Lightning-AI/metrics/pull/833))
- Added negative `ignore_index` for the Accuracy metric ([#362](https://github.com/Lightning-AI/metrics/pull/362))
- Added `adaptive_k` for the `RetrievalPrecision` metric ([#910](https://github.com/Lightning-AI/metrics/pull/910))
- Added `reset_real_features` argument image quality assessment metrics ([#722](https://github.com/Lightning-AI/metrics/pull/722))
- Added new keyword argument `compute_on_cpu` to all metrics ([#867](https://github.com/Lightning-AI/metrics/pull/867))

### Changed

- Made `num_classes` in `jaccard_index` a required argument ([#853](https://github.com/Lightning-AI/metrics/pull/853), [#914](https://github.com/Lightning-AI/metrics/pull/914))
- Added normalizer, tokenizer to ROUGE metric ([#838](https://github.com/Lightning-AI/metrics/pull/838))
- Improved shape checking of `permutation_invariant_training` ([#864](https://github.com/Lightning-AI/metrics/pull/864))
- Allowed reduction `None` ([#891](https://github.com/Lightning-AI/metrics/pull/891))
- `MetricTracker.best_metric` will now give a warning when computing on metric that do not have a best ([#913](https://github.com/Lightning-AI/metrics/pull/913))

### Deprecated

- Deprecated argument `compute_on_step` ([#792](https://github.com/Lightning-AI/metrics/pull/792))
- Deprecated passing in `dist_sync_on_step`, `process_group`, `dist_sync_fn` direct argument ([#833](https://github.com/Lightning-AI/metrics/pull/833))

### Removed

- Removed support for versions of [Pytorch-Lightning](https://github.com/Lightning-AI/lightning) lower than v1.5 ([#788](https://github.com/Lightning-AI/metrics/pull/788))
- Removed deprecated functions, and warnings in Text ([#773](https://github.com/Lightning-AI/metrics/pull/773))
  * `WER` and `functional.wer`
- Removed deprecated functions and warnings in Image ([#796](https://github.com/Lightning-AI/metrics/pull/796))
  * `SSIM` and `functional.ssim`
  * `PSNR` and `functional.psnr`
- Removed deprecated functions, and warnings in classification and regression ([#806](https://github.com/Lightning-AI/metrics/pull/806))
  * `FBeta` and `functional.fbeta`
  * `F1` and `functional.f1`
  * `Hinge` and `functional.hinge`
  * `IoU` and `functional.iou`
  * `MatthewsCorrcoef`
  * `PearsonCorrcoef`
  * `SpearmanCorrcoef`
- Removed deprecated functions, and warnings in detection and pairwise ([#804](https://github.com/Lightning-AI/metrics/pull/804))
  * `MAP` and `functional.pairwise.manhattan`
- Removed deprecated functions, and warnings in Audio ([#805](https://github.com/Lightning-AI/metrics/pull/805))
  * `PESQ` and `functional.audio.pesq`
  * `PIT` and `functional.audio.pit`
  * `SDR` and `functional.audio.sdr` and `functional.audio.si_sdr`
  * `SNR` and `functional.audio.snr` and `functional.audio.si_snr`
  * `STOI` and `functional.audio.stoi`
- Removed unused `get_num_classes` from `torchmetrics.utilities.data` ([#914](https://github.com/Lightning-AI/metrics/pull/914))

### Fixed

- Fixed device mismatch for `MAP` metric in specific cases ([#950](https://github.com/Lightning-AI/metrics/pull/950))
- Improved testing speed ([#820](https://github.com/Lightning-AI/metrics/pull/820))
- Fixed compatibility of `ClasswiseWrapper` with the `prefix` argument of `MetricCollection` ([#843](https://github.com/Lightning-AI/metrics/pull/843))
- Fixed `BestScore` on GPU ([#912](https://github.com/Lightning-AI/metrics/pull/912))
- Fixed Lsum computation for `ROUGEScore` ([#944](https://github.com/Lightning-AI/metrics/pull/944))

---

## [0.7.3] - 2022-03-23

### Fixed

- Fixed unsafe log operation in `TweedieDeviace` for power=1 ([#847](https://github.com/Lightning-AI/metrics/pull/847))
- Fixed bug in MAP metric related to either no ground truth or no predictions ([#884](https://github.com/Lightning-AI/metrics/pull/884))
- Fixed `ConfusionMatrix`, `AUROC` and `AveragePrecision` on GPU when running in deterministic mode ([#900](https://github.com/Lightning-AI/metrics/pull/900))
- Fixed NaN or Inf results returned by `signal_distortion_ratio` ([#899](https://github.com/Lightning-AI/metrics/pull/899))
- Fixed memory leak when using `update` method with tensor where `requires_grad=True` ([#902](https://github.com/Lightning-AI/metrics/pull/902))


## [0.7.2] - 2022-02-10

### Fixed

- Minor patches in JOSS paper.


## [0.7.1] - 2022-02-03

### Changed

- Used `torch.bucketize` in calibration error when `torch>1.8` for faster computations ([#769](https://github.com/Lightning-AI/metrics/pull/769))
- Improve mAP performance ([#742](https://github.com/Lightning-AI/metrics/pull/742))

### Fixed

- Fixed check for available modules ([#772](https://github.com/Lightning-AI/metrics/pull/772))
- Fixed Matthews correlation coefficient when the denominator is 0 ([#781](https://github.com/Lightning-AI/metrics/pull/781))


## [0.7.0] - 2022-01-17

### Added

- Added NLP metrics:
  - `MatchErrorRate` ([#619](https://github.com/Lightning-AI/metrics/pull/619))
  - `WordInfoLost` and `WordInfoPreserved` ([#630](https://github.com/Lightning-AI/metrics/pull/630))
  - `SQuAD` ([#623](https://github.com/Lightning-AI/metrics/pull/623))
  - `CHRFScore` ([#641](https://github.com/Lightning-AI/metrics/pull/641))
  - `TranslationEditRate` ([#646](https://github.com/Lightning-AI/metrics/pull/646))
  - `ExtendedEditDistance` ([#668](https://github.com/Lightning-AI/metrics/pull/668))
- Added `MultiScaleSSIM` into image metrics ([#679](https://github.com/Lightning-AI/metrics/pull/679))
- Added Signal to Distortion Ratio (`SDR`) to audio package ([#565](https://github.com/Lightning-AI/metrics/pull/565))
- Added `MinMaxMetric` to wrappers ([#556](https://github.com/Lightning-AI/metrics/pull/556))
- Added `ignore_index` to retrieval metrics ([#676](https://github.com/Lightning-AI/metrics/pull/676))
- Added support for multi references in `ROUGEScore` ([#680](https://github.com/Lightning-AI/metrics/pull/680))
- Added a default VSCode devcontainer configuration ([#621](https://github.com/Lightning-AI/metrics/pull/621))

### Changed

- Scalar metrics will now consistently have additional dimensions squeezed ([#622](https://github.com/Lightning-AI/metrics/pull/622))
- Metrics having third party dependencies removed from global import ([#463](https://github.com/Lightning-AI/metrics/pull/463))
- Untokenized for `BLEUScore` input stay consistent with all the other text metrics ([#640](https://github.com/Lightning-AI/metrics/pull/640))
- Arguments reordered for `TER`, `BLEUScore`, `SacreBLEUScore`, `CHRFScore` now expect input order as predictions first and target second ([#696](https://github.com/Lightning-AI/metrics/pull/696))
- Changed dtype of metric state from `torch.float` to `torch.long` in `ConfusionMatrix` to accommodate larger values ([#715](https://github.com/Lightning-AI/metrics/pull/715))
- Unify `preds`, `target` input argument's naming across all text metrics ([#723](https://github.com/Lightning-AI/metrics/pull/723), [#727](https://github.com/Lightning-AI/metrics/pull/727))
  * `bert`, `bleu`, `chrf`, `sacre_bleu`, `wip`, `wil`, `cer`, `ter`, `wer`, `mer`, `rouge`, `squad`

### Deprecated

- Renamed IoU -> Jaccard Index ([#662](https://github.com/Lightning-AI/metrics/pull/662))
- Renamed text WER metric ([#714](https://github.com/Lightning-AI/metrics/pull/714))
  * `functional.wer` -> `functional.word_error_rate`
  * `WER` -> `WordErrorRate`
- Renamed correlation coefficient classes: ([#710](https://github.com/Lightning-AI/metrics/pull/710))
  * `MatthewsCorrcoef` -> `MatthewsCorrCoef`
  * `PearsonCorrcoef` -> `PearsonCorrCoef`
  * `SpearmanCorrcoef` -> `SpearmanCorrCoef`
- Renamed audio STOI metric: ([#753](https://github.com/Lightning-AI/metrics/pull/753), [#758](https://github.com/Lightning-AI/metrics/pull/758))
  * `audio.STOI` to `audio.ShortTimeObjectiveIntelligibility`
  * `functional.audio.stoi` to `functional.audio.short_time_objective_intelligibility`
- Renamed audio PESQ metrics: ([#751](https://github.com/Lightning-AI/metrics/pull/751))
  * `functional.audio.pesq` -> `functional.audio.perceptual_evaluation_speech_quality`
  * `audio.PESQ` -> `audio.PerceptualEvaluationSpeechQuality`
- Renamed audio SDR metrics: ([#711](https://github.com/Lightning-AI/metrics/pull/711))
  * `functional.sdr` -> `functional.signal_distortion_ratio`
  * `functional.si_sdr` -> `functional.scale_invariant_signal_distortion_ratio`
  * `SDR` -> `SignalDistortionRatio`
  * `SI_SDR` -> `ScaleInvariantSignalDistortionRatio`
- Renamed audio SNR metrics: ([#712](https://github.com/Lightning-AI/metrics/pull/712))
  * `functional.snr` -> `functional.signal_distortion_ratio`
  * `functional.si_snr` -> `functional.scale_invariant_signal_noise_ratio`
  * `SNR` -> `SignalNoiseRatio`
  * `SI_SNR` -> `ScaleInvariantSignalNoiseRatio`
- Renamed F-score metrics: ([#731](https://github.com/Lightning-AI/metrics/pull/731), [#740](https://github.com/Lightning-AI/metrics/pull/740))
  * `functional.f1` ->  `functional.f1_score`
  * `F1` ->  `F1Score`
  * `functional.fbeta` ->  `functional.fbeta_score`
  * `FBeta` ->  `FBetaScore`
- Renamed Hinge metric: ([#734](https://github.com/Lightning-AI/metrics/pull/734))
  * `functional.hinge` ->  `functional.hinge_loss`
  * `Hinge` ->  `HingeLoss`
- Renamed image PSNR metrics ([#732](https://github.com/Lightning-AI/metrics/pull/732))
  * `functional.psnr` -> `functional.peak_signal_noise_ratio`
  * `PSNR` -> `PeakSignalNoiseRatio`
- Renamed image PIT metric: ([#737](https://github.com/Lightning-AI/metrics/pull/737))
  * `functional.pit` ->  `functional.permutation_invariant_training`
  * `PIT` ->  `PermutationInvariantTraining`
- Renamed image SSIM metric: ([#747](https://github.com/Lightning-AI/metrics/pull/747))
  * `functional.ssim` ->  `functional.scale_invariant_signal_noise_ratio`
  * `SSIM` ->  `StructuralSimilarityIndexMeasure`
- Renamed detection `MAP` to `MeanAveragePrecision` metric ([#754](https://github.com/Lightning-AI/metrics/pull/754))
- Renamed Fidelity & LPIPS image metric: ([#752](https://github.com/Lightning-AI/metrics/pull/752))
  * `image.FID` ->  `image.FrechetInceptionDistance`
  * `image.KID` ->  `image.KernelInceptionDistance`
  * `image.LPIPS` ->  `image.LearnedPerceptualImagePatchSimilarity`

### Removed

- Removed `embedding_similarity` metric ([#638](https://github.com/Lightning-AI/metrics/pull/638))
- Removed argument `concatenate_texts` from `wer` metric ([#638](https://github.com/Lightning-AI/metrics/pull/638))
- Removed arguments `newline_sep` and `decimal_places` from `rouge` metric ([#638](https://github.com/Lightning-AI/metrics/pull/638))

### Fixed

- Fixed MetricCollection kwargs filtering when no `kwargs` are present in update signature ([#707](https://github.com/Lightning-AI/metrics/pull/707))

---

## [0.6.2] - 2021-12-15

### Fixed

- Fixed `torch.sort` currently does not support bool `dtype` on CUDA ([#665](https://github.com/Lightning-AI/metrics/pull/665))
- Fixed mAP properly checks if ground truths are empty ([#684](https://github.com/Lightning-AI/metrics/pull/684))
- Fixed initialization of tensors to be on correct device for `MAP` metric ([#673](https://github.com/Lightning-AI/metrics/pull/673))


## [0.6.1] - 2021-12-06

### Changed

- Migrate MAP metrics from pycocotools to PyTorch ([#632](https://github.com/Lightning-AI/metrics/pull/632))
- Use `torch.topk` instead of `torch.argsort` in retrieval precision for speedup ([#627](https://github.com/Lightning-AI/metrics/pull/627))

### Fixed

- Fix empty predictions in MAP metric ([#594](https://github.com/Lightning-AI/metrics/pull/594), [#610](https://github.com/Lightning-AI/metrics/pull/610), [#624](https://github.com/Lightning-AI/metrics/pull/624))
- Fix edge case of AUROC with `average=weighted` on GPU ([#606](https://github.com/Lightning-AI/metrics/pull/606))
- Fixed `forward` in compositional metrics ([#645](https://github.com/Lightning-AI/metrics/pull/645))


## [0.6.0] - 2021-10-28

### Added

- Added audio metrics:
  - Perceptual Evaluation of Speech Quality (PESQ) ([#353](https://github.com/Lightning-AI/metrics/pull/353))
  - Short-Time Objective Intelligibility (STOI) ([#353](https://github.com/Lightning-AI/metrics/pull/353))
- Added Information retrieval metrics:
  - `RetrievalRPrecision` ([#577](https://github.com/Lightning-AI/metrics/pull/577))
  - `RetrievalHitRate` ([#576](https://github.com/Lightning-AI/metrics/pull/576))
- Added NLP metrics:
  - `SacreBLEUScore` ([#546](https://github.com/Lightning-AI/metrics/pull/546))
  - `CharErrorRate` ([#575](https://github.com/Lightning-AI/metrics/pull/575))
- Added other metrics:
  - Tweedie Deviance Score ([#499](https://github.com/Lightning-AI/metrics/pull/499))
  - Learned Perceptual Image Patch Similarity (LPIPS) ([#431](https://github.com/Lightning-AI/metrics/pull/431))
- Added `MAP` (mean average precision) metric to new detection package ([#467](https://github.com/Lightning-AI/metrics/pull/467))
- Added support for float targets in `nDCG` metric ([#437](https://github.com/Lightning-AI/metrics/pull/437))
- Added `average` argument to `AveragePrecision` metric for reducing multi-label and multi-class problems ([#477](https://github.com/Lightning-AI/metrics/pull/477))
- Added `MultioutputWrapper` ([#510](https://github.com/Lightning-AI/metrics/pull/510))
- Added metric sweeping:
  - `higher_is_better` as constant attribute ([#544](https://github.com/Lightning-AI/metrics/pull/544))
  - `higher_is_better` to rest of codebase ([#584](https://github.com/Lightning-AI/metrics/pull/584))
- Added simple aggregation metrics: `SumMetric`, `MeanMetric`, `CatMetric`, `MinMetric`, `MaxMetric` ([#506](https://github.com/Lightning-AI/metrics/pull/506))
- Added pairwise submodule with metrics ([#553](https://github.com/Lightning-AI/metrics/pull/553))
  - `pairwise_cosine_similarity`
  - `pairwise_euclidean_distance`
  - `pairwise_linear_similarity`
  - `pairwise_manhattan_distance`

### Changed

- `AveragePrecision` will now as default output the `macro` average for multilabel and multiclass problems ([#477](https://github.com/Lightning-AI/metrics/pull/477))
- `half`, `double`, `float` will no longer change the dtype of the metric states. Use `metric.set_dtype` instead ([#493](https://github.com/Lightning-AI/metrics/pull/493))
- Renamed `AverageMeter` to `MeanMetric` ([#506](https://github.com/Lightning-AI/metrics/pull/506))
- Changed `is_differentiable` from property to a constant attribute ([#551](https://github.com/Lightning-AI/metrics/pull/551))
- `ROC` and `AUROC` will no longer throw an error when either the positive or negative class is missing. Instead return 0 score and give a warning

### Deprecated

- Deprecated  `functional.self_supervised.embedding_similarity` in favour of new pairwise submodule

### Removed

- Removed `dtype` property ([#493](https://github.com/Lightning-AI/metrics/pull/493))

### Fixed

- Fixed bug in `F1` with `average='macro'` and `ignore_index!=None` ([#495](https://github.com/Lightning-AI/metrics/pull/495))
- Fixed bug in `pit` by using the returned first result to initialize device and type ([#533](https://github.com/Lightning-AI/metrics/pull/533))
- Fixed `SSIM` metric using too much memory ([#539](https://github.com/Lightning-AI/metrics/pull/539))
- Fixed bug where `device` property was not properly update when metric was a child of a module (#542)

---

## [0.5.1] - 2021-08-30

### Added

- Added `device` and `dtype` properties ([#462](https://github.com/Lightning-AI/metrics/pull/462))
- Added `TextTester` class for robustly testing text metrics ([#450](https://github.com/Lightning-AI/metrics/pull/450))

### Changed

- Added support for float targets in `nDCG` metric ([#437](https://github.com/Lightning-AI/metrics/pull/437))

### Removed

- Removed `rouge-score` as dependency for text package ([#443](https://github.com/Lightning-AI/metrics/pull/443))
- Removed `jiwer` as dependency for text package ([#446](https://github.com/Lightning-AI/metrics/pull/446))
- Removed `bert-score` as dependency for text package ([#473](https://github.com/Lightning-AI/metrics/pull/473))

### Fixed

- Fixed ranking of samples in `SpearmanCorrCoef` metric ([#448](https://github.com/Lightning-AI/metrics/pull/448))
- Fixed bug where compositional metrics where unable to sync because of type mismatch ([#454](https://github.com/Lightning-AI/metrics/pull/454))
- Fixed metric hashing ([#478](https://github.com/Lightning-AI/metrics/pull/478))
- Fixed `BootStrapper` metrics not working on GPU ([#462](https://github.com/Lightning-AI/metrics/pull/462))
- Fixed the semantic ordering of kernel height and width in `SSIM` metric ([#474](https://github.com/Lightning-AI/metrics/pull/474))


## [0.5.0] - 2021-08-09

### Added

- Added **Text-related (NLP) metrics**:
  - Word Error Rate (WER) ([#383](https://github.com/Lightning-AI/metrics/pull/383))
  - ROUGE ([#399](https://github.com/Lightning-AI/metrics/pull/399))
  - BERT score ([#424](https://github.com/Lightning-AI/metrics/pull/424))
  - BLUE score ([#360](https://github.com/Lightning-AI/metrics/pull/360))
- Added `MetricTracker` wrapper metric for keeping track of the same metric over multiple epochs ([#238](https://github.com/Lightning-AI/metrics/pull/238))
- Added other metrics:
  - Symmetric Mean Absolute Percentage error (SMAPE) ([#375](https://github.com/Lightning-AI/metrics/pull/375))
  - Calibration error ([#394](https://github.com/Lightning-AI/metrics/pull/394))
  - Permutation Invariant Training (PIT) ([#384](https://github.com/Lightning-AI/metrics/pull/384))
- Added support in `nDCG` metric for target with values larger than 1 ([#349](https://github.com/Lightning-AI/metrics/pull/349))
- Added support for negative targets in `nDCG` metric ([#378](https://github.com/Lightning-AI/metrics/pull/378))
- Added `None` as reduction option in `CosineSimilarity` metric ([#400](https://github.com/Lightning-AI/metrics/pull/400))
- Allowed passing labels in (n_samples, n_classes) to `AveragePrecision` ([#386](https://github.com/Lightning-AI/metrics/pull/386))

### Changed

- Moved `psnr` and `ssim` from `functional.regression.*` to `functional.image.*` ([#382](https://github.com/Lightning-AI/metrics/pull/382))
- Moved `image_gradient` from `functional.image_gradients` to `functional.image.gradients` ([#381](https://github.com/Lightning-AI/metrics/pull/381))
- Moved `R2Score` from `regression.r2score` to `regression.r2` ([#371](https://github.com/Lightning-AI/metrics/pull/371))
- Pearson metric now only store 6 statistics instead of all predictions and targets ([#380](https://github.com/Lightning-AI/metrics/pull/380))
- Use `torch.argmax` instead of `torch.topk` when `k=1` for better performance ([#419](https://github.com/Lightning-AI/metrics/pull/419))
- Moved check for number of samples in R2 score to support single sample updating ([#426](https://github.com/Lightning-AI/metrics/pull/426))

### Deprecated

- Rename `r2score` >> `r2_score` and `kldivergence` >> `kl_divergence` in `functional` ([#371](https://github.com/Lightning-AI/metrics/pull/371))
- Moved `bleu_score` from `functional.nlp` to `functional.text.bleu` ([#360](https://github.com/Lightning-AI/metrics/pull/360))

### Removed

- Removed restriction that `threshold` has to be in (0,1) range to support logit input (
    [#351](https://github.com/Lightning-AI/metrics/pull/351)
    [#401](https://github.com/Lightning-AI/metrics/pull/401))
- Removed restriction that `preds` could not be bigger than `num_classes` to support logit input ([#357](https://github.com/Lightning-AI/metrics/pull/357))
- Removed module `regression.psnr` and `regression.ssim` ([#382](https://github.com/Lightning-AI/metrics/pull/382)):
- Removed ([#379](https://github.com/Lightning-AI/metrics/pull/379)):
    * function `functional.mean_relative_error`
    * `num_thresholds` argument in `BinnedPrecisionRecallCurve`

### Fixed

- Fixed bug where classification metrics with `average='macro'` would lead to wrong result if a class was missing ([#303](https://github.com/Lightning-AI/metrics/pull/303))
- Fixed `weighted`, `multi-class` AUROC computation to allow for 0 observations of some class, as contribution to final AUROC is 0 ([#376](https://github.com/Lightning-AI/metrics/pull/376))
- Fixed that `_forward_cache` and `_computed` attributes are also moved to the correct device if metric is moved ([#413](https://github.com/Lightning-AI/metrics/pull/413))
- Fixed calculation in `IoU` metric when using `ignore_index` argument ([#328](https://github.com/Lightning-AI/metrics/pull/328))

---

## [0.4.1] - 2021-07-05

### Changed

- Extend typing ([#330](https://github.com/Lightning-AI/metrics/pull/330),
    [#332](https://github.com/Lightning-AI/metrics/pull/332),
    [#333](https://github.com/Lightning-AI/metrics/pull/333),
    [#335](https://github.com/Lightning-AI/metrics/pull/335),
    [#314](https://github.com/Lightning-AI/metrics/pull/314))

### Fixed

- Fixed DDP by `is_sync` logic to `Metric` ([#339](https://github.com/Lightning-AI/metrics/pull/339))


## [0.4.0] - 2021-06-29

### Added

- Added **Image-related metrics**:
  - Fréchet inception distance (FID) ([#213](https://github.com/Lightning-AI/metrics/pull/213))
  - Kernel Inception Distance (KID) ([#301](https://github.com/Lightning-AI/metrics/pull/301))
  - Inception Score ([#299](https://github.com/Lightning-AI/metrics/pull/299))
  - KL divergence ([#247](https://github.com/Lightning-AI/metrics/pull/247))
- Added **Audio metrics**: SNR, SI_SDR, SI_SNR ([#292](https://github.com/Lightning-AI/metrics/pull/292))
- Added other metrics:
  - Cosine Similarity ([#305](https://github.com/Lightning-AI/metrics/pull/305))
  - Specificity ([#210](https://github.com/Lightning-AI/metrics/pull/210))
  - Mean Absolute Percentage error (MAPE) ([#248](https://github.com/Lightning-AI/metrics/pull/248))
- Added `add_metrics` method to `MetricCollection` for adding additional metrics after initialization ([#221](https://github.com/Lightning-AI/metrics/pull/221))
- Added pre-gather reduction in the case of `dist_reduce_fx="cat"` to reduce communication cost ([#217](https://github.com/Lightning-AI/metrics/pull/217))
- Added better error message for `AUROC` when `num_classes` is not provided for multiclass input ([#244](https://github.com/Lightning-AI/metrics/pull/244))
- Added support for unnormalized scores (e.g. logits) in `Accuracy`, `Precision`, `Recall`, `FBeta`, `F1`, `StatScore`, `Hamming`, `ConfusionMatrix` metrics ([#200](https://github.com/Lightning-AI/metrics/pull/200))
- Added `squared` argument to `MeanSquaredError` for computing `RMSE` ([#249](https://github.com/Lightning-AI/metrics/pull/249))
- Added `is_differentiable` property to `ConfusionMatrix`, `F1`, `FBeta`, `Hamming`, `Hinge`, `IOU`, `MatthewsCorrcoef`, `Precision`, `Recall`, `PrecisionRecallCurve`, `ROC`, `StatScores` ([#253](https://github.com/Lightning-AI/metrics/pull/253))
- Added `sync` and `sync_context` methods for manually controlling when metric states are synced ([#302](https://github.com/Lightning-AI/metrics/pull/302))

### Changed

- Forward cache is reset when `reset` method is called ([#260](https://github.com/Lightning-AI/metrics/pull/260))
- Improved per-class metric handling for imbalanced datasets for `precision`, `recall`, `precision_recall`, `fbeta`, `f1`, `accuracy`, and `specificity` ([#204](https://github.com/Lightning-AI/metrics/pull/204))
- Decorated `torch.jit.unused` to `MetricCollection` forward ([#307](https://github.com/Lightning-AI/metrics/pull/307))
- Renamed `thresholds` argument to binned metrics for manually controlling the thresholds ([#322](https://github.com/Lightning-AI/metrics/pull/322))
- Extend typing ([#324](https://github.com/Lightning-AI/metrics/pull/324),
    [#326](https://github.com/Lightning-AI/metrics/pull/326),
    [#327](https://github.com/Lightning-AI/metrics/pull/327))

### Deprecated

- Deprecated `functional.mean_relative_error`, use `functional.mean_absolute_percentage_error` ([#248](https://github.com/Lightning-AI/metrics/pull/248))
- Deprecated `num_thresholds` argument in `BinnedPrecisionRecallCurve` ([#322](https://github.com/Lightning-AI/metrics/pull/322))

### Removed

- Removed argument `is_multiclass` ([#319](https://github.com/Lightning-AI/metrics/pull/319))

### Fixed

- AUC can also support more dimensional inputs when all but one dimension are of size 1 ([#242](https://github.com/Lightning-AI/metrics/pull/242))
- Fixed `dtype` of modular metrics after reset has been called ([#243](https://github.com/Lightning-AI/metrics/pull/243))
- Fixed calculation in `matthews_corrcoef` to correctly match formula ([#321](https://github.com/Lightning-AI/metrics/pull/321))

---

## [0.3.2] - 2021-05-10

### Added

- Added `is_differentiable` property:
    * To `AUC`, `AUROC`, `CohenKappa` and `AveragePrecision` ([#178](https://github.com/Lightning-AI/metrics/pull/178))
    * To `PearsonCorrCoef`, `SpearmanCorrcoef`, `R2Score` and `ExplainedVariance` ([#225](https://github.com/Lightning-AI/metrics/pull/225))

### Changed

- `MetricCollection` should return metrics with prefix on `items()`, `keys()` ([#209](https://github.com/Lightning-AI/metrics/pull/209))
- Calling `compute` before `update` will now give warning ([#164](https://github.com/Lightning-AI/metrics/pull/164))

### Removed

- Removed `numpy` as direct dependency ([#212](https://github.com/Lightning-AI/metrics/pull/212))

### Fixed

- Fixed auc calculation and add tests ([#197](https://github.com/Lightning-AI/metrics/pull/197))
- Fixed loading persisted metric states using `load_state_dict()` ([#202](https://github.com/Lightning-AI/metrics/pull/202))
- Fixed `PSNR` not working with `DDP` ([#214](https://github.com/Lightning-AI/metrics/pull/214))
- Fixed metric calculation with unequal batch sizes ([#220](https://github.com/Lightning-AI/metrics/pull/220))
- Fixed metric concatenation for list states for zero-dim input ([#229](https://github.com/Lightning-AI/metrics/pull/229))
- Fixed numerical instability in `AUROC` metric for large input ([#230](https://github.com/Lightning-AI/metrics/pull/230))

## [0.3.1] - 2021-04-21

- Cleaning remaining inconsistency and fix PL develop integration (
    [#191](https://github.com/Lightning-AI/metrics/pull/191),
    [#192](https://github.com/Lightning-AI/metrics/pull/192),
    [#193](https://github.com/Lightning-AI/metrics/pull/193),
    [#194](https://github.com/Lightning-AI/metrics/pull/194)
)


## [0.3.0] - 2021-04-20

### Added

- Added `BootStrapper` to easily calculate confidence intervals for metrics ([#101](https://github.com/Lightning-AI/metrics/pull/101))
- Added Binned metrics  ([#128](https://github.com/Lightning-AI/metrics/pull/128))
- Added metrics for Information Retrieval ([(PL^5032)](https://github.com/Lightning-AI/lightning/pull/5032)):
    * `RetrievalMAP` ([PL^5032](https://github.com/Lightning-AI/lightning/pull/5032))
    * `RetrievalMRR` ([#119](https://github.com/Lightning-AI/metrics/pull/119))
    * `RetrievalPrecision` ([#139](https://github.com/Lightning-AI/metrics/pull/139))
    * `RetrievalRecall` ([#146](https://github.com/Lightning-AI/metrics/pull/146))
    * `RetrievalNormalizedDCG` ([#160](https://github.com/Lightning-AI/metrics/pull/160))
    * `RetrievalFallOut` ([#161](https://github.com/Lightning-AI/metrics/pull/161))
- Added other metrics:
    * `CohenKappa` ([#69](https://github.com/Lightning-AI/metrics/pull/69))
    * `MatthewsCorrcoef` ([#98](https://github.com/Lightning-AI/metrics/pull/98))
    * `PearsonCorrcoef` ([#157](https://github.com/Lightning-AI/metrics/pull/157))
    * `SpearmanCorrcoef` ([#158](https://github.com/Lightning-AI/metrics/pull/158))
    * `Hinge` ([#120](https://github.com/Lightning-AI/metrics/pull/120))
- Added `average='micro'` as an option in AUROC for multilabel problems ([#110](https://github.com/Lightning-AI/metrics/pull/110))
- Added multilabel support to `ROC` metric ([#114](https://github.com/Lightning-AI/metrics/pull/114))
- Added testing for `half` precision ([#77](https://github.com/Lightning-AI/metrics/pull/77),
    [#135](https://github.com/Lightning-AI/metrics/pull/135)
)
- Added `AverageMeter` for ad-hoc averages of values ([#138](https://github.com/Lightning-AI/metrics/pull/138))
- Added `prefix` argument to `MetricCollection` ([#70](https://github.com/Lightning-AI/metrics/pull/70))
- Added `__getitem__` as metric arithmetic operation ([#142](https://github.com/Lightning-AI/metrics/pull/142))
- Added property `is_differentiable` to metrics and test for differentiability ([#154](https://github.com/Lightning-AI/metrics/pull/154))
- Added support for `average`, `ignore_index` and `mdmc_average` in `Accuracy` metric ([#166](https://github.com/Lightning-AI/metrics/pull/166))
- Added `postfix` arg to `MetricCollection` ([#188](https://github.com/Lightning-AI/metrics/pull/188))

### Changed

- Changed `ExplainedVariance` from storing all preds/targets to tracking 5 statistics ([#68](https://github.com/Lightning-AI/metrics/pull/68))
- Changed behaviour of `confusionmatrix` for multilabel data to better match `multilabel_confusion_matrix` from sklearn ([#134](https://github.com/Lightning-AI/metrics/pull/134))
- Updated FBeta arguments ([#111](https://github.com/Lightning-AI/metrics/pull/111))
- Changed `reset` method to use `detach.clone()` instead of `deepcopy` when resetting to default ([#163](https://github.com/Lightning-AI/metrics/pull/163))
- Metrics passed as dict to `MetricCollection` will now always be in deterministic order ([#173](https://github.com/Lightning-AI/metrics/pull/173))
- Allowed `MetricCollection` pass metrics as arguments ([#176](https://github.com/Lightning-AI/metrics/pull/176))

### Deprecated

- Rename argument `is_multiclass` -> `multiclass` ([#162](https://github.com/Lightning-AI/metrics/pull/162))

### Removed

- Prune remaining deprecated ([#92](https://github.com/Lightning-AI/metrics/pull/92))

### Fixed

- Fixed when `_stable_1d_sort` to work when `n>=N` ([PL^6177](https://github.com/Lightning-AI/lightning/pull/6177))
- Fixed `_computed` attribute not being correctly reset ([#147](https://github.com/Lightning-AI/metrics/pull/147))
- Fixed to Blau score ([#165](https://github.com/Lightning-AI/metrics/pull/165))
- Fixed backwards compatibility for logging with older version of pytorch-lightning ([#182](https://github.com/Lightning-AI/metrics/pull/182))

---

## [0.2.0] - 2021-03-12

### Changed

- Decoupled PL dependency ([#13](https://github.com/Lightning-AI/metrics/pull/13))
- Refactored functional - mimic the module-like structure: classification, regression, etc. ([#16](https://github.com/Lightning-AI/metrics/pull/16))
- Refactored utilities -  split to topics/submodules ([#14](https://github.com/Lightning-AI/metrics/pull/14))
- Refactored `MetricCollection` ([#19](https://github.com/Lightning-AI/metrics/pull/19))

### Removed

- Removed deprecated metrics from PL base ([#12](https://github.com/Lightning-AI/metrics/pull/12),
    [#15](https://github.com/Lightning-AI/metrics/pull/15))

---

## [0.1.0] - 2021-02-22

- Added `Accuracy` metric now generalizes to Top-k accuracy for (multi-dimensional) multi-class inputs using the `top_k` parameter ([PL^4838](https://github.com/Lightning-AI/lightning/pull/4838))
- Added `Accuracy` metric now enables the computation of subset accuracy for multi-label or multi-dimensional multi-class inputs with the `subset_accuracy` parameter ([PL^4838](https://github.com/Lightning-AI/lightning/pull/4838))
- Added `HammingDistance` metric to compute the hamming distance (loss) ([PL^4838](https://github.com/Lightning-AI/lightning/pull/4838))
- Added `StatScores` metric to compute the number of true positives, false positives, true negatives and false negatives ([PL^4839](https://github.com/Lightning-AI/lightning/pull/4839))
- Added `R2Score` metric ([PL^5241](https://github.com/Lightning-AI/lightning/pull/5241))
- Added `MetricCollection` ([PL^4318](https://github.com/Lightning-AI/lightning/pull/4318))
- Added `.clone()` method to metrics ([PL^4318](https://github.com/Lightning-AI/lightning/pull/4318))
- Added `IoU` class interface ([PL^4704](https://github.com/Lightning-AI/lightning/pull/4704))
- The `Recall` and `Precision` metrics (and their functional counterparts `recall` and `precision`) can now be generalized to Recall@K and Precision@K with the use of `top_k` parameter ([PL^4842](https://github.com/Lightning-AI/lightning/pull/4842))
- Added compositional metrics ([PL^5464](https://github.com/Lightning-AI/lightning/pull/5464))
- Added AUC/AUROC class interface ([PL^5479](https://github.com/Lightning-AI/lightning/pull/5479))
- Added `QuantizationAwareTraining` callback ([PL^5706](https://github.com/Lightning-AI/lightning/pull/5706))
- Added `ConfusionMatrix` class interface ([PL^4348](https://github.com/Lightning-AI/lightning/pull/4348))
- Added multiclass AUROC metric ([PL^4236](https://github.com/Lightning-AI/lightning/pull/4236))
- Added `PrecisionRecallCurve, ROC, AveragePrecision` class metric ([PL^4549](https://github.com/Lightning-AI/lightning/pull/4549))
- Classification metrics overhaul ([PL^4837](https://github.com/Lightning-AI/lightning/pull/4837))
- Added `F1` class metric ([PL^4656](https://github.com/Lightning-AI/lightning/pull/4656))
- Added metrics aggregation in Horovod and fixed early stopping ([PL^3775](https://github.com/Lightning-AI/lightning/pull/3775))
- Added `persistent(mode)` method to metrics, to enable and disable metric states being added to `state_dict` ([PL^4482](https://github.com/Lightning-AI/lightning/pull/4482))
- Added unification of regression metrics ([PL^4166](https://github.com/Lightning-AI/lightning/pull/4166))
- Added persistent flag to `Metric.add_state` ([PL^4195](https://github.com/Lightning-AI/lightning/pull/4195))
- Added classification metrics ([PL^4043](https://github.com/Lightning-AI/lightning/pull/4043))
- Added new Metrics API. ([PL^3868](https://github.com/Lightning-AI/lightning/pull/3868), [PL^3921](https://github.com/Lightning-AI/lightning/pull/3921))
- Added EMB similarity ([PL^3349](https://github.com/Lightning-AI/lightning/pull/3349))
- Added SSIM metrics ([PL^2671](https://github.com/Lightning-AI/lightning/pull/2671))
- Added BLEU metrics ([PL^2535](https://github.com/Lightning-AI/lightning/pull/2535))
