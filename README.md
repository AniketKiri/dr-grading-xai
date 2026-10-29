# Explainable diabetic retinopathy grading from colour fundus photographs

Five-grade DR severity grading (ICDR 0 to 4) on the colour fundus photography (CFP) subset of
MMRDR, with calibrated confidence, a referral rule for uncertain cases, Grad-CAM explanations with
a randomisation sanity check, and analyses built on the dataset's image-level lesion flags.

Context: a research prototype for AI-supported optometric triage, intended to help optometrists in
community practice decide which patients need extended review. It is not a clinically validated tool.

This is a research prototype. It is not a clinically validated diagnostic tool.

## What the pipeline does

| Stage | Command | Output |
|---|---|---|
| 0. Inspect | `python -m drxai.inspect_data` | Prints CSV columns, grade counts, id prefixes and id-to-file matching. Writes nothing |
| 1. Prepare | `python -m drxai.prepare` | Cropped, cached images, `manifest.csv`, splits, `near_duplicates.csv`, `integrity_report.json` |
| 2. Train | `python -m drxai.train --run-name NAME` | `runs/NAME/` with best checkpoint (chosen on validation QWK), history, config used |
| 3. Evaluate | `python -m drxai.evaluate --run runs/NAME --quality` | `metrics.json`, `results.md`, `predictions_test.csv`, figures |
| 4. Explain | `python -m drxai.explain --run runs/NAME` | Grad-CAM gallery, clinician-facing case cards, randomisation sanity check, lesion agreement |
| 5. Compare | `python -m drxai.compare --runs runs/A runs/B` | Primary model chosen on validation; paired bootstrap CI for the test difference |
| 6. Lesion flags | `python -m drxai.lesions_report --run runs/NAME` | Grade-lesion consistency, lesion probes against an ImageNet-only floor, missed referable cases by lesion |

## Data

MMRDR (Tang et al., *Scientific Data*, 2026; doi:10.1038/s41597-026-07005-9) is released on figshare
under CC BY 4.0: doi:10.6084/m9.figshare.29423747. The full archive is about 18.6 GB across CFP, OCT
and ultra-widefield imaging. Only the CFP subset is used here.

The dataset is not included in this repository. After download, the expected layout is:

```
data/MMRDR/DatasetRT/MMRDR-CFP/
    img/                 colour fundus photographs
    <annotation>.csv     one row per image with its DR grade
```

Set `paths.data_root` in `configs/default.yaml` (or pass `--data-root`) if you place it elsewhere.
Run `inspect_data` first and check its output against the `columns`, `split_from_id_prefix` and
`explain.lesions` settings in the config. The column names, the split encoding and the lesion
annotation format are all confirmed from the data at that step, not assumed.

## Installation

Developed and run with Python 3.11 in a conda environment named `drxai`, with PyTorch installed from
the pip CUDA 12.6 wheels (torch 2.14.1, torchvision 0.29.1) rather than from conda. Install PyTorch
first, then the rest:

```bash
conda create -n drxai python=3.11 -y
conda activate drxai
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126   # match your CUDA
pip install -r requirements.txt
pip install -e .
```

Set `PYTHONNOUSERSITE=1` before running anything (`set PYTHONNOUSERSITE=1` on Windows) so that
packages in the user site directory cannot shadow the environment's own.

## Reproducing the results

```bash
pytest tests/test_numerics.py                 # metric and geometry checks against hand-worked values
pytest tests/test_smoke.py -s                 # every stage end to end on synthetic data, about a minute
python -m drxai.inspect_data
python -m drxai.prepare --workers 4
python -m drxai.train --run-name resnet50_corn
python -m drxai.train --run-name resnet50_ce --head ce        # softmax baseline
python -m drxai.evaluate --run runs/resnet50_corn --quality
python -m drxai.evaluate --run runs/resnet50_ce
python -m drxai.compare --runs runs/resnet50_corn runs/resnet50_ce
python -m drxai.explain --run runs/<primary chosen by compare>
python -m drxai.lesions_report --run runs/<primary chosen by compare>
pip freeze > environment-lock.txt             # exact versions used for the reported results
```

Every run saves the exact configuration it used (`config_used.json`) and the library versions and GPU
(`train_summary.json`). The seed is set in the config (default 42). `cudnn.benchmark` is enabled for
speed, so GPU results can vary slightly between runs. Pass `--seed` to repeat a run with another seed.

## Notebooks

`notebooks/` holds three read-only notebooks (data exploration, results, explainability). They only
read files the pipeline has already saved, so run them after the commands above. They train nothing,
run no model and compute nothing reported elsewhere: every reported number comes from the pipeline's
own output files. Apart from those numbers, they only make small display summaries such as grade
counts by split.

## Design decisions

**Splits.** If MMRDR defines a train/test split (in a column or in the image-id prefix), it is used
unchanged and the test set is never modified. Validation is carved from the training pool only.
The test split is never loaded during training or model selection.

**No patient identifiers.** MMRDR does not appear to release patient ids, so images of the same eye
cannot be kept on one side of a split. As a partial guard, `prepare` finds the same photograph stored
under two ids (high-pass fingerprint correlation, central field of view only), keeps such copies
in one group, and drops any training image that duplicates a test image. Different photographs of the
same eye are not detected. This is a stated limitation, not a solved problem.

**Grey camera surround.** 317 of the 11,062 images used (about 3%) have a dark grey surround (level
about 20) instead of black, which a fixed brightness threshold counts as retina. In training and
validation data these images are 0.5% of grade 0 but 4.3% to 9.0% of grades 1 to 4
(`logs/grey_surround.txt`), so the border alone would hint at the grade. `prepare` estimates each image's surround from its four corners and, where it is grey, raises
the threshold above it and sets the surround to black. Images with a black surround, and five with a
white canvas, are processed exactly as before (checked byte for byte). The `grey_surround` manifest
column keeps the flag so their test performance can be reported as a subgroup.

**Ordinal head.** Grades are ordered, and quadratic weighted kappa penalises distant errors more. The
default head is CORN (Shi, Cao and Raschka, 2023), which predicts conditional threshold probabilities
and guarantees a monotone survival function. The predicted grade is the median of the predictive
distribution. A plain softmax head (`--head ce`) is trained as the baseline.

**What is chosen where.** Everything that is *chosen* uses training or validation data only: the
best epoch (validation QWK), the temperature, the referral threshold (set on validation for 80%
automatic coverage), the image-quality band edges (training images), and which model is primary
(validation QWK, via `compare`). The test split is only measured, once, after those choices are fixed.

**Hyperparameters.** No search was run. Learning rate, schedule, batch size, input size and epochs are
standard fine-tuning defaults fixed before training, chosen to fit a 6 GB GPU. The one design choice
compared is the output head (ordinal against softmax), decided on validation. Data loading uses
2 worker processes: on Windows each worker re-imports PyTorch (about 1.4 GB of committed memory),
and 4 workers exhausted a 16 GB laptop. This changes loading speed only, not the experiment. `--lr` and `--seed`
are exposed so a small validation-only sweep or seed repeat can be added.

**Imbalance.** Training uses class sampling weighted by the inverse square root of class frequency.
Full inverse weighting would show the few severe and proliferative images many times per epoch.

**Floors.** Every headline metric is reported next to a majority-class floor (always predicting the
commonest training grade) and as a margin above it. The referral curve is compared with referring the
same number of cases at random.

**Image quality.** `prepare` records sharpness (variance of the Laplacian), brightness, contrast and
field-of-view fraction for every image and lists the least sharp 2%. `evaluate` reports accuracy,
QWK and mean confidence in sharpness bands, so it shows whether confidence falls where performance
does. `--quality` adds simulated blur and under-exposure on the test images.

**Calibration.** A single temperature is fitted on validation and applied to test. For the CORN
head, temperature changes confidence only, never the predicted grade.

**Explanations.** Grad-CAM (Selvaraju et al., 2017) on the last convolutional block. For the
ordinal head the target is the expected grade, so the map shows regions that raise the predicted
severity. Two checks guard against over-reading the maps:

- a model-randomisation test (Adebayo et al., 2018): if the map barely changes when the explained
  block's weights are re-initialised, it reflects image structure rather than what the model learned;
- case cards that show the model's prediction and action, the explanation, and the clinical judgement
  as three separate panels, including a confident error that the referral rule cannot catch;
- lesion-flag analyses (`lesions_report`): MMRDR gives seven present/absent lesion flags per image
  but no locations, so heatmaps cannot be scored spatially. Instead: grade-lesion consistency (label
  quality), a linear probe of each lesion in the model's features compared with the same probe on an
  ImageNet-only network, and the lesion make-up of missed referable cases.
- spatial agreement (pointing game and enrichment over lesion area) is implemented for datasets with
  lesion masks or boxes (`explain.lesions`), but is switched off for MMRDR.

## Integrity checks

`prepare` stops on any critical failure and records every check in `cache/integrity_report.json`:
grades in range, unique ids, every id resolving to exactly one image whose filename matches,
duplicate detection with a guard against an over-eager detector, no group spanning two splits,
cached images readable at the expected size, and a manifest round-trip.

## Repository layout

```
configs/default.yaml        every setting in one place
src/drxai/
    inspect_data.py prepare.py train.py evaluate.py explain.py compare.py    the stages
    data.py labels.py lesions.py ordinal.py metrics.py calibration.py    NumPy-only, unit-tested
    datasets.py model.py inference.py plots.py
tests/
    test_numerics.py        hand-checked values for every reported metric
    test_smoke.py           synthetic end-to-end run
```

## Acknowledgements and external resources

- MMRDR dataset, CC BY 4.0 (Tang et al., 2026).
- ImageNet-pretrained ResNet-50 and EfficientNet-B0 weights from torchvision.
- CORN ordinal regression: Shi, Cao and Raschka, *Pattern Analysis and Applications*, 2023.
- Grad-CAM: Selvaraju et al., ICCV 2017. Saliency sanity checks: Adebayo et al., NeurIPS 2018.
- PyTorch, torchvision, scikit-learn, SciPy, NumPy, pandas, Matplotlib, Pillow.
