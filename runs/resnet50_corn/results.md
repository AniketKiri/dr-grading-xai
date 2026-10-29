# Test results (n = 2225)

| Metric | Model | 95% CI | Majority-class floor | Margin |
|---|---|---|---|---|
| Quadratic weighted kappa | 0.908 | 0.893 to 0.921 | 0.000 | +0.908 |
| Macro-F1 | 0.725 | 0.699 to 0.748 | 0.149 | +0.575 |
| Accuracy | 0.827 | 0.811 to 0.842 | 0.595 | +0.232 |

| Grade | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| 0 | 0.954 | 0.951 | 0.952 | 1323 |
| 1 | 0.481 | 0.587 | 0.529 | 259 |
| 2 | 0.723 | 0.619 | 0.667 | 409 |
| 3 | 0.551 | 0.700 | 0.617 | 100 |
| 4 | 0.938 | 0.791 | 0.858 | 134 |

## Error direction
- Errors: 386; under-graded 55.4%, over-graded 44.6%; two or more grades out: 50
- Referable DR (grade >= 2): sensitivity 0.804, specificity 0.954, AUROC 0.972, missed referable cases 126
- Severe disease (grade >= 3, n = 234): share called grade >= 3 0.829 (95% CI 0.782 to 0.877)

## Calibration
- Temperature (fitted on validation): 1.503
- ECE uncalibrated 0.056; temperature-scaled 0.025

## Referral curve
| Coverage | Referred | Accuracy | QWK | Missed referable caught | Random-referral floor |
|---|---|---|---|---|---|
| 1.00 | 0 | 0.827 | 0.908 | 0 | 0.0 |
| 0.95 | 111 | 0.848 | 0.922 | 14 | 6.3 |
| 0.90 | 223 | 0.866 | 0.931 | 30 | 12.6 |
| 0.85 | 334 | 0.885 | 0.939 | 50 | 18.9 |
| 0.80 | 445 | 0.906 | 0.951 | 69 | 25.2 |
| 0.75 | 556 | 0.931 | 0.958 | 89 | 31.5 |
| 0.70 | 667 | 0.947 | 0.965 | 102 | 37.8 |
| 0.60 | 890 | 0.972 | 0.980 | 114 | 50.4 |
| 0.50 | 1113 | 0.987 | 0.992 | 124 | 63.0 |

## Referral rule (threshold fixed on validation)
- Refer if confidence < 0.637 (target coverage 80%); achieved coverage on test 79.5%, 456 referred
- Accuracy on automatically decided cases 0.908, QWK 0.951; accuracy on referred cases 0.511
- Missed referable cases sent to review: 70 of 126 (random referral of the same number: 25.8)

## Performance by image sharpness (band edges from training images)
Sharpness falls with grade, so compare the grade mix before reading band accuracy.

| Band | n | Grades 0/1/2/3/4 | Accuracy | QWK | Mean confidence |
|---|---|---|---|---|---|
| low | 763 | 371/100/173/40/79 | 0.796 | 0.906 | 0.809 |
| middle | 714 | 443/79/135/28/29 | 0.819 | 0.903 | 0.817 |
| high | 748 | 509/80/101/32/26 | 0.865 | 0.906 | 0.852 |
- Spearman, sharpness vs confidence 0.092; sharpness vs correct 0.072

## Grey camera surround subgroup (measurement only; nothing is chosen from it)
Cells with fewer than 10 images show the count and no metric.

| Subset | n | Grades 0/1/2/3/4 | Accuracy (95% CI) | QWK (95% CI) |
|---|---|---|---|---|
| Grey surround | 63 | 4/19/20/11/9 | 0.683 (0.571 to 0.794) | 0.810 (0.680 to 0.898) |
| Black surround | 2162 | 1319/240/389/89/125 | 0.831 (0.815 to 0.846) | 0.908 (0.892 to 0.922) |
| Difference (grey minus black) | | | -0.148 (-0.261 to -0.036) | -0.098 (-0.231 to -0.008) |

Share predicted referable within each true grade (separates a surround effect from grade mix):

| True grade | n grey | Grey | n black | Black |
|---|---|---|---|---|
| 0 | 4 |  | 1319 | 0.006 |
| 1 | 19 | 0.263 | 240 | 0.250 |
| 2 | 20 | 0.650 | 389 | 0.717 |
| 3 | 11 | 0.727 | 89 | 0.978 |
| 4 | 9 |  | 125 | 0.968 |

Validation QWK (for model choice only): 0.920
