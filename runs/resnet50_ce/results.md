# Test results (n = 2225)

| Metric | Model | 95% CI | Majority-class floor | Margin |
|---|---|---|---|---|
| Quadratic weighted kappa | 0.909 | 0.895 to 0.923 | 0.000 | +0.909 |
| Macro-F1 | 0.725 | 0.697 to 0.749 | 0.149 | +0.575 |
| Accuracy | 0.831 | 0.815 to 0.846 | 0.595 | +0.236 |

| Grade | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| 0 | 0.932 | 0.971 | 0.951 | 1323 |
| 1 | 0.525 | 0.533 | 0.529 | 259 |
| 2 | 0.727 | 0.599 | 0.657 | 409 |
| 3 | 0.563 | 0.670 | 0.612 | 100 |
| 4 | 0.891 | 0.858 | 0.875 | 134 |

## Error direction
- Errors: 376; under-graded 59.6%, over-graded 40.4%; two or more grades out: 56
- Referable DR (grade >= 2): sensitivity 0.807, specificity 0.958, AUROC 0.972, missed referable cases 124
- Severe disease (grade >= 3, n = 234): share called grade >= 3 0.846 (95% CI 0.799 to 0.891)

## Calibration
- Temperature (fitted on validation): 1.303
- ECE uncalibrated 0.030; temperature-scaled 0.018

## Referral curve
| Coverage | Referred | Accuracy | QWK | Missed referable caught | Random-referral floor |
|---|---|---|---|---|---|
| 1.00 | 0 | 0.831 | 0.909 | 0 | 0.0 |
| 0.95 | 111 | 0.855 | 0.923 | 28 | 6.2 |
| 0.90 | 223 | 0.874 | 0.934 | 43 | 12.4 |
| 0.85 | 334 | 0.889 | 0.940 | 55 | 18.6 |
| 0.80 | 445 | 0.912 | 0.947 | 70 | 24.8 |
| 0.75 | 556 | 0.927 | 0.955 | 81 | 31.0 |
| 0.70 | 667 | 0.941 | 0.960 | 95 | 37.2 |
| 0.60 | 890 | 0.970 | 0.973 | 113 | 49.6 |
| 0.50 | 1113 | 0.988 | 0.989 | 120 | 62.0 |

## Referral rule (threshold fixed on validation)
- Refer if confidence < 0.617 (target coverage 80%); achieved coverage on test 80.4%, 437 referred
- Accuracy on automatically decided cases 0.911, QWK 0.946; accuracy on referred cases 0.506
- Missed referable cases sent to review: 69 of 124 (random referral of the same number: 24.4)

## Performance by image sharpness (band edges from training images)
Sharpness falls with grade, so compare the grade mix before reading band accuracy.

| Band | n | Grades 0/1/2/3/4 | Accuracy | QWK | Mean confidence |
|---|---|---|---|---|---|
| low | 763 | 371/100/173/40/79 | 0.802 | 0.907 | 0.798 |
| middle | 714 | 443/79/135/28/29 | 0.831 | 0.903 | 0.811 |
| high | 748 | 509/80/101/32/26 | 0.861 | 0.909 | 0.856 |
- Spearman, sharpness vs confidence 0.142; sharpness vs correct 0.064

## Grey camera surround subgroup (measurement only; nothing is chosen from it)
Cells with fewer than 10 images show the count and no metric.

| Subset | n | Grades 0/1/2/3/4 | Accuracy (95% CI) | QWK (95% CI) |
|---|---|---|---|---|
| Grey surround | 63 | 4/19/20/11/9 | 0.683 (0.571 to 0.794) | 0.810 (0.674 to 0.894) |
| Black surround | 2162 | 1319/240/389/89/125 | 0.835 (0.819 to 0.851) | 0.910 (0.894 to 0.924) |
| Difference (grey minus black) | | | -0.153 (-0.271 to -0.044) | -0.100 (-0.228 to -0.007) |

Share predicted referable within each true grade (separates a surround effect from grade mix):

| True grade | n grey | Grey | n black | Black |
|---|---|---|---|---|
| 0 | 4 |  | 1319 | 0.007 |
| 1 | 19 | 0.105 | 240 | 0.229 |
| 2 | 20 | 0.500 | 389 | 0.728 |
| 3 | 11 | 0.727 | 89 | 0.978 |
| 4 | 9 |  | 125 | 0.976 |

Validation QWK (for model choice only): 0.904
