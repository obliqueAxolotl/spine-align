# Labelling network -- eval_test

Samples: 16

## Part labels
accuracy 0.882, mIoU 0.696

| class | IoU |
|---|---|
| spinous_process | 0.876 |
| lamina_left | 0.528 |
| lamina_right | 0.480 |
| transverse_left | 0.804 |
| transverse_right | 0.826 |
| facet_left | 0.566 |
| facet_right | 0.539 |
| pedicle_left | 0.578 |
| pedicle_right | 0.583 |
| vertebral_body | 0.949 |
| other_bone | 0.930 |

## Level coordinate
- relative index correct per vertebra (peaks): **0.9375** over 64 vertebrae (target > 0.99)
- relative index correct per vertebra (round): 0.9844
- off-by-one vertebrae: 4
- samples with every vertebra correct: 0.938
- point MAE 0.150, p95 0.477
- sigma coverage: 0.716 within 1 (0.683 ideal), 0.939 within 2 (0.954 ideal)

## Centroid offsets
- per-point error median 4.25 mm, p95 9.02 mm
- clustered centroid error median 1.29 mm, p95 2.75 mm, worst 3.85 mm
- missed 1, spurious 4, order-consistent samples 1.0

## Confidence
- ECE 0.0357 (uncalibrated 0.0783), Brier 0.1153 (uncalibrated 0.1205), positive rate 0.853

## Per level (caudal first)

| level | n | part_acc | index_ok_peaks | level_mae | offset_err_mm | centroid_err_mm | centroid_err_mm_worst |
|---|---|---|---|---|---|---|---|
| L5 | 4 | 0.815 | 1.000 | 0.022 | 9.551 | 1.838 | 1.838 |
| L4 | 12 | 0.791 | 1.000 | 0.027 | 6.315 | 2.892 | 3.854 |
| L3 | 14 | 0.861 | 0.917 | 0.141 | 4.401 | 1.552 | 2.662 |
| L2 | 14 | 0.884 | 0.923 | 0.155 | 4.327 | 1.428 | 2.097 |
| L1 | 14 | 0.882 | 0.929 | 0.178 | 3.984 | 0.992 | 2.140 |
| T12 | 14 | 0.830 | 0.923 | 0.222 | 4.498 | 1.305 | 2.619 |
| T11 | 11 | 0.916 | 1.000 | 0.196 | 4.811 | 1.467 | 2.692 |
| T10 | 2 | 0.441 | - | - | - | - | - |

