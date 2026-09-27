# Held-out comparison (7500 test-bucket items, images disjoint from all training)

Models: `base` = base model, no adapter, `rft` = RFT adapter (the policy stage's starting point), `new` = `artifacts/jevomir-v2/adapter`

## Per model

| subset | model | n | acc | acc (permuted opts) | order agreement | NLL | Brier | ECE | mean conf |
|---|---|---|---|---|---|---|---|---|---|
| clevr | base | 1500 | 0.895 | 0.881 | 0.931 | 0.280 | 0.155 | 0.0264 | 0.878 |
| clevr | rft | 1500 | 0.988 | 0.987 | 0.996 | 0.048 | 0.021 | 0.0055 | 0.991 |
| clevr | new | 1500 | 0.987 | 0.989 | 0.996 | 0.037 | 0.019 | 0.0051 | 0.988 |
| iconqa | base | 1500 | 0.857 | 0.851 | 0.881 | 0.344 | 0.191 | 0.0286 | 0.872 |
| iconqa | rft | 1500 | 0.976 | 0.980 | 0.988 | 0.065 | 0.034 | 0.0111 | 0.979 |
| iconqa | new | 1500 | 0.987 | 0.985 | 0.999 | 0.053 | 0.026 | 0.0054 | 0.982 |
| nlvr2 | base | 1500 | 0.847 | 0.827 | 0.865 | 0.361 | 0.225 | 0.0364 | 0.817 |
| nlvr2 | rft | 1500 | 0.917 | 0.914 | 0.995 | 0.307 | 0.140 | 0.0570 | 0.967 |
| nlvr2 | new | 1500 | 0.923 | 0.921 | 0.995 | 0.217 | 0.123 | 0.0319 | 0.947 |
| tallyqa | base | 1500 | 0.717 | 0.719 | 0.814 | 0.854 | 0.368 | 0.0462 | 0.690 |
| tallyqa | rft | 1500 | 0.759 | 0.757 | 0.938 | 0.782 | 0.346 | 0.1044 | 0.862 |
| tallyqa | new | 1500 | 0.763 | 0.761 | 0.943 | 0.646 | 0.313 | 0.0491 | 0.808 |
| vqav2 | base | 1500 | 0.897 | 0.890 | 0.929 | 0.293 | 0.146 | 0.0499 | 0.855 |
| vqav2 | rft | 1500 | 0.924 | 0.920 | 0.985 | 0.218 | 0.113 | 0.0328 | 0.953 |
| vqav2 | new | 1500 | 0.929 | 0.923 | 0.985 | 0.187 | 0.103 | 0.0127 | 0.934 |
| ALL | base | 7500 | 0.843 | 0.834 | 0.884 | 0.427 | 0.217 | 0.0282 | 0.822 |
| ALL | rft | 7500 | 0.913 | 0.912 | 0.980 | 0.284 | 0.131 | 0.0382 | 0.951 |
| ALL | new | 7500 | 0.918 | 0.916 | 0.983 | 0.228 | 0.117 | 0.0155 | 0.932 |

## `base` minus `rft` (paired bootstrap 95% CI, 1000 resamples; * = CI excludes 0)

Lower is better for NLL, Brier and ECE.

| subset | Δacc | ΔNLL | ΔBrier | ΔECE | right→wrong | wrong→right | McNemar p | answers changed |
|---|---|---|---|---|---|---|---|---|
| clevr | -0.093 [-0.109, -0.079] * | +0.232 [+0.201, +0.263] * | +0.134 [+0.117, +0.151] * | +0.0209 [+0.0095, +0.0334] * | 145 | 5 | 8.58e-37 | 154 |
| iconqa | -0.119 [-0.138, -0.101] * | +0.279 [+0.245, +0.316] * | +0.157 [+0.136, +0.177] * | +0.0174 [+0.0071, +0.0331] * | 195 | 16 | 2.72e-40 | 214 |
| nlvr2 | -0.070 [-0.089, -0.053] * | +0.054 [+0.002, +0.106] * | +0.084 [+0.063, +0.106] * | -0.0206 [-0.0389, +0.0047] | 152 | 47 | 4.16e-14 | 199 |
| tallyqa | -0.042 [-0.059, -0.027] * | +0.073 [+0.015, +0.127] * | +0.022 [+0.004, +0.039] * | -0.0582 [-0.0803, -0.0286] * | 114 | 51 | 1.04e-06 | 256 |
| vqav2 | -0.027 [-0.039, -0.013] * | +0.075 [+0.045, +0.105] * | +0.033 [+0.021, +0.046] * | +0.0171 [+0.0014, +0.0365] * | 67 | 27 | 4.5e-05 | 113 |
| ALL | -0.070 [-0.078, -0.063] * | +0.143 [+0.124, +0.161] * | +0.086 [+0.078, +0.094] * | -0.0100 [-0.0184, -0.0005] * | 673 | 146 | 1.43e-81 | 936 |

### Sample of changed answers (`rft` → `base`)

| subset | question | options | gold | rft | base |
|---|---|---|---|---|---|
| tallyqa | How many people are there? | A=0; B=1; C=2; D=3; E=4; F=5; G=6; H=7; I=8; J=9; K=10 | F | G (0.40) ✗ | H (0.25) ✗ |
| clevr | How many things are either blue things behind the tiny blue cylinder or tiny cylinders beh | A=0; B=1; C=2; D=3; E=4; F=5; G=6; H=7; I=8; J=9; K=10 | E | D (0.99) ✗ | C (0.57) ✗ |
| tallyqa | How many pillows are pictured? | A=0; B=1; C=2; D=3; E=4; F=5; G=6; H=7; I=8; J=9; K=10 | I | H (0.35) ✗ | C (0.17) ✗ |
| iconqa | The first picture is a tractor. Which picture is seventh? | A=tractor; B=sun; C=barn | B | B (0.97) ✓ | A (0.68) ✗ |
| iconqa | The first picture is a fish. Which picture is sixth? | A=sub; B=fish; C=turtle | A | A (0.78) ✓ | C (0.99) ✗ |
| tallyqa | How many horses do not have riders? | A=0; B=1; C=2; D=3; E=4; F=5; G=6; H=7; I=8; J=9; K=10 | B | C (0.87) ✗ | A (0.34) ✗ |
| tallyqa | How many motorcycles are in the picture? | A=0; B=1; C=2; D=3; E=4; F=5; G=6; H=7; I=8; J=9; K=10 | C | C (0.52) ✓ | F (0.21) ✗ |
| iconqa | The first picture is a plane. Which picture is seventh? | A=plane; B=bear; C=train | B | B (0.97) ✓ | C (0.96) ✗ |
| tallyqa | How many people are in the photo? | A=0; B=1; C=2; D=3; E=4; F=5; G=6; H=7; I=8; J=9; K=10 | D | E (0.94) ✗ | D (0.54) ✓ |
| nlvr2 | The first image is the image on the left, the second image is the image on the right. Exam | A=Yes; B=No | B | B (0.75) ✓ | A (0.53) ✗ |
| iconqa | What fraction of the shapes are circles? | A=6/10; B=5/12; C=6/7; D=2/11 | A | A (0.98) ✓ | C (0.54) ✗ |
| clevr | How many things are shiny cubes that are to the right of the large green shiny cube or cya | A=0; B=1; C=2; D=3; E=4; F=5; G=6; H=7; I=8; J=9; K=10 | C | C (0.99) ✓ | B (0.68) ✗ |
| tallyqa | How many layers is the cake? | A=0; B=1; C=2; D=3; E=4; F=5; G=6; H=7; I=8; J=9; K=10 | D | D (0.77) ✓ | C (0.36) ✗ |
| clevr | How many objects are tiny purple objects in front of the purple block or metallic objects  | A=0; B=1; C=2; D=3; E=4; F=5; G=6; H=7; I=8; J=9; K=10 | C | C (0.72) ✓ | B (0.51) ✗ |
| nlvr2 | The first image is the image on the left, the second image is the image on the right. Anal | A=Yes; B=No | A | A (1.00) ✓ | B (0.59) ✗ |
| vqav2 | How many glasses are in this picture? | A=0; B=1; C=2; D=3; E=4; F=5; G=6; H=7; I=8; J=9; K=10 | G | F (0.82) ✗ | E (0.41) ✗ |
| nlvr2 | The first image is the image on the left, the second image is the image on the right. Anal | A=Yes; B=No | B | B (1.00) ✓ | A (0.73) ✗ |
| iconqa | How many rectangles are there? | A=6; B=1; C=3; D=5; E=8 | A | D (0.62) ✗ | A (0.81) ✓ |
| iconqa | The first picture is a house. Which picture is eighth? | A=house; B=paw; C=dog | B | B (1.00) ✓ | C (0.81) ✗ |
| clevr | There is a yellow rubber object that is left of the purple sphere; how many small cubes ar | A=0; B=1; C=2; D=3; E=4; F=5; G=6; H=7; I=8; J=9; K=10 | D | D (0.90) ✓ | B (0.39) ✗ |
| tallyqa | How many chairs can you see? | A=0; B=1; C=2; D=3; E=4; F=5; G=6; H=7; I=8; J=9; K=10 | D | D (0.56) ✓ | C (0.37) ✗ |
| clevr | How many objects are cyan objects or small cyan shiny things? | A=0; B=1; C=2; D=3; E=4; F=5; G=6; H=7; I=8; J=9; K=10 | D | D (0.92) ✓ | C (0.57) ✗ |
| tallyqa | How many knives can you see? | A=0; B=1; C=2; D=3; E=4; F=5; G=6; H=7; I=8; J=9; K=10 | B | C (0.65) ✗ | B (0.46) ✓ |
| iconqa | How many dots are on the frame? | A=1; B=8; C=9; D=10; E=5 | C | C (1.00) ✓ | B (0.86) ✗ |
| vqav2 | Could this be at a school? | A=Yes; B=No | A | B (0.97) ✗ | A (0.56) ✓ |

## `new` minus `rft` (paired bootstrap 95% CI, 1000 resamples; * = CI excludes 0)

Lower is better for NLL, Brier and ECE.

| subset | Δacc | ΔNLL | ΔBrier | ΔECE | right→wrong | wrong→right | McNemar p | answers changed |
|---|---|---|---|---|---|---|---|---|
| clevr | -0.001 [-0.005, +0.004] | -0.011 [-0.024, -0.001] * | -0.002 [-0.007, +0.002] | -0.0003 [-0.0055, +0.0031] | 6 | 5 | 1 | 11 |
| iconqa | +0.011 [+0.005, +0.017] * | -0.012 [-0.021, -0.004] * | -0.008 [-0.013, -0.003] * | -0.0057 [-0.0095, +0.0024] | 2 | 18 | 0.000402 | 20 |
| nlvr2 | +0.007 [+0.000, +0.014] | -0.090 [-0.118, -0.063] * | -0.018 [-0.026, -0.010] * | -0.0251 [-0.0321, -0.0128] * | 9 | 19 | 0.0872 | 28 |
| tallyqa | +0.004 [-0.007, +0.015] | -0.136 [-0.169, -0.107] * | -0.033 [-0.043, -0.024] * | -0.0553 [-0.0656, -0.0394] * | 37 | 43 | 0.576 | 116 |
| vqav2 | +0.005 [-0.001, +0.013] | -0.031 [-0.047, -0.017] * | -0.010 [-0.016, -0.005] * | -0.0201 [-0.0254, -0.0048] * | 12 | 20 | 0.215 | 36 |
| ALL | +0.005 [+0.002, +0.008] * | -0.056 [-0.066, -0.048] * | -0.014 [-0.017, -0.011] * | -0.0227 [-0.0252, -0.0175] * | 66 | 105 | 0.00354 | 211 |

### Sample of changed answers (`rft` → `new`)

| subset | question | options | gold | rft | new |
|---|---|---|---|---|---|
| tallyqa | How many people are in this picture? | A=0; B=1; C=2; D=3; E=4; F=5; G=6; H=7; I=8; J=9; K=10 | H | G (0.38) ✗ | H (0.44) ✓ |
| tallyqa | How many planes are going right? | A=0; B=1; C=2; D=3; E=4; F=5; G=6; H=7; I=8; J=9; K=10 | E | E (0.45) ✓ | F (0.45) ✗ |
| tallyqa | How many people can be seen? | A=0; B=1; C=2; D=3; E=4; F=5; G=6; H=7; I=8; J=9; K=10 | F | E (0.47) ✗ | F (0.58) ✓ |
| tallyqa | How many people are there? | A=0; B=1; C=2; D=3; E=4; F=5; G=6; H=7; I=8; J=9; K=10 | B | B (0.46) ✓ | D (0.45) ✗ |
| vqav2 | How many people are visible behind the car? | A=0; B=1; C=2; D=3; E=4; F=5; G=6; H=7; I=8; J=9; K=10 | B | B (0.49) ✓ | C (0.56) ✗ |
| vqav2 | Is this photo in black and white? | A=Yes; B=No | B | B (0.56) ✓ | A (0.50) ✗ |
| nlvr2 | The first image is the image on the left, the second image is the image on the right. Give | A=Yes; B=No | B | A (0.82) ✗ | B (0.56) ✓ |
| nlvr2 | The first image is the image on the left, the second image is the image on the right. Eval | A=Yes; B=No | A | B (0.68) ✗ | A (0.50) ✓ |
| tallyqa | How many dining tables are visible? | A=0; B=1; C=2; D=3; E=4; F=5; G=6; H=7; I=8; J=9; K=10 | C | B (0.49) ✗ | C (0.52) ✓ |
| tallyqa | How many lights are above the bed? | A=0; B=1; C=2; D=3; E=4; F=5; G=6; H=7; I=8; J=9; K=10 | C | C (0.46) ✓ | D (0.50) ✗ |
| nlvr2 | The first image is the image on the left, the second image is the image on the right. For  | A=Yes; B=No | B | B (0.73) ✓ | A (0.82) ✗ |
| nlvr2 | The first image is the image on the left, the second image is the image on the right. Exam | A=Yes; B=No | B | B (0.53) ✓ | A (0.85) ✗ |
| vqav2 | Is there an alkaline solution in the sink? | A=Yes; B=No | B | B (0.75) ✓ | A (0.50) ✗ |
| tallyqa | How many apples are there? | A=0; B=1; C=2; D=3; E=4; F=5; G=6; H=7; I=8; J=9; K=10 | I | G (0.22) ✗ | J (0.30) ✗ |
| tallyqa | How many people are in the photo? | A=0; B=1; C=2; D=3; E=4; F=5; G=6; H=7; I=8; J=9; K=10 | J | I (0.43) ✗ | J (0.69) ✓ |
| tallyqa | How many piles of meat are there on the table? | A=0; B=1; C=2; D=3; E=4; F=5; G=6; H=7; I=8; J=9; K=10 | E | E (0.43) ✓ | F (0.35) ✗ |
| nlvr2 | The first image is the image on the left, the second image is the image on the right. For  | A=Yes; B=No | B | B (0.56) ✓ | A (0.59) ✗ |
| nlvr2 | The first image is the image on the left, the second image is the image on the right. Anal | A=Yes; B=No | B | A (0.68) ✗ | B (0.53) ✓ |
| vqav2 | Are both this person's plaid bags the same color? | A=Yes; B=No | B | A (0.71) ✗ | B (0.53) ✓ |
| nlvr2 | The first image is the image on the left, the second image is the image on the right. Give | A=Yes; B=No | B | A (0.59) ✗ | B (0.56) ✓ |
| tallyqa | How many cups are there? | A=0; B=1; C=2; D=3; E=4; F=5; G=6; H=7; I=8; J=9; K=10 | C | B (0.49) ✗ | C (0.66) ✓ |
| clevr | Is the number of things that are in front of the red rubber object greater than the number | A=Yes; B=No | A | A (0.88) ✓ | B (0.65) ✗ |
| tallyqa | How many cars can be seen? | A=0; B=1; C=2; D=3; E=4; F=5; G=6; H=7; I=8; J=9; K=10 | F | E (0.39) ✗ | F (0.35) ✓ |
| tallyqa | How many airplanes are there? | A=0; B=1; C=2; D=3; E=4; F=5; G=6; H=7; I=8; J=9; K=10 | C | D (0.58) ✗ | C (0.51) ✓ |
| tallyqa | How many cars are parked? | A=0; B=1; C=2; D=3; E=4; F=5; G=6; H=7; I=8; J=9; K=10 | C | C (0.82) ✓ | B (0.70) ✗ |
