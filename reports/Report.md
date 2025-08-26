# Report - Verified with Error Bars

We have divided results into two high-performing user-scenarios and low-performing user-scenarios from models where the threshold was chosen on training data, and no sampling techniques were used (most representative of raw performance)

- Group A (> 0.70 avg AUC): 7 scenarios  
- Group B (≤ 0.70 avg AUC): 38 scenarios

We also analyze unbalanced scenarios (≤0.25 or ≥0.75) within each group.

# Group A

## Group A – Sampling Technique Comparison

Compare average train-set ROC AUC across sampling methods and pipelines.

### ROC AUC by Method & Pipeline

| method      | global_ssl | global_supervised | personal_ssl |
|:------------|------------|-------------------|--------------|
| original    | 0.750 ± 0.083 | 0.895 ± 0.039 | 0.901 ± 0.053 |
| oversample  | 0.770 ± 0.048 | 0.884 ± 0.040 | 0.921 ± 0.048 |
| undersample | 0.767 ± 0.080 | 0.883 ± 0.040 | 0.926 ± 0.047 |

### Overall ROC AUC by Method

| method      | ROC_AUC_mean |
|:------------|--------------|
| original    | 0.849 ± 0.035 |
| oversample  | 0.858 ± 0.026 |
| undersample | 0.859 ± 0.034 |

## Group A: Overall Performance by Pipeline (original only)

| Pipeline          | Accuracy | Sensitivity | Specificity | ROC_AUC | AvgSamples |
|:------------------|----------|-------------|-------------|---------|------------|
| global_ssl        | 0.585 ± 0.056 | 0.798 ± 0.065 | 0.369 ± 0.112 | 0.750 ± 0.083 | 253.429 |
| global_supervised | 0.631 ± 0.098 | 0.807 ± 0.118 | 0.434 ± 0.110 | 0.895 ± 0.039 | 253.429 |
| personal_ssl      | 0.801 ± 0.059 | 0.770 ± 0.082 | 0.695 ± 0.064 | 0.901 ± 0.053 | 32.857 |

### Group A – Severely Imbalanced Performance by Pipeline

| Pipeline          | Accuracy | Sensitivity | Specificity | ROC_AUC |
|:------------------|----------|-------------|-------------|---------|
| global_ssl        | 0.391 ± 0.000 | 0.958 ± 0.000 | 0.112 ± 0.000 | 0.774 ± 0.000 |
| global_supervised | 0.618 ± 0.000 | 0.707 ± 0.000 | 0.560 ± 0.000 | 0.780 ± 0.000 |
| personal_ssl      | 0.961 ± 0.000 | 0.504 ± 0.000 | 0.959 ± 0.000 | 0.980 ± 0.000 |

### Group A – Balanced Performance by Pipeline

| Pipeline          | Accuracy | Sensitivity | Specificity | ROC_AUC |
|:------------------|----------|-------------|-------------|---------|
| global_ssl        | 0.618 ± 0.047 | 0.772 ± 0.056 | 0.412 ± 0.111 | 0.745 ± 0.090 |
| global_supervised | 0.633 ± 0.104 | 0.824 ± 0.095 | 0.412 ± 0.113 | 0.914 ± 0.017 |
| personal_ssl      | 0.774 ± 0.032 | 0.814 ± 0.048 | 0.651 ± 0.021 | 0.888 ± 0.033 |

## Group A – Task-Specific Analysis (Crave vs Use)

### Crave (3 scenarios → 9 evals)
Overall pos-rate: 40.4% (± 13.6%)

### Crave Aggregate Performance by Pipeline

| Pipeline          | Accuracy | Sensitivity | Specificity | ROC_AUC |
|:------------------|----------|-------------|-------------|---------|
| global_ssl        | 0.691 ± 0.013 | 0.700 ± 0.074 | 0.575 ± 0.089 | 0.776 ± 0.113 |
| global_supervised | 0.711 ± 0.053 | 0.804 ± 0.067 | 0.538 ± 0.076 | 0.900 ± 0.024 |
| personal_ssl      | 0.820 ± 0.069 | 0.701 ± 0.117 | 0.709 ± 0.092 | 0.882 ± 0.083 |

**Severely Imbalanced Crave (1 scenarios)**

### Crave Severely Imbalanced Performance

| Pipeline     | Accuracy | Sensitivity | Specificity | ROC_AUC |
|:-------------|----------|-------------|-------------|---------|
| personal_ssl | 0.961 ± 0.000 | 0.504 ± 0.000 | 0.959 ± 0.000 | 0.980 ± 0.000 |

### Use (4 scenarios → 12 evals)
Overall pos-rate: 40.7% (± 8.9%)

### Use Aggregate Performance by Pipeline

| Pipeline          | Accuracy | Sensitivity | Specificity | ROC_AUC |
|:------------------|----------|-------------|-------------|---------|
| global_ssl        | 0.506 ± 0.083 | 0.872 ± 0.041 | 0.214 ± 0.115 | 0.730 ± 0.031 |
| global_supervised | 0.570 ± 0.147 | 0.810 ± 0.167 | 0.356 ± 0.152 | 0.891 ± 0.055 |
| personal_ssl      | 0.786 ± 0.050 | 0.822 ± 0.026 | 0.684 ± 0.033 | 0.916 ± 0.022 |

**Severely Imbalanced Use (1 scenarios)**

### Use Severely Imbalanced Performance

| Pipeline          | Accuracy | Sensitivity | Specificity | ROC_AUC |
|:------------------|----------|-------------|-------------|---------|
| global_ssl        | 0.391 ± 0.000 | 0.958 ± 0.000 | 0.112 ± 0.000 | 0.774 ± 0.000 |
| global_supervised | 0.618 ± 0.000 | 0.707 ± 0.000 | 0.560 ± 0.000 | 0.780 ± 0.000 |

## Group A – High-AUC (> 0.7) Deep-Dive: 21 evals across 7 scenarios

### High-AUC Scenarios by Balance Category

| Balance   | Count |
|:----------|-------|
| Balanced  | 6 |
| Severe    | 1 |

Sample size statistics for AUC>0.70:
- Mean samples: 107.9
- Std samples: 110.8
- ≥30 samples (%): 66.7

## Group A – Scenarios List with Class Ratio

### Unbalanced (<25% or >75%)
- ID13 | Almond_Use | global_ssl | 23.84%
- ID13 | Almond_Use | global_supervised | 23.84%
- ID25 | Carrot_Crave | personal_ssl | 7.84%

### Normal (25%–75%)
- ID28 | Coffee_Use | global_ssl | 40.00%
- ID28 | Coffee_Use | global_supervised | 40.00%
- ID28 | Coffee_Use | personal_ssl | 40.00%
- ID13 | Nectarine_Use | global_ssl | 46.48%
- ID13 | Nectarine_Use | global_supervised | 46.48%
- ID13 | Nectarine_Use | personal_ssl | 48.53%
- ID13 | Almond_Use | personal_ssl | 35.90%
- ID12 | Nectarine_Use | global_ssl | 46.48%
- ID12 | Nectarine_Use | global_supervised | 46.48%
- ID12 | Nectarine_Use | personal_ssl | 50.00%
- ID12 | Nectarine_Crave | global_ssl | 48.34%
- ID12 | Nectarine_Crave | global_supervised | 48.34%
- ID12 | Nectarine_Crave | personal_ssl | 50.00%
- ID25 | Carrot_Crave | global_ssl | 34.39%
- ID25 | Carrot_Crave | global_supervised | 34.39%
- ID20 | Nectarine_Crave | global_ssl | 48.34%
- ID20 | Nectarine_Crave | global_supervised | 48.34%
- ID20 | Nectarine_Crave | personal_ssl | 43.75%

# Group B

## Group B – Sampling Technique Comparison

Compare average train-set ROC AUC across sampling methods and pipelines.

### ROC AUC by Method & Pipeline

| method      | global_ssl | global_supervised | personal_ssl |
|:------------|------------|-------------------|--------------|
| original    | 0.535 ± 0.018 | 0.536 ± 0.026 | 0.564 ± 0.024 |
| oversample  | 0.555 ± 0.018 | 0.526 ± 0.023 | 0.568 ± 0.021 |
| undersample | 0.518 ± 0.019 | 0.540 ± 0.025 | 0.576 ± 0.025 |

### Overall ROC AUC by Method

| method      | ROC_AUC_mean |
|:------------|--------------|
| original    | 0.545 ± 0.013 |
| oversample  | 0.549 ± 0.012 |
| undersample | 0.545 ± 0.013 |

## Group B: Overall Performance by Pipeline (original only)

| Pipeline          | Accuracy | Sensitivity | Specificity | ROC_AUC | AvgSamples |
|:------------------|----------|-------------|-------------|---------|------------|
| global_ssl        | 0.498 ± 0.021 | 0.608 ± 0.025 | 0.328 ± 0.025 | 0.535 ± 0.018 | 260.395 |
| global_supervised | 0.452 ± 0.021 | 0.733 ± 0.031 | 0.173 ± 0.019 | 0.536 ± 0.026 | 260.395 |
| personal_ssl      | 0.567 ± 0.026 | 0.525 ± 0.027 | 0.510 ± 0.029 | 0.564 ± 0.024 | 46.421 |

### Group B – Severely Imbalanced Performance by Pipeline

| Pipeline          | Accuracy | Sensitivity | Specificity | ROC_AUC |
|:------------------|----------|-------------|-------------|---------|
| global_ssl        | 0.446 ± 0.062 | 0.439 ± 0.097 | 0.352 ± 0.070 | 0.491 ± 0.041 |
| global_supervised | 0.468 ± 0.078 | 0.393 ± 0.074 | 0.398 ± 0.075 | 0.465 ± 0.048 |
| personal_ssl      | 0.600 ± 0.044 | 0.362 ± 0.044 | 0.630 ± 0.045 | 0.522 ± 0.037 |

### Group B – Balanced Performance by Pipeline

| Pipeline          | Accuracy | Sensitivity | Specificity | ROC_AUC |
|:------------------|----------|-------------|-------------|---------|
| global_ssl        | 0.508 ± 0.021 | 0.639 ± 0.024 | 0.324 ± 0.023 | 0.544 ± 0.018 |
| global_supervised | 0.449 ± 0.017 | 0.797 ± 0.025 | 0.131 ± 0.015 | 0.549 ± 0.025 |
| personal_ssl      | 0.551 ± 0.033 | 0.600 ± 0.032 | 0.455 ± 0.038 | 0.583 ± 0.027 |

## Group B – Task-Specific Analysis (Crave vs Use)

### Crave (17 scenarios → 51 evals)
Overall pos-rate: 33.4% (± 14.7%)

### Crave Aggregate Performance by Pipeline

| Pipeline          | Accuracy | Sensitivity | Specificity | ROC_AUC |
|:------------------|----------|-------------|-------------|---------|
| global_ssl        | 0.516 ± 0.039 | 0.575 ± 0.041 | 0.385 ± 0.041 | 0.533 ± 0.029 |
| global_supervised | 0.440 ± 0.031 | 0.657 ± 0.046 | 0.262 ± 0.030 | 0.563 ± 0.038 |
| personal_ssl      | 0.579 ± 0.033 | 0.525 ± 0.036 | 0.522 ± 0.037 | 0.546 ± 0.032 |

**Severely Imbalanced Crave (7 scenarios)**

### Crave Severely Imbalanced Performance

| Pipeline          | Accuracy | Sensitivity | Specificity | ROC_AUC |
|:------------------|----------|-------------|-------------|---------|
| global_ssl        | 0.656 ± 0.033 | 0.192 ± 0.071 | 0.714 ± 0.032 | 0.647 ± 0.027 |
| global_supervised | 0.611 ± 0.055 | 0.192 ± 0.071 | 0.672 ± 0.054 | 0.490 ± 0.100 |
| personal_ssl      | 0.633 ± 0.066 | 0.351 ± 0.063 | 0.657 ± 0.069 | 0.511 ± 0.053 |

### Use (21 scenarios → 63 evals)
Overall pos-rate: 38.3% (± 13.6%)

### Use Aggregate Performance by Pipeline

| Pipeline          | Accuracy | Sensitivity | Specificity | ROC_AUC |
|:------------------|----------|-------------|-------------|---------|
| global_ssl        | 0.484 ± 0.023 | 0.634 ± 0.035 | 0.282 ± 0.034 | 0.537 ± 0.024 |
| global_supervised | 0.462 ± 0.029 | 0.795 ± 0.042 | 0.101 ± 0.022 | 0.513 ± 0.037 |
| personal_ssl      | 0.557 ± 0.038 | 0.525 ± 0.040 | 0.501 ± 0.044 | 0.579 ± 0.036 |

**Severely Imbalanced Use (7 scenarios)**

### Use Severely Imbalanced Performance

| Pipeline          | Accuracy | Sensitivity | Specificity | ROC_AUC |
|:------------------|----------|-------------|-------------|---------|
| global_ssl        | 0.341 ± 0.096 | 0.562 ± 0.146 | 0.171 ± 0.083 | 0.412 ± 0.057 |
| global_supervised | 0.397 ± 0.118 | 0.493 ± 0.096 | 0.262 ± 0.099 | 0.453 ± 0.042 |
| personal_ssl      | 0.555 ± 0.055 | 0.377 ± 0.055 | 0.592 ± 0.055 | 0.538 ± 0.056 |

## Group B – Scenarios List with Class Ratio

### Unbalanced (<25% or >75%)
- ID9 | Melon_Crave | personal_ssl | 4.00%
- ID5 | Melon_Crave | personal_ssl | 12.16%
- ID19 | Almond_Crave | global_ssl | 12.37%
- ID19 | Almond_Crave | global_supervised | 12.37%
- ID19 | Almond_Crave | personal_ssl | 20.69%
- ID19 | Melon_Crave | personal_ssl | 24.00%
- ID19 | Almond_Use | global_ssl | 23.84%
- ID19 | Almond_Use | global_supervised | 23.84%
- ID19 | Almond_Use | personal_ssl | 15.15%
- ID28 | Almond_Use | global_ssl | 23.84%
- ID28 | Almond_Use | global_supervised | 23.84%
- ID10 | Carrot_Crave | personal_ssl | 22.92%
- ID13 | Carrot_Use | personal_ssl | 5.26%
- ID26 | Carrot_Use | personal_ssl | 8.20%
- ID11 | Almond_Use | global_ssl | 23.84%
- ID11 | Almond_Use | global_supervised | 23.84%
- ID25 | Almond_Crave | global_ssl | 12.37%
- ID25 | Almond_Crave | global_supervised | 12.37%
- ID25 | Almond_Crave | personal_ssl | 3.28%
- ID25 | Almond_Use | global_ssl | 23.84%
- ID25 | Almond_Use | global_supervised | 23.84%
- ID25 | Almond_Use | personal_ssl | 7.55%
- ID20 | Melon_Crave | personal_ssl | 19.05%
- ID20 | Melon_Use | personal_ssl | 18.75%

### Normal (25%–75%)
- ID9 | Melon_Crave | global_ssl | 27.20%
- ID9 | Melon_Crave | global_supervised | 27.20%
- ID5 | Melon_Crave | global_ssl | 27.20%
- ID5 | Melon_Crave | global_supervised | 27.20%
- ID27 | Nectarine_Use | global_ssl | 46.48%
- ID27 | Nectarine_Use | global_supervised | 46.48%
- ID27 | Nectarine_Use | personal_ssl | 53.33%
- ID27 | Melon_Crave | global_ssl | 27.20%
- ID27 | Melon_Crave | global_supervised | 27.20%
- ID27 | Melon_Crave | personal_ssl | 62.86%
- ID27 | Nectarine_Crave | global_ssl | 48.34%
- ID27 | Nectarine_Crave | global_supervised | 48.34%
- ID27 | Nectarine_Crave | personal_ssl | 66.67%
- ID27 | Melon_Use | global_ssl | 54.34%
- ID27 | Melon_Use | global_supervised | 54.34%
- ID27 | Melon_Use | personal_ssl | 72.06%
- ID19 | Melon_Crave | global_ssl | 27.20%
- ID19 | Melon_Crave | global_supervised | 27.20%
- ID19 | Melon_Use | global_ssl | 54.34%
- ID19 | Melon_Use | global_supervised | 54.34%
- ID19 | Melon_Use | personal_ssl | 51.43%
- ID15 | Carrot_Crave | global_ssl | 34.39%
- ID15 | Carrot_Crave | global_supervised | 34.39%
- ID15 | Carrot_Crave | personal_ssl | 52.08%
- ID15 | Carrot_Use | global_ssl | 32.94%
- ID15 | Carrot_Use | global_supervised | 32.94%
- ID15 | Carrot_Use | personal_ssl | 50.00%
- ID28 | Almond_Use | personal_ssl | 52.94%
- ID10 | Nectarine_Use | global_ssl | 46.48%
- ID10 | Nectarine_Use | global_supervised | 46.48%
- ID10 | Nectarine_Use | personal_ssl | 27.91%
- ID10 | Carrot_Crave | global_ssl | 34.39%
- ID10 | Carrot_Crave | global_supervised | 34.39%
- ID10 | Nectarine_Crave | global_ssl | 48.34%
- ID10 | Nectarine_Crave | global_supervised | 48.34%
- ID10 | Nectarine_Crave | personal_ssl | 28.00%
- ID10 | Carrot_Use | global_ssl | 32.94%
- ID10 | Carrot_Use | global_supervised | 32.94%
- ID10 | Carrot_Use | personal_ssl | 29.17%
- ID13 | Carrot_Use | global_ssl | 32.94%
- ID13 | Carrot_Use | global_supervised | 32.94%
- ID14 | Carrot_Crave | global_ssl | 34.39%
- ID14 | Carrot_Crave | global_supervised | 34.39%
- ID14 | Carrot_Crave | personal_ssl | 47.83%
- ID14 | Carrot_Use | global_ssl | 32.94%
- ID14 | Carrot_Use | global_supervised | 32.94%
- ID14 | Carrot_Use | personal_ssl | 48.65%
- ID26 | Carrot_Use | global_ssl | 32.94%
- ID26 | Carrot_Use | global_supervised | 32.94%
- ID18 | Carrot_Crave | global_ssl | 34.39%
- ID18 | Carrot_Crave | global_supervised | 34.39%
- ID18 | Carrot_Crave | personal_ssl | 47.06%
- ID18 | Carrot_Use | global_ssl | 32.94%
- ID18 | Carrot_Use | global_supervised | 32.94%
- ID18 | Carrot_Use | personal_ssl | 41.67%
- ID11 | Nectarine_Use | global_ssl | 46.48%
- ID11 | Nectarine_Use | global_supervised | 46.48%
- ID11 | Nectarine_Use | personal_ssl | 54.84%
- ID11 | Nectarine_Crave | global_ssl | 48.34%
- ID11 | Nectarine_Crave | global_supervised | 48.34%
- ID11 | Nectarine_Crave | personal_ssl | 50.00%
- ID11 | Almond_Use | personal_ssl | 44.44%
- ID11 | Carrot_Use | global_ssl | 32.94%
- ID11 | Carrot_Use | global_supervised | 32.94%
- ID11 | Carrot_Use | personal_ssl | 40.00%
- ID12 | Melon_Crave | global_ssl | 27.20%
- ID12 | Melon_Crave | global_supervised | 27.20%
- ID12 | Melon_Crave | personal_ssl | 45.45%
- ID12 | Melon_Use | global_ssl | 54.34%
- ID12 | Melon_Use | global_supervised | 54.34%
- ID12 | Melon_Use | personal_ssl | 47.69%
- ID12 | GHB_Use | global_ssl | 40.00%
- ID12 | GHB_Use | global_supervised | 40.00%
- ID12 | GHB_Use | personal_ssl | 40.00%
- ID21 | Nectarine_Use | global_ssl | 46.48%
- ID21 | Nectarine_Use | global_supervised | 46.48%
- ID21 | Nectarine_Use | personal_ssl | 48.28%
- ID21 | Melon_Crave | global_ssl | 27.20%
- ID21 | Melon_Crave | global_supervised | 27.20%
- ID21 | Melon_Crave | personal_ssl | 52.94%
- ID21 | Nectarine_Crave | global_ssl | 48.34%
- ID21 | Nectarine_Crave | global_supervised | 48.34%
- ID21 | Nectarine_Crave | personal_ssl | 50.68%
- ID20 | Nectarine_Use | global_ssl | 46.48%
- ID20 | Nectarine_Use | global_supervised | 46.48%
- ID20 | Nectarine_Use | personal_ssl | 27.27%
- ID20 | Melon_Crave | global_ssl | 27.20%
- ID20 | Melon_Crave | global_supervised | 27.20%
- ID20 | Melon_Use | global_ssl | 54.34%
- ID20 | Melon_Use | global_supervised | 54.34%

# All Scenarios

## All Scenarios – Sampling Technique Comparison

Compare average train-set ROC AUC across sampling methods and pipelines.

### ROC AUC by Method & Pipeline

| method      | global_ssl | global_supervised | personal_ssl |
|:------------|------------|-------------------|--------------|
| original    | 0.569 ± 0.021 | 0.591 ± 0.034 | 0.617 ± 0.029 |
| oversample  | 0.588 ± 0.021 | 0.581 ± 0.031 | 0.623 ± 0.027 |
| undersample | 0.557 ± 0.024 | 0.593 ± 0.034 | 0.630 ± 0.029 |

### Overall ROC AUC by Method

| method      | ROC_AUC_mean |
|:------------|--------------|
| original    | 0.592 ± 0.019 |
| oversample  | 0.597 ± 0.016 |
| undersample | 0.594 ± 0.019 |

## All Scenarios: Overall Performance by Pipeline (original only)

| Pipeline          | Accuracy | Sensitivity | Specificity | ROC_AUC | AvgSamples |
|:------------------|----------|-------------|-------------|---------|------------|
| global_ssl        | 0.511 ± 0.023 | 0.631 ± 0.028 | 0.320 ± 0.026 | 0.569 ± 0.021 | 255.043 |
| global_supervised | 0.481 ± 0.027 | 0.734 ± 0.038 | 0.204 ± 0.025 | 0.591 ± 0.034 | 255.043 |
| personal_ssl      | 0.599 ± 0.025 | 0.539 ± 0.026 | 0.538 ± 0.027 | 0.617 ± 0.029 | 42.936 |

### All Scenarios – Severely Imbalanced Performance by Pipeline

| Pipeline          | Accuracy | Sensitivity | Specificity | ROC_AUC |
|:------------------|----------|-------------|-------------|---------|
| global_ssl        | 0.445 ± 0.056 | 0.510 ± 0.092 | 0.278 ± 0.066 | 0.531 ± 0.042 |
| global_supervised | 0.490 ± 0.074 | 0.444 ± 0.078 | 0.369 ± 0.075 | 0.510 ± 0.058 |
| personal_ssl      | 0.628 ± 0.041 | 0.373 ± 0.041 | 0.656 ± 0.042 | 0.558 ± 0.036 |

### All Scenarios – Balanced Performance by Pipeline

| Pipeline          | Accuracy | Sensitivity | Specificity | ROC_AUC |
|:------------------|----------|-------------|-------------|---------|
| global_ssl        | 0.525 ± 0.021 | 0.656 ± 0.025 | 0.329 ± 0.024 | 0.576 ± 0.020 |
| global_supervised | 0.479 ± 0.021 | 0.793 ± 0.029 | 0.171 ± 0.023 | 0.606 ± 0.036 |
| personal_ssl      | 0.588 ± 0.032 | 0.603 ± 0.031 | 0.493 ± 0.036 | 0.641 ± 0.030 |

## All Scenarios – Task-Specific Analysis (Crave vs Use)

### Crave (22 scenarios → 66 evals)
Overall pos-rate: 34.4% (± 14.9%)

### Crave Aggregate Performance by Pipeline

| Pipeline          | Accuracy | Sensitivity | Specificity | ROC_AUC |
|:------------------|----------|-------------|-------------|---------|
| global_ssl        | 0.538 ± 0.038 | 0.584 ± 0.039 | 0.376 ± 0.039 | 0.570 ± 0.029 |
| global_supervised | 0.482 ± 0.032 | 0.662 ± 0.045 | 0.276 ± 0.031 | 0.614 ± 0.040 |
| personal_ssl      | 0.605 ± 0.034 | 0.501 ± 0.038 | 0.546 ± 0.037 | 0.597 ± 0.039 |

**Severely Imbalanced Crave (9 scenarios)**

### Crave Severely Imbalanced Performance

| Pipeline          | Accuracy | Sensitivity | Specificity | ROC_AUC |
|:------------------|----------|-------------|-------------|---------|
| global_ssl        | 0.601 ± 0.062 | 0.292 ± 0.106 | 0.476 ± 0.117 | 0.647 ± 0.027 |
| global_supervised | 0.571 ± 0.052 | 0.292 ± 0.106 | 0.448 ± 0.109 | 0.490 ± 0.100 |
| personal_ssl      | 0.674 ± 0.058 | 0.370 ± 0.056 | 0.695 ± 0.060 | 0.570 ± 0.060 |

### Use (25 scenarios → 75 evals)
Overall pos-rate: 38.7% (± 13.0%)

### Use Aggregate Performance by Pipeline

| Pipeline          | Accuracy | Sensitivity | Specificity | ROC_AUC |
|:------------------|----------|-------------|-------------|---------|
| global_ssl        | 0.487 ± 0.028 | 0.673 ± 0.037 | 0.271 ± 0.034 | 0.568 ± 0.026 |
| global_supervised | 0.479 ± 0.037 | 0.797 ± 0.049 | 0.142 ± 0.030 | 0.574 ± 0.047 |
| personal_ssl      | 0.594 ± 0.036 | 0.572 ± 0.036 | 0.530 ± 0.041 | 0.633 ± 0.038 |

**Severely Imbalanced Use (8 scenarios)**

### Use Severely Imbalanced Performance

| Pipeline          | Accuracy | Sensitivity | Specificity | ROC_AUC |
|:------------------|----------|-------------|-------------|---------|
| global_ssl        | 0.351 ± 0.088 | 0.641 ± 0.130 | 0.159 ± 0.074 | 0.485 ± 0.068 |
| global_supervised | 0.441 ± 0.106 | 0.536 ± 0.088 | 0.321 ± 0.085 | 0.518 ± 0.065 |
| personal_ssl      | 0.555 ± 0.055 | 0.377 ± 0.055 | 0.592 ± 0.055 | 0.538 ± 0.056 |

## Recommendations & Conclusions

- **Personal SSL** consistently delivers highest AUC when sample size ≥50 and balance between 25–75%.
- **Global pipelines** trade ~15–25 pts sensitivity for ~15 pts specificity; choose based on recall vs precision needs.
- **Substance effects**: Carrot & Melon suffer most from class imbalance; Nectarine is stable; Almond/Coffee/GHB show high variance.
- **Key drivers of high AUC**: moderate class balance plus sufficient sample size—no single pipeline or substance universally dominates.

## Note on Error Bars
All error values represent standard errors calculated using error propagation from the original standard deviations in the data. For aggregated metrics, the standard error was calculated as: SE = sqrt(sum(std_i^2))/n, where std_i are the standard deviations of individual measurements and n is the number of measurements.