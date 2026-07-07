# Agentic Feature Engineering Module (`src/model/agentic_features`)

## Overview

The **Agentic Feature Engineering** module represents an autonomous, AI-designed approach to astronomical time-series characterization. Designed by **Antigravity (Gemini 3.1 Pro Agent)** without referencing or examining any existing model embedding scripts (e.g., Chronos, Moirai, ASTROMER), this suite introduces **36 novel, physically and statistically motivated features per photometric band**.

While traditional libraries ([FATS](../handcrafted_features/README.md#L13-L40) and [light_curve](../handcrafted_features/README.md#L41-L77)) focus heavily on Lomb-Scargle periodograms and basic statistical moments, our Agentic Feature Suite targets complex time-domain behaviors, non-linear trend scattering, error-weighted distributions, autocorrelation decay timescales, and morphology asymmetries.

When concatenated across two bands (e.g., $g$ and $r$), the pipeline produces a **72-dimensional feature vector** per star, fully compatible with the StarEmbed benchmark classification and clustering suite.

---

## 36 Agent-Engineered Features Computed in `generated_features.py`

The 36 features are structured across 6 orthogonal physical/statistical dimensions:

### 1. Morphology & Outburst Asymmetry (6 features)
Designed to separate asymmetric pulsating stars (RR Lyrae, Cepheids) and flare events from symmetric eclipsing binaries and sinusoidal variables.
* **`asymmetry_rise_decay_ratio`**: Ratio of average rising slope $(\Delta\text{mag}/\Delta t > 0)$ to average falling slope.
* **`asymmetry_rise_decay_time_ratio`**: Ratio of time spent rising vs. time spent decaying.
* **`outlier_asymmetry_index`**: Normalized difference between bright $>3\sigma$ outliers and faint $>3\sigma$ outliers: $(N_{\text{bright}} - N_{\text{faint}}) / N$.
* **`flux_percentile_ratio_95_05`**: $(Q_{95} - Q_{05}) / Q_{50}$. Measures overall outburst/eclipse amplitude relative to median brightness.
* **`flux_percentile_ratio_90_10`**: $(Q_{90} - Q_{10}) / Q_{50}$. Core amplitude ratio robust to extreme tail outliers.
* **`peak_to_trough_sharpness_ratio`**: Ratio of second-derivative curvature around local maxima versus local minima. Eclipsing binaries exhibit sharp faint troughs and flat bright peaks.

### 2. Autocorrelation & Independent Timescales (6 features)
Provides characteristic variability timescales independently of Lomb-Scargle periodograms, excelling on stochastic or multi-periodic sources.
* **`autocor_decay_time_50`**: Time lag $\tau$ at which the empirical autocorrelation function (ACF) first drops below $0.5$.
* **`autocor_decay_time_20`**: Time lag $\tau$ where ACF drops below $0.2$.
* **`autocor_first_zero`**: Time lag $\tau$ where ACF first crosses zero.
* **`autocor_integral`**: Integrated area under the ACF curve from lag $0$ to the first zero crossing (integral timescale).
* **`median_time_between_maxima`**: Median time gap between smoothed local peaks.
* **`median_time_between_minima`**: Median time gap between smoothed local troughs.

### 3. Noise-Robust & Error-Weighted Statistics (6 features)
Standard moments treat all observations equally, allowing noisy measurements with massive uncertainty $(\sigma_i)$ to distort feature representations. These features weight observations by inverse variance: $w_i = 1 / (\sigma_i^2 + \epsilon)$.
* **`error_weighted_mean`**: Inverse-variance weighted mean magnitude.
* **`error_weighted_std`**: Weighted standard deviation.
* **`error_weighted_skewness`**: Weighted skewness.
* **`error_weighted_kurtosis`**: Weighted excess kurtosis.
* **`error_weighted_amplitude`**: Magnitude difference between weighted 95th and 5th percentiles.
* **`signal_to_noise_variability`**: Ratio of observed magnitude standard deviation to average measurement error: $\text{Std}(\text{mag}) / \text{Mean}(\sigma_i)$.

### 4. Trend, Scatter & Non-Linearity (6 features)
Quantifies intrinsic stellar variability beyond instrumental noise and evaluates polynomial/monotonic trends.
* **`robust_scatter_around_quad_trend`**: Median Absolute Deviation (MAD) of residuals from a quadratic polynomial fit divided by median observational error.
* **`quad_trend_curvature`**: Second-order coefficient $(c_2)$ of the quadratic fit $c_2 t^2 + c_1 t + c_0$, normalized by time span.
* **`robust_scatter_around_rolling_median`**: MAD of residuals from a 5-point rolling median divided by median error.
* **`max_deviation_from_median`**: Maximum absolute residual relative to rolling MAD (detects isolated extreme flares/dips).
* **`consecutive_difference_autocorrelation`**: Autocorrelation of first differences $\Delta \text{mag}$ at lag 1. Distinguishes random walk behavior from mean-reverting noise.
* **`trend_monotonicity_index`**: Spearman rank correlation coefficient between observation timestamp and magnitude.

### 5. High-Frequency Jitter, Slopes & Entropy (6 features)
Evaluates information content and derivative distributions to distinguish smooth periodic signals from flickering accretion disks or white noise.
* **`magnitude_shannon_entropy`**: Shannon entropy of a 10-bin magnitude histogram. High entropy indicates chaotic/uniform variation; low entropy indicates stable baseline brightness.
* **`slope_shannon_entropy`**: Shannon entropy of a 10-bin histogram of time derivatives $(\Delta \text{mag} / \Delta t)$.
* **`skewness_of_slopes`**: Skewness of time derivatives. Eclipsing binaries exhibit heavy negative skewness due to rapid eclipse entry/exit.
* **`kurtosis_of_slopes`**: Kurtosis of time derivatives (measures sudden spike/dip frequency).
* **`high_frequency_noise_ratio`**: Ratio of variance of first differences to overall variance: $\text{Var}(\Delta \text{mag}) / \text{Var}(\text{mag})$. Approaches $2.0$ for white noise, $<0.5$ for smooth periodic signals.
* **`cumulative_variation_index`**: Sum of absolute consecutive differences normalized by amplitude and length: $\frac{\sum |\Delta \text{mag}_i|}{N \cdot \text{Range}}$.

### 6. Dwell States, Cadence & Quantiles (6 features)
Characterizes how long the star lingers in bright versus faint states and accounts for survey cadence density.
* **`fraction_time_in_faint_state`**: Fraction of observations within $10\%$ of total amplitude from faintest magnitude.
* **`fraction_time_in_bright_state`**: Fraction of observations within $10\%$ of total amplitude from brightest magnitude.
* **`dwell_time_ratio`**: Ratio of bright state dwell time to faint state dwell time. Eclipsing binaries spend $>80\%$ of time in the bright state; pulsating stars linger in faint states.
* **`time_span_coverage_density`**: Cadence density: $N / (t_{\text{max}} - t_{\text{min}})$.
* **`cadence_irregularity_index`**: Relative dispersion of sampling intervals: $\text{Std}(\Delta t) / \text{Mean}(\Delta t)$.
* **`quantile_dispersion_ratio`**: Tail dispersion ratio: $(Q_{80} - Q_{20}) / (Q_{90} - Q_{10})$.

---

## Pipeline Usage & Execution

### 1. Extract Features from Dataset
Run `extract_agentic_feats.py` to compute features across any HuggingFace dataset split. The script utilizes CPU multiprocessing and automatically generates both a CSV report and a standardized HuggingFace Dataset ready for benchmarking.

```bash
python extract_agentic_feats.py \
    --split validation \
    --dataset_path /path/to/input_dataset \
    --output_path /path/to/output_directory \
    --num_workers 4
```

#### Arguments
* `--split`: Split to process (`train`, `validation`, `test`, `anom`, or `all`).
* `--dataset_path`: Path to input HuggingFace dataset directory on disk.
* `--output_path`: Directory path where extracted CSV and HuggingFace dataset will be saved.
* `--num_workers`: Number of parallel CPU processes (default: `4`).
* `--bands_to_process`: Optional comma-separated list of bands (e.g., `"g,r"`). Automatically detected if omitted.

### 2. Evaluate with Benchmark Suite
Because `extract_agentic_feats.py` automatically builds `embeddings_g`, `embeddings_r`, and a concatenated `combined_embedding` column, the output is immediately compatible with all evaluation scripts in `src/benchmark`:

#### Linear Classifier (Logistic Regression & kNN)
```bash
python ../../benchmark/classification/linear_knn.py \
    --input_embs /path/to/output_directory/validation \
    --scenario concat
```

#### Random Forest with Hyperparameter Optimization
```bash
python ../../benchmark/classification/rf_hpo.py \
    --input-embs /path/to/output_directory/validation \
    --skip-hpo
```
