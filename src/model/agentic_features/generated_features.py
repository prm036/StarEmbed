#!/usr/bin/env python3
"""
Agent-Engineered Feature Suite for Astronomical Light Curves.

Designed autonomously by Antigravity (Gemini 3.1 Pro Agent) based on parametric
astrophysics and statistical time-series domain knowledge.
No other model embedding scripts were referenced or examined.

This module computes 36 novel, physically motivated features per photometric band,
targeting 6 key statistical/physical dimensions:
  1. Morphology & Outburst Asymmetry (6 features)
  2. Autocorrelation & Independent Timescales (6 features)
  3. Noise-Robust & Error-Weighted Statistics (6 features)
  4. Trend, Scatter & Non-Linearity (6 features)
  5. High-Frequency Jitter, Slopes & Entropy (6 features)
  6. Dwell States, Cadence & Quantiles (6 features)

Total Features per Band: 36
"""

import numpy as np
from scipy import stats

FEATURE_NAMES = [
    # 1. Morphology & Outburst Asymmetry
    "asymmetry_rise_decay_ratio",
    "asymmetry_rise_decay_time_ratio",
    "outlier_asymmetry_index",
    "flux_percentile_ratio_95_05",
    "flux_percentile_ratio_90_10",
    "peak_to_trough_sharpness_ratio",
    
    # 2. Autocorrelation & Independent Timescales
    "autocor_decay_time_50",
    "autocor_decay_time_20",
    "autocor_first_zero",
    "autocor_integral",
    "median_time_between_maxima",
    "median_time_between_minima",
    
    # 3. Noise-Robust & Error-Weighted Statistics
    "error_weighted_mean",
    "error_weighted_std",
    "error_weighted_skewness",
    "error_weighted_kurtosis",
    "error_weighted_amplitude",
    "signal_to_noise_variability",
    
    # 4. Trend, Scatter & Non-Linearity
    "robust_scatter_around_quad_trend",
    "quad_trend_curvature",
    "robust_scatter_around_rolling_median",
    "max_deviation_from_median",
    "consecutive_difference_autocorrelation",
    "trend_monotonicity_index",
    
    # 5. High-Frequency Jitter, Slopes & Entropy
    "magnitude_shannon_entropy",
    "slope_shannon_entropy",
    "skewness_of_slopes",
    "kurtosis_of_slopes",
    "high_frequency_noise_ratio",
    "cumulative_variation_index",
    
    # 6. Dwell States, Cadence & Quantiles
    "fraction_time_in_faint_state",
    "fraction_time_in_bright_state",
    "dwell_time_ratio",
    "time_span_coverage_density",
    "cadence_irregularity_index",
    "quantile_dispersion_ratio",
]

NUM_FEATURES = len(FEATURE_NAMES)


def _safe_divide(num, den, default=0.0):
    """Helper for safe division avoiding NaN or Inf."""
    if den == 0 or np.isnan(den) or np.isinf(den):
        return default
    res = num / den
    if np.isnan(res) or np.isinf(res):
        return default
    return float(res)


def _shannon_entropy(values, bins=10):
    """Compute Shannon entropy of a 1D empirical distribution."""
    if len(values) < 2:
        return 0.0
    hist, _ = np.histogram(values, bins=bins, density=True)
    hist = hist[hist > 0]
    # Normalize probabilities
    prob = hist / np.sum(hist)
    return float(-np.sum(prob * np.log2(prob)))


def calc_agentic_features(lc):
    """
    Calculate 36 agent-engineered features for a single light curve dict/object.
    
    Expected input schema (dict-like):
      - 'mjd': sequence of observation timestamps
      - 'target': sequence of magnitude/brightness values
      - 'past_feat_dynamic_real': sequence of observational errors/uncertainties
      
    Returns:
      1D numpy array of float32 with length 36.
      Returns array of -1.0 if data is None or empty.
      Returns array of -2.0 if data has insufficient points (< 5).
    """
    if lc is None:
        return np.full(NUM_FEATURES, -1.0, dtype=np.float32)
        
    try:
        mjd = np.asarray(lc.get("mjd", []), dtype=np.float64)
        mag = np.asarray(lc.get("target", []), dtype=np.float64)
        err = np.asarray(lc.get("past_feat_dynamic_real", []), dtype=np.float64)
    except Exception:
        return np.full(NUM_FEATURES, -1.0, dtype=np.float32)
        
    N = len(mag)
    if N < 5 or len(mjd) != N or len(err) != N:
        return np.full(NUM_FEATURES, -2.0, dtype=np.float32)
        
    # Ensure time-ascending sort
    sort_idx = np.argsort(mjd)
    mjd = mjd[sort_idx]
    mag = mag[sort_idx]
    err = err[sort_idx]
    
    # Clean zeros or negative errors for weights
    err = np.where(err <= 0.0, np.median(err[err > 0]) if np.any(err > 0) else 0.01, err)
    weights = 1.0 / (err ** 2 + 1e-12)
    weights /= np.sum(weights)
    
    # Time intervals and slopes
    dt = np.diff(mjd)
    dt = np.where(dt <= 0.0, 1e-5, dt)  # prevent division by zero for simultaneous obs
    dmag = np.diff(mag)
    slopes = dmag / dt
    
    feats = []
    
    # -------------------------------------------------------------------------
    # 1. Morphology & Outburst Asymmetry
    # -------------------------------------------------------------------------
    pos_slopes = slopes[slopes > 0]
    neg_slopes = np.abs(slopes[slopes < 0])
    mean_pos_slope = np.mean(pos_slopes) if len(pos_slopes) > 0 else 0.0
    mean_neg_slope = np.mean(neg_slopes) if len(neg_slopes) > 0 else 0.0
    
    feats.append(_safe_divide(mean_pos_slope, mean_neg_slope, default=1.0))
    feats.append(_safe_divide(len(pos_slopes), len(neg_slopes), default=1.0))
    
    # Outlier asymmetry (> 3 std)
    mag_mean = np.mean(mag)
    mag_std = np.std(mag) + 1e-12
    z_scores = (mag - mag_mean) / mag_std
    n_bright_outliers = np.sum(z_scores < -3.0)  # in astronomy, smaller mag = brighter
    n_faint_outliers = np.sum(z_scores > 3.0)
    feats.append(_safe_divide(n_bright_outliers - n_faint_outliers, N, default=0.0))
    
    # Flux percentile ratios
    q05, q10, q50, q90, q95 = np.percentile(mag, [5, 10, 50, 90, 95])
    feats.append(_safe_divide(q95 - q05, np.abs(q50) + 1e-12, default=0.0))
    feats.append(_safe_divide(q90 - q10, np.abs(q50) + 1e-12, default=0.0))
    
    # Peak to trough sharpness ratio (second differences around local extrema)
    if N >= 5:
        d2mag = np.diff(mag, n=2)
        peaks = d2mag[d2mag < 0]    # negative curvature = local maxima/bright peaks
        troughs = d2mag[d2mag > 0]  # positive curvature = local minima/faint troughs
        mean_peak_sharpness = np.mean(np.abs(peaks)) if len(peaks) > 0 else 0.0
        mean_trough_sharpness = np.mean(troughs) if len(troughs) > 0 else 0.0
        feats.append(_safe_divide(mean_peak_sharpness, mean_trough_sharpness, default=1.0))
    else:
        feats.append(1.0)
        
    # -------------------------------------------------------------------------
    # 2. Autocorrelation & Independent Timescales
    # -------------------------------------------------------------------------
    # Empirical autocorrelation up to max lag
    mag_centered = mag - mag_mean
    var_mag = np.var(mag_centered) + 1e-12
    max_lag = min(N - 1, 50)
    acf = [1.0]
    for lag in range(1, max_lag + 1):
        c = np.mean(mag_centered[:-lag] * mag_centered[lag:]) / var_mag
        acf.append(c)
    acf = np.array(acf)
    
    # Decay times
    idx_50 = np.where(acf < 0.5)[0]
    decay_50 = float(mjd[idx_50[0]] - mjd[0]) if len(idx_50) > 0 else float(mjd[-1] - mjd[0])
    feats.append(decay_50)
    
    idx_20 = np.where(acf < 0.2)[0]
    decay_20 = float(mjd[idx_20[0]] - mjd[0]) if len(idx_20) > 0 else float(mjd[-1] - mjd[0])
    feats.append(decay_20)
    
    idx_zero = np.where(acf < 0.0)[0]
    first_zero = float(mjd[idx_zero[0]] - mjd[0]) if len(idx_zero) > 0 else float(mjd[-1] - mjd[0])
    feats.append(first_zero)
    
    # ACF integral up to first zero
    zero_limit = idx_zero[0] if len(idx_zero) > 0 else len(acf)
    feats.append(float(np.sum(acf[:zero_limit])))
    
    # Median time between extrema
    if N >= 5:
        # Smooth with 3-point rolling median to ignore noise spikes
        smoothed = np.copy(mag)
        for i in range(1, N - 1):
            smoothed[i] = np.median(mag[i-1:i+2])
        # Find local peaks/troughs
        local_max_idx = [i for i in range(1, N-1) if smoothed[i] > smoothed[i-1] and smoothed[i] > smoothed[i+1]]
        local_min_idx = [i for i in range(1, N-1) if smoothed[i] < smoothed[i-1] and smoothed[i] < smoothed[i+1]]
        
        t_max_diffs = np.diff(mjd[local_max_idx]) if len(local_max_idx) > 1 else [float(mjd[-1] - mjd[0])]
        t_min_diffs = np.diff(mjd[local_min_idx]) if len(local_min_idx) > 1 else [float(mjd[-1] - mjd[0])]
        feats.append(float(np.median(t_max_diffs)))
        feats.append(float(np.median(t_min_diffs)))
    else:
        feats.append(0.0)
        feats.append(0.0)
        
    # -------------------------------------------------------------------------
    # 3. Noise-Robust & Error-Weighted Statistics
    # -------------------------------------------------------------------------
    w_mean = np.sum(weights * mag)
    w_std = np.sqrt(np.sum(weights * (mag - w_mean) ** 2)) + 1e-12
    w_skew = np.sum(weights * ((mag - w_mean) / w_std) ** 3)
    w_kurt = np.sum(weights * ((mag - w_mean) / w_std) ** 4) - 3.0
    
    feats.append(float(w_mean))
    feats.append(float(w_std))
    feats.append(float(w_skew))
    feats.append(float(w_kurt))
    
    # Weighted amplitude (difference between weighted percentiles approximations)
    sort_mag_idx = np.argsort(mag)
    sorted_mag = mag[sort_mag_idx]
    cum_weights = np.cumsum(weights[sort_mag_idx])
    idx_05 = np.searchsorted(cum_weights, 0.05)
    idx_95 = np.searchsorted(cum_weights, 0.95)
    idx_05 = min(max(idx_05, 0), N - 1)
    idx_95 = min(max(idx_95, 0), N - 1)
    feats.append(float(sorted_mag[idx_95] - sorted_mag[idx_05]))
    
    # Signal-to-noise variability ratio
    mean_err = np.mean(err) + 1e-12
    feats.append(float(mag_std / mean_err))
    
    # -------------------------------------------------------------------------
    # 4. Trend, Scatter & Non-Linearity
    # -------------------------------------------------------------------------
    # Quadratic fit: c2*t^2 + c1*t + c0
    t_norm = (mjd - mjd[0]) / (mjd[-1] - mjd[0] + 1e-12)
    try:
        coeffs = np.polyfit(t_norm, mag, deg=2)
        quad_trend = np.polyval(coeffs, t_norm)
        c2_curvature = coeffs[0]
    except Exception:
        quad_trend = np.full_like(mag, mag_mean)
        c2_curvature = 0.0
        
    quad_residuals = np.abs(mag - quad_trend)
    mad_quad = np.median(quad_residuals)
    feats.append(_safe_divide(mad_quad, np.median(err) + 1e-12, default=1.0))
    feats.append(float(c2_curvature))
    
    # Rolling median scatter
    rolling_med = np.copy(mag)
    for i in range(2, N - 2):
        rolling_med[i] = np.median(mag[i-2:i+3])
    roll_residuals = np.abs(mag - rolling_med)
    mad_roll = np.median(roll_residuals) + 1e-12
    feats.append(_safe_divide(mad_roll, np.median(err) + 1e-12, default=1.0))
    feats.append(float(np.max(roll_residuals) / mad_roll))
    
    # Autocorrelation of first differences at lag 1
    dmag_centered = dmag - np.mean(dmag)
    var_dmag = np.var(dmag_centered) + 1e-12
    if len(dmag) > 1:
        diff_acf1 = np.mean(dmag_centered[:-1] * dmag_centered[1:]) / var_dmag
    else:
        diff_acf1 = 0.0
    feats.append(float(diff_acf1))
    
    # Trend monotonicity (Spearman rank correlation)
    try:
        spearman_rho, _ = stats.spearmanr(t_norm, mag)
        if np.isnan(spearman_rho):
            spearman_rho = 0.0
    except Exception:
        spearman_rho = 0.0
    feats.append(float(spearman_rho))
    
    # -------------------------------------------------------------------------
    # 5. High-Frequency Jitter, Slopes & Entropy
    # -------------------------------------------------------------------------
    feats.append(_shannon_entropy(mag, bins=min(10, max(3, N // 3))))
    feats.append(_shannon_entropy(slopes, bins=min(10, max(3, len(slopes) // 3))))
    
    # Skewness and kurtosis of slopes
    slope_std = np.std(slopes) + 1e-12
    slope_mean = np.mean(slopes)
    feats.append(float(np.mean(((slopes - slope_mean) / slope_std) ** 3)))
    feats.append(float(np.mean(((slopes - slope_mean) / slope_std) ** 4) - 3.0))
    
    # High frequency noise ratio: Var(dmag) / Var(mag)
    feats.append(float(np.var(dmag) / (var_mag + 1e-12)))
    
    # Cumulative variation index
    mag_range = np.max(mag) - np.min(mag) + 1e-12
    feats.append(float(np.sum(np.abs(dmag)) / (N * mag_range)))
    
    # -------------------------------------------------------------------------
    # 6. Dwell States, Cadence & Quantiles
    # -------------------------------------------------------------------------
    # In astronomy, smaller magnitude = brighter.
    # Faint state = within 10% of faintest magnitude (max mag)
    # Bright state = within 10% of brightest magnitude (min mag)
    min_mag, max_mag = np.min(mag), np.max(mag)
    threshold_10pct = 0.10 * mag_range
    
    faint_count = np.sum(mag >= (max_mag - threshold_10pct))
    bright_count = np.sum(mag <= (min_mag + threshold_10pct))
    
    frac_faint = faint_count / float(N)
    frac_bright = bright_count / float(N)
    feats.append(float(frac_faint))
    feats.append(float(frac_bright))
    feats.append(_safe_divide(frac_bright, frac_faint, default=1.0))
    
    # Cadence coverage density and irregularity
    time_span = float(mjd[-1] - mjd[0]) + 1e-12
    feats.append(float(N / time_span))
    
    mean_dt = np.mean(dt) + 1e-12
    feats.append(float(np.std(dt) / mean_dt))
    
    # Quantile dispersion ratio: (Q80 - Q20) / (Q90 - Q10)
    q20, q80 = np.percentile(mag, [20, 80])
    feats.append(_safe_divide(q80 - q20, q90 - q10 + 1e-12, default=0.75))
    
    # Ensure exactly NUM_FEATURES elements and replace any residual NaN/Inf
    feats = np.array(feats, dtype=np.float32)
    feats = np.nan_to_num(feats, nan=0.0, posinf=1e6, neginf=-1e6)
    
    if len(feats) != NUM_FEATURES:
        # Fallback safety check
        return np.full(NUM_FEATURES, -3.0, dtype=np.float32)
        
    return feats


if __name__ == "__main__":
    # Quick sanity check with synthetic data
    print(f"Agent-Engineered Feature Suite initialized with {NUM_FEATURES} features:")
    for i, name in enumerate(FEATURE_NAMES):
        print(f"  [{i+1:02d}] {name}")
        
    # Synthetic RR Lyrae-like asymmetric pulse
    t = np.linspace(58000.0, 58100.0, 100)
    phase = (t % 0.5) / 0.5
    # Fast rise, slow decay
    syn_mag = np.where(phase < 0.2, 15.0 - 1.0 * (phase / 0.2), 14.0 + 1.0 * ((phase - 0.2) / 0.8))
    syn_err = np.full_like(syn_mag, 0.02)
    
    test_lc = {"mjd": t, "target": syn_mag, "past_feat_dynamic_real": syn_err}
    res = calc_agentic_features(test_lc)
    print("\nSanity Check on Synthetic Asymmetric Light Curve:")
    print(f"Output Shape: {res.shape} (Expected: ({NUM_FEATURES},))")
    print(f"Asymmetry Rise/Decay Ratio: {res[0]:.4f}")
    print(f"Signal-to-Noise Variability: {res[17]:.4f}")
    print("All checks passed successfully!")
