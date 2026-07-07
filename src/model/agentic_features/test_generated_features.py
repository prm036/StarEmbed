#!/usr/bin/env python3
"""
Verification and Testing Script for Agent-Engineered Features.

Generates synthetic astronomical light curves representing distinct astrophysical classes
(RR Lyrae Pulsators, Eclipsing Binaries, Flare Stars, White Noise, and Sparse Edge Cases)
and verifies that all 36 agent-engineered features compute cleanly without NaNs, infinities,
or runtime exceptions.
"""

import numpy as np
import pandas as pd
from generated_features import calc_agentic_features, FEATURE_NAMES, NUM_FEATURES


def generate_rr_lyrae(n_points=100):
    """Synthetic RR Lyrae pulsating star: fast rise, slow decay."""
    t = np.sort(np.random.uniform(58000.0, 58100.0, n_points))
    period = 0.55
    phase = (t % period) / period
    mag = np.where(
        phase < 0.15,
        15.5 - 1.0 * (phase / 0.15),
        14.5 + 1.0 * ((phase - 0.15) / 0.85)
    )
    err = np.random.uniform(0.01, 0.03, n_points)
    mag += np.random.normal(0, err)
    return {"mjd": t, "target": mag, "past_feat_dynamic_real": err}


def generate_eclipsing_binary(n_points=120):
    """Synthetic Eclipsing Binary: constant bright state with sharp faint dips."""
    t = np.sort(np.random.uniform(58000.0, 58100.0, n_points))
    period = 2.3
    phase = (t % period) / period
    mag = np.full_like(t, 14.0)
    # Primary eclipse dip
    mag = np.where((phase > 0.45) & (phase < 0.55), 14.0 + 1.5 * (1.0 - np.abs(phase - 0.5) / 0.05), mag)
    err = np.random.uniform(0.01, 0.02, n_points)
    mag += np.random.normal(0, err)
    return {"mjd": t, "target": mag, "past_feat_dynamic_real": err}


def generate_flare_star(n_points=100):
    """Synthetic Flare Star: stable faint baseline with sudden bright spikes."""
    t = np.sort(np.random.uniform(58000.0, 58100.0, n_points))
    mag = np.full_like(t, 17.0)
    # Add 3 random flares
    flare_idx = np.random.choice(n_points, 3, replace=False)
    mag[flare_idx] -= np.random.uniform(1.5, 3.0, 3)
    err = np.random.uniform(0.02, 0.05, n_points)
    mag += np.random.normal(0, err)
    return {"mjd": t, "target": mag, "past_feat_dynamic_real": err}


def generate_white_noise(n_points=100):
    """Synthetic pure instrumental white noise."""
    t = np.sort(np.random.uniform(58000.0, 58100.0, n_points))
    mag = np.random.normal(16.0, 0.1, n_points)
    err = np.full_like(t, 0.1)
    return {"mjd": t, "target": mag, "past_feat_dynamic_real": err}


def generate_sparse_edge_case():
    """Sparse light curve with only 6 observations and extreme error bars."""
    t = np.array([58001.0, 58010.5, 58025.2, 58050.1, 58070.8, 58099.3])
    mag = np.array([15.1, 15.3, 14.9, 15.8, 15.2, 15.0])
    err = np.array([0.05, 0.50, 0.02, 1.20, 0.03, 0.04])
    return {"mjd": t, "target": mag, "past_feat_dynamic_real": err}


def main():
    print("====================================================================")
    print("Verifying Agent-Engineered Feature Suite on Synthetic Light Curves")
    print("====================================================================\n")

    test_cases = {
        "RR Lyrae Pulsator": generate_rr_lyrae(),
        "Eclipsing Binary": generate_eclipsing_binary(),
        "Flare Star": generate_flare_star(),
        "White Noise": generate_white_noise(),
        "Sparse Edge Case (6 pts)": generate_sparse_edge_case(),
    }

    results = {}
    for name, lc in test_cases.items():
        print(f"Testing [{name}] (N = {len(lc['target'])} obs)...", end=" ")
        try:
            feats = calc_agentic_features(lc)
            
            # Assertions
            assert len(feats) == NUM_FEATURES, f"Shape mismatch: {len(feats)} != {NUM_FEATURES}"
            assert not np.any(np.isnan(feats)), "Found NaN in feature array!"
            assert not np.any(np.isinf(feats)), "Found Inf in feature array!"
            
            results[name] = feats
            print("PASSED [No NaNs, No Infs, Correct Shape]")
        except Exception as e:
            print(f"FAILED: {e}")
            return 1

    # Display comparative analysis on key discriminative features
    print("\n====================================================================")
    print("Comparative Analysis: How Agentic Features Discriminate Classes")
    print("====================================================================")
    
    df = pd.DataFrame(results, index=FEATURE_NAMES)
    
    key_features = [
        "asymmetry_rise_decay_ratio",
        "outlier_asymmetry_index",
        "autocor_decay_time_50",
        "signal_to_noise_variability",
        "skewness_of_slopes",
        "dwell_time_ratio",
        "high_frequency_noise_ratio",
        "magnitude_shannon_entropy"
    ]
    
    print(df.loc[key_features].round(4).to_string())
    print("\nVerification Complete: All 36 Agent-Engineered Features are robust and ready!")
    return 0


if __name__ == "__main__":
    exit(main())
