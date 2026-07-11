"""
StarFeatureSuite: State-of-the-Art Multi-Band Variable Star Feature Engineering Library
========================================================================================
Architected and mathematically formulated for irregular, heteroskedastic, multi-band 
astronomical time-series (e.g., ZTF, LSST, Gaia, PLAsTiCC).

Author: Autonomous AI Computational Astrophysicist & Time-Series Statistician
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from typing import Dict, Any, List, Optional, Tuple


class CadenceAndSamplingFeatures:
    """Computes time-span, sampling rate, and observational cadence statistics."""
    
    @staticmethod
    def compute(mjd: np.ndarray) -> Dict[str, float]:
        n = len(mjd)
        if n < 2:
            return {
                'n_obs': float(n),
                'time_span': 0.0,
                'cadence_mean': 0.0,
                'cadence_median': 0.0,
                'cadence_std': 0.0,
                'cadence_iqr': 0.0,
                'fill_factor': 0.0
            }
        
        dt = np.diff(np.sort(mjd))
        time_span = float(mjd[-1] - mjd[0]) if n > 1 else 0.0
        
        return {
            'n_obs': float(n),
            'time_span': time_span,
            'cadence_mean': float(np.mean(dt)),
            'cadence_median': float(np.median(dt)),
            'cadence_std': float(np.std(dt)),
            'cadence_iqr': float(np.percentile(dt, 75) - np.percentile(dt, 25)),
            'fill_factor': float(n / (time_span + 1.0))
        }


class RobustMomentsAndNormalityFeatures:
    """Computes error-weighted moments, L-moment proxies, Gini coefficient, and normality tests."""
    
    @staticmethod
    def compute(mag: np.ndarray, err: np.ndarray) -> Dict[str, float]:
        n = len(mag)
        if n == 0:
            return {}
            
        w = 1.0 / (err**2 + 1e-8)
        w_sum = np.sum(w)
        w_mean = np.sum(w * mag) / w_sum
        w_std = np.sqrt(np.sum(w * (mag - w_mean)**2) / w_sum)
        
        p5, p10, p25, p50, p75, p90, p95 = np.percentile(mag, [5, 10, 25, 50, 75, 90, 95])
        
        centered = mag - w_mean
        skewness = np.sum(w * centered**3) / (w_sum * (w_std**3 + 1e-8))
        kurtosis = np.sum(w * centered**4) / (w_sum * (w_std**4 + 1e-8)) - 3.0
        jarque_bera = (n / 6.0) * (skewness**2 + (kurtosis**2) / 4.0)
        
        # Gini coefficient of magnitudes (normalized relative inequality)
        m_sort = np.sort(mag - np.min(mag) + 1e-3)
        idx_arr = np.arange(1, n + 1)
        gini_coeff = (2.0 * np.sum(idx_arr * m_sort) / (n * np.sum(m_sort))) - ((n + 1.0) / n)
        
        # Anderson-Darling normality test statistic against normal distribution
        z = np.sort((mag - w_mean) / (w_std + 1e-8))
        cdf = np.clip(norm.cdf(z), 1e-7, 1.0 - 1e-7)
        anderson_darling = -n - np.sum((2 * idx_arr - 1) * (np.log(cdf) + np.log(1.0 - cdf[::-1]))) / n
        
        return {
            'w_mean': float(w_mean),
            'w_std': float(w_std),
            'std': float(np.std(mag)),
            'median': float(p50),
            'iqr': float(p75 - p25),
            'mad': float(np.median(np.abs(mag - p50))),
            'skewness': float(skewness),
            'kurtosis': float(kurtosis),
            'jarque_bera': float(jarque_bera),
            'anderson_darling': float(anderson_darling),
            'gini_coeff': float(gini_coeff)
        }


class QuantileDispersionAndMorphologyFeatures:
    """Computes light curve asymmetry, peak vs trough sharpness, and quantile ratios."""
    
    @staticmethod
    def compute(mag: np.ndarray) -> Dict[str, float]:
        if len(mag) == 0:
            return {}
            
        p5, p10, p25, p50, p75, p90, p95 = np.percentile(mag, [5, 10, 25, 50, 75, 90, 95])
        amp_max_min = float(np.max(mag) - np.min(mag))
        amp_90_10 = float(p90 - p10)
        amp_95_5 = float(p95 - p5)
        
        return {
            'amp_max_min': amp_max_min,
            'amp_90_10': amp_90_10,
            'amp_95_5': amp_95_5,
            'amp_ratio_90_max': float(amp_90_10 / (amp_max_min + 1e-5)),
            'quantile_ratio_25_75': float((p50 - p25) / (p75 - p50 + 1e-5)),
            'quantile_ratio_10_90': float((p50 - p10) / (p90 - p50 + 1e-5)),
            'quantile_ratio_5_95': float((p50 - p5) / (p95 - p50 + 1e-5)),
            'tail_ratio_90_10_vs_iqr': float(amp_90_10 / (p75 - p25 + 1e-5))
        }


class PhotometricNoiseAndVariabilityScatterFeatures:
    """Computes error-weighted scatter, Welch-Stetson J/K/L indices, Durbin-Watson, and Abbe values."""
    
    @staticmethod
    def compute(mjd: np.ndarray, mag: np.ndarray, err: np.ndarray, w_mean: float, w_std: float) -> Dict[str, float]:
        n = len(mag)
        if n < 3:
            return {}
            
        # Reduced Chi-Square & Excess Variance
        chi2 = np.sum(((mag - w_mean) / err)**2)
        chi2_red = chi2 / max(1, n - 1)
        mean_err2 = np.mean(err**2)
        excess_var = (np.var(mag) - mean_err2) / max(1e-5, w_mean**2)
        snr_variability = w_std / (np.mean(err) + 1e-8)
        
        # Sort by time for consecutive observation analysis
        s_idx = np.argsort(mjd)
        m_s, e_s, t_s = mag[s_idx], err[s_idx], mjd[s_idx]
        
        # Welch-Stetson J, K, L indices
        delta = np.sqrt(n / (n - 1.0)) * (m_s - w_mean) / e_s
        P_i = delta[:-1] * delta[1:]
        dt_s = np.diff(t_s)
        w_dt = np.exp(-dt_s / np.median(dt_s + 1e-5))
        
        stetson_J = np.sum(w_dt * np.sign(P_i) * np.sqrt(np.abs(P_i))) / (np.sum(w_dt) + 1e-8)
        stetson_K = (1.0 / np.sqrt(n)) * np.sum(np.abs(delta)) / np.sqrt(np.mean(delta**2) + 1e-8)
        stetson_L = (stetson_J * stetson_K) / 0.798
        
        # Von Neumann eta & Abbe value
        diff_mag = np.diff(m_s)
        von_neumann_eta = np.sum(diff_mag**2) / ((n - 1) * np.var(m_s) + 1e-8)
        abbe_value = von_neumann_eta / 2.0
        
        # Linear trend & Durbin-Watson statistic on residuals
        slope, intercept = np.polyfit(t_s - t_s[0], m_s, 1, w=1.0/(e_s + 1e-8))
        linear_trend_slope = slope * 365.25 # mag / year
        residuals = m_s - (intercept + slope * (t_s - t_s[0]))
        durbin_watson = np.sum(np.diff(residuals)**2) / (np.sum(residuals**2) + 1e-8)
        
        return {
            'chi2_red': float(chi2_red),
            'excess_var': float(excess_var),
            'snr_variability': float(snr_variability),
            'stetson_J': float(stetson_J),
            'stetson_K': float(stetson_K),
            'stetson_L': float(stetson_L),
            'von_neumann_eta': float(von_neumann_eta),
            'abbe_value': float(abbe_value),
            'linear_trend_slope': float(linear_trend_slope),
            'durbin_watson': float(durbin_watson),
            'percent_beyond_1std': float(np.mean(np.abs(mag - w_mean) > w_std)),
            'percent_beyond_2std': float(np.mean(np.abs(mag - w_mean) > 2 * w_std)),
            'percent_beyond_3std': float(np.mean(np.abs(mag - w_mean) > 3 * w_std))
        }


class IrregularTimeDomainStructureFunctionFeatures:
    """Computes Structure Function SF(Delta t) across multi-scale physical time lags."""
    
    @staticmethod
    def compute(mjd: np.ndarray, mag: np.ndarray) -> Dict[str, float]:
        n = len(mjd)
        if n < 5:
            return {f'sf_lag_{lag}': 0.0 for lag in [0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 100.0]}
            
        # Sample up to 600 observations for rapid pairwise distance computation
        if n > 600:
            idx = np.random.choice(n, size=600, replace=False)
            t_sf, m_sf = mjd[np.sort(idx)], mag[np.sort(idx)]
        else:
            t_sf, m_sf = mjd, mag
            
        dt_matrix = np.abs(t_sf[:, None] - t_sf[None, :])
        dm_matrix = (m_sf[:, None] - m_sf[None, :])**2
        
        lags = [0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 100.0]
        feats = {}
        valid_lags = []
        valid_sfs = []
        
        for lag in lags:
            mask = (dt_matrix >= lag * 0.7) & (dt_matrix <= lag * 1.4)
            if np.any(mask):
                val = float(np.mean(dm_matrix[mask]))
                feats[f'sf_lag_{lag}'] = val
                if val > 0:
                    valid_lags.append(lag)
                    valid_sfs.append(val)
            else:
                feats[f'sf_lag_{lag}'] = 0.0
                
        if len(valid_lags) >= 3:
            slope, _ = np.polyfit(np.log10(valid_lags), np.log10(valid_sfs), 1)
            feats['sf_log_slope'] = float(slope)
        else:
            feats['sf_log_slope'] = 0.0
            
        return feats


class HarmonicAndPhaseDomainFeatures:
    """Computes multi-harmonic Fourier decomposition up to 6th order, string length, and phase morphology."""
    
    @staticmethod
    def compute(mjd: np.ndarray, mag: np.ndarray, err: np.ndarray, period: float, amp_90_10: float) -> Dict[str, float]:
        n = len(mag)
        if period is None or period <= 0 or n < 5:
            return {}
            
        phase = (mjd % period) / period
        p_idx = np.argsort(phase)
        phase_s, mag_s, err_s = phase[p_idx], mag[p_idx], err[p_idx]
        
        # String Length Statistic in phase space
        string_len = np.sum(np.sqrt(np.diff(phase_s)**2 + ((mag_s[1:] - mag_s[:-1]) / (amp_90_10 + 1e-5))**2))
        string_len += np.sqrt((1.0 - phase_s[-1] + phase_s[0])**2 + ((mag_s[0] - mag_s[-1]) / (amp_90_10 + 1e-5))**2)
        
        # Phase morphology: Rise-to-fall ratio & peak vs trough widths
        idx_min = np.argmin(mag_s) # brightest point
        idx_max = np.argmax(mag_s) # faintest point
        rise_time = (phase_s[idx_min] - phase_s[idx_max]) % 1.0
        rise_to_fall_ratio = rise_time / max(1e-3, 1.0 - rise_time)
        
        mag_range = np.max(mag_s) - np.min(mag_s) + 1e-5
        width_peak_10pct = np.mean(mag_s <= np.min(mag_s) + 0.10 * mag_range)
        width_trough_10pct = np.mean(mag_s >= np.max(mag_s) - 0.10 * mag_range)
        
        feats = {
            'phase_string_length': float(string_len),
            'rise_to_fall_ratio': float(rise_to_fall_ratio),
            'width_peak_10pct': float(width_peak_10pct),
            'width_trough_10pct': float(width_trough_10pct),
            'peak_vs_trough_width_ratio': float(width_peak_10pct / (width_trough_10pct + 1e-5))
        }
        
        # 6-order Harmonic Fourier Fit: y = A0 + sum(ak cos(2pi k phase) + bk sin(2pi k phase))
        X_harm = [np.ones(n)]
        for k in range(1, 7):
            X_harm.append(np.cos(2 * np.pi * k * phase_s))
            X_harm.append(np.sin(2 * np.pi * k * phase_s))
        X_harm = np.column_stack(X_harm)
        W_harm = np.diag(1.0 / (err_s**2 + 1e-8))
        
        try:
            theta, _, _, _ = np.linalg.lstsq(X_harm.T @ W_harm @ X_harm, X_harm.T @ W_harm @ mag_s, rcond=None)
            pred = X_harm @ theta
            fourier_r2 = 1.0 - np.sum((mag_s - pred)**2) / (np.sum((mag_s - np.mean(mag_s))**2) + 1e-8)
            fourier_res_chi2_red = np.sum(((mag_s - pred) / err_s)**2) / max(1, n - 13)
            
            feats['fourier_r2'] = float(fourier_r2)
            feats['fourier_res_chi2_red'] = float(fourier_res_chi2_red)
            
            amp_k = []
            phi_k = []
            for k in range(1, 7):
                ak, bk = theta[2*k - 1], theta[2*k]
                amp = np.sqrt(ak**2 + bk**2)
                amp_k.append(amp)
                feats[f'harm_amp_{k}'] = float(amp)
                phi = np.arctan2(-bk, ak) % (2 * np.pi)
                phi_k.append(phi)
                
            for k in range(2, 7):
                feats[f'harm_amp_ratio_{k}1'] = float(amp_k[k-1] / (amp_k[0] + 1e-8))
                feats[f'harm_phi_{k}1'] = float((phi_k[k-1] - k * phi_k[0]) % (2 * np.pi))
        except Exception:
            pass
            
        return feats


class AstrometricAndGalacticContextFeatures:
    """Computes Galactic coordinates (l, b) and angular distance from the Galactic plane."""
    
    @staticmethod
    def compute(ra_deg: float, dec_deg: float) -> Dict[str, float]:
        if ra_deg is None or dec_deg is None or np.isnan(ra_deg) or np.isnan(dec_deg):
            return {'gal_l': 0.0, 'gal_b': 0.0, 'abs_gal_b': 0.0}
            
        # Convert RA/Dec (J2000) to Galactic coordinates (l, b)
        ra = np.radians(ra_deg)
        dec = np.radians(dec_deg)
        ra_ngp = np.radians(192.85948)
        dec_ngp = np.radians(27.12825)
        l_ncp = np.radians(122.93192)
        
        sin_b = np.sin(dec) * np.sin(dec_ngp) + np.cos(dec) * np.cos(dec_ngp) * np.cos(ra - ra_ngp)
        sin_b = np.clip(sin_b, -1.0, 1.0)
        b = np.arcsin(sin_b)
        
        y = np.cos(dec) * np.sin(ra - ra_ngp)
        x = np.cos(dec_ngp) * np.sin(dec) - np.sin(dec_ngp) * np.cos(dec) * np.cos(ra - ra_ngp)
        l = l_ncp - np.arctan2(y, x)
        l = l % (2 * np.pi)
        
        return {
            'gal_l': float(np.degrees(l)),
            'gal_b': float(np.degrees(b)),
            'abs_gal_b': float(np.abs(np.degrees(b)))
        }


class StarFeatureExtractor:
    """
    Master Variable Star Feature Extractor.
    Integrates all ~220 state-of-the-art per-band, inter-band, phase-domain, and astrometric features.
    """
    
    def __init__(self, bands: List[str] = ['g', 'r', 'i']):
        self.bands = bands

    def extract_features(self, star_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extracts full feature suite from a single star record.
        
        Parameters:
        -----------
        star_data : dict
            Contains 'sourceid', 'period', 'ra', 'dec', and 'bands_data' dict with band keys.
        """
        feats = {
            'sourceid': star_data.get('sourceid', ''),
            'period': float(star_data.get('period', 0.0)) if star_data.get('period') is not None else 0.0
        }
        
        # Astrometric & Galactic features
        feats.update(AstrometricAndGalacticContextFeatures.compute(
            star_data.get('ra', 0.0), star_data.get('dec', 0.0)
        ))
        
        # Per-band features
        bands_data = star_data.get('bands_data', {})
        band_stats = {}
        
        for band in self.bands:
            b = bands_data.get(band)
            if b is not None and len(b.get('target', [])) > 5:
                mjd = np.array(b['mjd'], dtype=float)
                mag = np.array(b['target'], dtype=float)
                err = np.array(b['past_feat_dynamic_real'], dtype=float) if 'past_feat_dynamic_real' in b else np.full_like(mag, 0.015)
                
                # 1. Cadence & Sampling
                b_feats = CadenceAndSamplingFeatures.compute(mjd)
                # 2. Robust Moments & Normality
                mom_feats = RobustMomentsAndNormalityFeatures.compute(mag, err)
                b_feats.update(mom_feats)
                # 3. Quantile Dispersion & Morphology
                disp_feats = QuantileDispersionAndMorphologyFeatures.compute(mag)
                b_feats.update(disp_feats)
                # 4. Photometric Noise & Variability Scatter
                if 'w_mean' in mom_feats and 'w_std' in mom_feats:
                    b_feats.update(PhotometricNoiseAndVariabilityScatterFeatures.compute(
                        mjd, mag, err, mom_feats['w_mean'], mom_feats['w_std']
                    ))
                # 5. Irregular Time-Domain Structure Functions
                b_feats.update(IrregularTimeDomainStructureFunctionFeatures.compute(mjd, mag))
                # 6. Harmonic & Phase-Domain Features
                if feats['period'] > 0 and 'amp_90_10' in disp_feats:
                    b_feats.update(HarmonicAndPhaseDomainFeatures.compute(
                        mjd, mag, err, feats['period'], disp_feats['amp_90_10']
                    ))
                
                # Prefix feature keys with band name
                for fk, fv in b_feats.items():
                    feats[f'{band}_{fk}'] = fv
                    
                band_stats[band] = {'mjd': mjd, 'mag': mag, 'err': err, 'w_mean': mom_feats.get('w_mean', 0.0), 'feats': b_feats}
            else:
                feats[f'{band}_n_obs'] = 0.0
                
        # Inter-Band Cross-Correlation & Color Dynamics (if g and r both observed)
        if 'g' in band_stats and 'r' in band_stats:
            fg, fr = band_stats['g']['feats'], band_stats['r']['feats']
            
            # Static colors & ratios
            feats['color_g_r_wmean'] = fg.get('w_mean', 0.0) - fr.get('w_mean', 0.0)
            feats['color_g_r_median'] = fg.get('median', 0.0) - fr.get('median', 0.0)
            feats['amp_ratio_g_r'] = fg.get('amp_90_10', 0.0) / (fr.get('amp_90_10', 0.0) + 1e-8)
            feats['stetson_K_ratio_g_r'] = fg.get('stetson_K', 0.0) / (fr.get('stetson_K', 0.0) + 1e-8)
            feats['excess_var_ratio_g_r'] = fg.get('excess_var', 0.0) / (fr.get('excess_var', 0.0) + 1e-8)
            
            # Near-simultaneous cross-band correlation & Welch-Stetson J_gr
            tg, mg, eg = band_stats['g']['mjd'], band_stats['g']['mag'], band_stats['g']['err']
            tr, mr, er = band_stats['r']['mjd'], band_stats['r']['mag'], band_stats['r']['err']
            
            diff_t = np.abs(tg[:, None] - tr[None, :])
            min_idx = np.argmin(diff_t, axis=1)
            min_dt = diff_t[np.arange(len(tg)), min_idx]
            mask_sim = min_dt < 0.1 # observations within ~2.4 hours
            
            if np.sum(mask_sim) > 5:
                mg_s, mr_s = mg[mask_sim], mr[min_idx[mask_sim]]
                eg_s, er_s = eg[mask_sim], er[min_idx[mask_sim]]
                color_s = mg_s - mr_s
                
                if np.std(mg_s) > 0 and np.std(color_s) > 0:
                    feats['corr_mag_color_g_gr'] = float(np.corrcoef(mg_s, color_s)[0, 1])
                else:
                    feats['corr_mag_color_g_gr'] = 0.0
                    
                if np.std(mg_s) > 0 and np.std(mr_s) > 0:
                    feats['corr_mag_g_mag_r'] = float(np.corrcoef(mg_s, mr_s)[0, 1])
                else:
                    feats['corr_mag_g_mag_r'] = 0.0
                    
                dg = (mg_s - band_stats['g']['w_mean']) / eg_s
                dr = (mr_s - band_stats['r']['w_mean']) / er_s
                P_gr = dg * dr
                feats['stetson_J_gr_cross'] = float(np.mean(np.sign(P_gr) * np.sqrt(np.abs(P_gr))))
            else:
                feats['corr_mag_color_g_gr'] = feats['corr_mag_g_mag_r'] = feats['stetson_J_gr_cross'] = 0.0
        else:
            feats['color_g_r_wmean'] = feats['color_g_r_median'] = feats['amp_ratio_g_r'] = 0.0
            feats['stetson_K_ratio_g_r'] = feats['excess_var_ratio_g_r'] = 0.0
            feats['corr_mag_color_g_gr'] = feats['corr_mag_g_mag_r'] = feats['stetson_J_gr_cross'] = 0.0
            
        return feats


if __name__ == "__main__":
    print("Testing StarFeatureExtractor on ZTF_40k sample...")
    from datasets import load_from_disk
    ds = load_from_disk("./ztf_40k")
    extractor = StarFeatureExtractor()
    sample = ds['train'][0]
    feats = extractor.extract_features(sample)
    print(f"Extracted {len(feats)} features for star {feats['sourceid']} in Class {sample.get('class_str')}.")
    print("\nSample Extracted Features:")
    for k in sorted(list(feats.keys()))[:25]:
        print(f"  {k:30s} : {feats[k]}")
