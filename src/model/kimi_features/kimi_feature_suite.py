"""
kimi_feature_suite.py
=====================
Implementation of Kimi-2.5's 102-feature engineering suite as defined in ZTF_Feature_Engineering_Suite.pdf.

Architecture:
- 46 features for g-band (prefixed with g_)
- 46 features for r-band (prefixed with r_)
- 10 multi-band/cross-correlation/period features (clr_mn, clr_md, clr_sd, amp_r, bnd_corr, p_agree, gp, rp, etc.)
Total = 102 features per star.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from typing import Dict, Any, List, Optional, Tuple

class KimiSingleBandExtractor:
    """Extracts the 46 per-band features formulated by Kimi-2.5."""
    
    @staticmethod
    def compute(mjd: np.ndarray, mag: np.ndarray, err: np.ndarray, band: str) -> Dict[str, float]:
        n = len(mag)
        prefix = f"{band}_"
        if n < 5:
            # Return zeroed default dict if insufficient observations
            return KimiSingleBandExtractor._get_empty(prefix)
            
        # Ensure finite weights
        w = 1.0 / (err**2 + 1e-6)
        w_sum = np.sum(w)
        wmean = float(np.sum(w * mag) / w_sum)
        wstd = float(np.sqrt(np.sum(w * (mag - wmean)**2) / w_sum))
        
        # A1. Central Tendency & Dispersion
        p5, p10, p25, p50, p75, p90, p95 = np.percentile(mag, [5, 10, 25, 50, 75, 90, 95])
        med = float(p50)
        mad = float(np.median(np.abs(mag - med)))
        mn = float(np.mean(mag))
        std = float(np.std(mag))
        q31 = float(p75 - p25)
        amp = float((p95 - p5) / 2.0)
        rng = float(np.max(mag) - np.min(mag))
        
        # A2. Variability Indices
        mean_err2 = float(np.mean(err**2))
        eta_e = float((std**2) / (mean_err2 + 1e-8))
        exvar = float(max(0.0, ((n / max(1, n - 1)) * std**2 - mean_err2)))
        beyond1 = float(np.sum(np.abs(mag - wmean) > wstd) / n)
        
        # A3. Stetson Indices
        s_idx = np.argsort(mjd)
        m_s, e_s = mag[s_idx], err[s_idx]
        delta = np.sqrt(n / max(1, n - 1)) * (m_s - wmean) / e_s
        P_i = delta[:-1] * delta[1:]
        stet_k = float((1.0 / np.sqrt(n)) * np.sum(np.abs(delta)) / np.sqrt(np.mean(delta**2) + 1e-8))
        stet_j = float(np.sum(np.sign(P_i) * np.sqrt(np.abs(P_i))) / (np.sum(np.sqrt(np.abs(P_i))) + 1e-8)) if len(P_i) > 0 else 0.0
        
        # A4. Statistical Tests
        centered = mag - wmean
        skew = float(np.sum(w * centered**3) / (w_sum * (wstd**3 + 1e-8)))
        kurt = float(np.sum(w * centered**4) / (w_sum * (wstd**4 + 1e-8)) - 3.0)
        
        # Anderson-Darling
        z = np.sort((mag - wmean) / (wstd + 1e-8))
        cdf = np.clip(norm.cdf(z), 1e-7, 1.0 - 1e-7)
        idx_arr = np.arange(1, n + 1)
        ad = float(-n - np.sum((2 * idx_arr - 1) * (np.log(cdf) + np.log(1.0 - cdf[::-1]))) / n)
        
        # A5. Trend & Slope Features
        t_s = mjd[s_idx]
        t_cen = t_s - np.sum(w * t_s) / w_sum
        denom_t = np.sum(w * t_cen**2) + 1e-8
        trend = float(np.sum(w * t_cen * (m_s - wmean)) / denom_t) # mag/day
        
        dt_s = np.diff(t_s)
        dm_s = np.diff(m_s)
        valid_dt = dt_s > 1e-4
        if np.any(valid_dt):
            slopes = dm_s[valid_dt] / dt_s[valid_dt]
            maxslope = float(np.max(np.abs(slopes)))
            pst = float(np.sum(slopes > 0) / len(slopes))
        else:
            maxslope = 0.0
            pst = 0.5
            
        cum_sum = np.cumsum(m_s - mn)
        rcs = float(np.max(cum_sum) - np.min(cum_sum))
        
        # C. Shape & Information Theory
        gskew = float((p90 + p10 - 2.0 * med) / (p90 - p10 + 1e-8))
        welch = float((mn - med) / (mad + 1e-8))
        pr = float((p90 - med) / (med - p10 + 1e-8))
        
        # Entropy
        counts, _ = np.histogram(mag, bins=max(5, min(20, n // 3)))
        probs = counts / np.sum(counts)
        probs = probs[probs > 0]
        ent = float(-np.sum(probs * np.log2(probs)) / np.log2(len(counts)))
        
        c1 = float(np.sum(np.abs(mag - mn) < std) / n)
        mbr = float(np.sum(np.abs(mag - med) < 0.1) / n)
        
        # B. Periodicity & Harmonics (Lomb-Scargle + 3-Harmonic Fourier Fit)
        best_period, psig = KimiSingleBandExtractor._estimate_period(t_s, m_s, e_s)
        
        # 3-Harmonic Fourier Fit on phase
        fourier_feats = KimiSingleBandExtractor._fit_3_harmonics(t_s, m_s, e_s, best_period)
        
        # Phase-Folded Shape features
        phase = (t_s / best_period) % 1.0
        p_idx = np.argsort(phase)
        m_p = m_s[p_idx]
        
        psk = float(np.sum((m_p - mn)**3) / (n * (std**3 + 1e-8)))
        pkt = float(np.sum((m_p - mn)**4) / (n * (std**4 + 1e-8)) - 3.0)
        
        # Bimodality coefficient
        pbm = float((psk**2 + 1.0) / (pkt + 3.0 + 1e-8))
        prng = float(np.max(m_p) - np.min(m_p))
        
        half_1 = m_p[:n//2]
        half_2 = m_p[n//2:]
        if len(half_1) > 0 and len(half_2) > 0:
            range1 = np.max(half_1) - np.min(half_1)
            range2 = np.max(half_2) - np.min(half_2)
            prat = float(range1 / (range2 + 1e-8))
            psym = float(np.abs(np.mean(half_1) - np.mean(half_2)) / (prng + 1e-8))
        else:
            prat, psym = 1.0, 0.0
            
        feats = {
            # A1
            f"{prefix}wmean": wmean, f"{prefix}wstd": wstd, f"{prefix}med": med, f"{prefix}mad": mad,
            f"{prefix}mn": mn, f"{prefix}std": std, f"{prefix}q31": q31, f"{prefix}amp": amp, f"{prefix}rng": rng,
            # A2
            f"{prefix}eta_e": eta_e, f"{prefix}exvar": exvar, f"{prefix}beyond1": beyond1,
            # A3
            f"{prefix}stet_k": stet_k, f"{prefix}stet_j": stet_j,
            # A4
            f"{prefix}skew": skew, f"{prefix}kurt": kurt, f"{prefix}ad": ad,
            # A5
            f"{prefix}trend": trend, f"{prefix}maxslope": maxslope, f"{prefix}pst": pst, f"{prefix}rcs": rcs,
            # C
            f"{prefix}gskew": gskew, f"{prefix}welch": welch, f"{prefix}pr": pr,
            f"{prefix}ent": ent, f"{prefix}c1": c1, f"{prefix}mbr": mbr,
            # B1
            f"{prefix}period": best_period, f"{prefix}psig": psig,
            # B3
            f"{prefix}psk": psk, f"{prefix}pkt": pkt, f"{prefix}pbm": pbm,
            f"{prefix}prng": prng, f"{prefix}prat": prat, f"{prefix}psym": psym,
        }
        for k, v in fourier_feats.items():
            feats[f"{prefix}{k}"] = v
            
        return feats

    @staticmethod
    def _estimate_period(t: np.ndarray, y: np.ndarray, dy: np.ndarray) -> Tuple[float, float]:
        """Fast grid Lomb-Scargle period estimate."""
        n = len(t)
        if n < 5:
            return 1.0, 0.0
            
        t_span = t[-1] - t[0]
        if t_span <= 0:
            return 1.0, 0.0
            
        freqs = np.linspace(1.0 / max(1.0, t_span), 10.0, min(500, n * 10))
        w = 1.0 / (dy**2 + 1e-8)
        y_w = y - np.sum(w * y) / np.sum(w)
        
        omega = 2.0 * np.pi * freqs[:, None]
        wt = omega * t[None, :]
        
        # Fast approximate power
        cos_wt = np.cos(wt)
        sin_wt = np.sin(wt)
        
        p_c = np.sum(w[None, :] * y_w[None, :] * cos_wt, axis=1)**2 / (np.sum(w[None, :] * cos_wt**2, axis=1) + 1e-8)
        p_s = np.sum(w[None, :] * y_w[None, :] * sin_wt, axis=1)**2 / (np.sum(w[None, :] * sin_wt**2, axis=1) + 1e-8)
        power = 0.5 * (p_c + p_s) / (np.sum(w * y_w**2) + 1e-8)
        
        best_idx = np.argmax(power)
        best_period = float(1.0 / freqs[best_idx])
        psig = float((power[best_idx] - np.mean(power)) / (np.std(power) + 1e-8))
        return best_period, psig

    @staticmethod
    def _fit_3_harmonics(t: np.ndarray, y: np.ndarray, dy: np.ndarray, period: float) -> Dict[str, float]:
        """Fits 3 harmonics: y(phi) = a0 + sum_{k=1..3} [A_k cos + B_k sin]."""
        if len(t) < 8 or period <= 0:
            return {
                'a0': 0.0, 'a1': 0.0, 'a2': 0.0, 'a3': 0.0,
                'p1': 0.0, 'p2': 0.0, 'p3': 0.0,
                'ar2': 0.0, 'ar3': 0.0, 'pd2': 0.0, 'pd3': 0.0, 'fr2': 0.0
            }
            
        phi = 2.0 * np.pi * (t / period)
        X = np.column_stack([
            np.ones_like(t),
            np.cos(phi), np.sin(phi),
            np.cos(2*phi), np.sin(2*phi),
            np.cos(3*phi), np.sin(3*phi)
        ])
        w = 1.0 / (dy**2 + 1e-8)
        W = np.diag(w)
        
        try:
            # Weighted least squares: (X^T W X + lambda I)^(-1) X^T W y
            XT_W = X.T @ W
            beta = np.linalg.solve(XT_W @ X + 1e-5 * np.eye(7), XT_W @ y)
        except Exception:
            beta = np.zeros(7)
            
        a0 = float(beta[0])
        a1 = float(np.hypot(beta[1], beta[2]))
        a2 = float(np.hypot(beta[3], beta[4]))
        a3 = float(np.hypot(beta[5], beta[6]))
        
        p1 = float(np.arctan2(beta[2], beta[1]))
        p2 = float(np.arctan2(beta[4], beta[3]))
        p3 = float(np.arctan2(beta[6], beta[5]))
        
        ar2 = float(a2 / (a1 + 1e-8))
        ar3 = float(a3 / (a1 + 1e-8))
        pd2 = float(p2 - 2.0 * p1)
        pd3 = float(p3 - 3.0 * p1)
        
        y_pred = X @ beta
        y_mean = np.sum(w * y) / np.sum(w)
        ss_tot = np.sum(w * (y - y_mean)**2) + 1e-8
        ss_res = np.sum(w * (y - y_pred)**2)
        fr2 = float(max(0.0, 1.0 - (ss_res / ss_tot)))
        
        return {
            'a0': a0, 'a1': a1, 'a2': a2, 'a3': a3,
            'p1': p1, 'p2': p2, 'p3': p3,
            'ar2': ar2, 'ar3': ar3, 'pd2': pd2, 'pd3': pd3, 'fr2': fr2
        }

    @staticmethod
    def _get_empty(prefix: str) -> Dict[str, float]:
        keys = [
            'wmean', 'wstd', 'med', 'mad', 'mn', 'std', 'q31', 'amp', 'rng',
            'eta_e', 'exvar', 'beyond1', 'stet_k', 'stet_j', 'skew', 'kurt', 'ad',
            'trend', 'maxslope', 'pst', 'rcs', 'gskew', 'welch', 'pr', 'ent', 'c1', 'mbr',
            'period', 'psig', 'psk', 'pkt', 'pbm', 'prng', 'prat', 'psym',
            'a0', 'a1', 'a2', 'a3', 'p1', 'p2', 'p3', 'ar2', 'ar3', 'pd2', 'pd3', 'fr2'
        ]
        return {f"{prefix}{k}": 0.0 for k in keys}


class KimiFeatureExtractor:
    """Extracts all 102 Kimi features (46 g-band + 46 r-band + 10 multi-band/color)."""
    
    def __init__(self):
        pass
        
    def extract_features(self, star_record: Dict[str, Any]) -> Dict[str, Any]:
        feats = {'sourceid': star_record.get('source_id', star_record.get('star_id', star_record.get('sourceid', 'unknown')))}
        
        # 1. Single Band g
        if 'time_g' in star_record and len(star_record['time_g']) > 0:
            mjd_g, mag_g, err_g = np.array(star_record['time_g']), np.array(star_record['mag_g']), np.array(star_record['err_g'])
            g_feats = KimiSingleBandExtractor.compute(mjd_g, mag_g, err_g, 'g')
        elif 'bands_data' in star_record and 'g' in star_record['bands_data']:
            b = star_record['bands_data']['g']
            mjd_g, mag_g, err_g = np.array(b['mjd']), np.array(b['target']), np.array(b['past_feat_dynamic_real'])
            g_feats = KimiSingleBandExtractor.compute(mjd_g, mag_g, err_g, 'g')
        else:
            mjd_g, mag_g, err_g = np.array([]), np.array([]), np.array([])
            g_feats = KimiSingleBandExtractor._get_empty('g_')
        feats.update(g_feats)
        
        # 2. Single Band r
        if 'time_r' in star_record and len(star_record['time_r']) > 0:
            mjd_r, mag_r, err_r = np.array(star_record['time_r']), np.array(star_record['mag_r']), np.array(star_record['err_r'])
            r_feats = KimiSingleBandExtractor.compute(mjd_r, mag_r, err_r, 'r')
        elif 'bands_data' in star_record and 'r' in star_record['bands_data']:
            b = star_record['bands_data']['r']
            mjd_r, mag_r, err_r = np.array(b['mjd']), np.array(b['target']), np.array(b['past_feat_dynamic_real'])
            r_feats = KimiSingleBandExtractor.compute(mjd_r, mag_r, err_r, 'r')
        else:
            mjd_r, mag_r, err_r = np.array([]), np.array([]), np.array([])
            r_feats = KimiSingleBandExtractor._get_empty('r_')
        feats.update(r_feats)
        
        # 3. Multi-Band / Color / Correlation features
        if len(mag_g) > 0 and len(mag_r) > 0:
            clr_mn = float(g_feats.get('g_mn', 0.0) - r_feats.get('r_mn', 0.0))
            clr_md = float(g_feats.get('g_med', 0.0) - r_feats.get('r_med', 0.0))
            clr_sd = float(g_feats.get('g_std', 0.0) - r_feats.get('r_std', 0.0))
            amp_r = float(g_feats.get('g_amp', 0.0) / (r_feats.get('r_amp', 0.0) + 1e-8))
            
            # Near-simultaneous band correlation
            diff_t = np.abs(mjd_g[:, None] - mjd_r[None, :])
            min_idx = np.argmin(diff_t, axis=1)
            min_dt = diff_t[np.arange(len(mjd_g)), min_idx]
            mask_sim = min_dt < 0.1 # observations within ~2.4 hours
            if np.sum(mask_sim) > 4:
                mg_s, mr_s = mag_g[mask_sim], mag_r[min_idx[mask_sim]]
                if np.std(mg_s) > 0 and np.std(mr_s) > 0:
                    bnd_corr = float(np.corrcoef(mg_s, mr_s)[0, 1])
                else:
                    bnd_corr = 0.0
            else:
                bnd_corr = 0.0
        else:
            clr_mn = clr_md = clr_sd = amp_r = bnd_corr = 0.0
            
        g_period = float(g_feats.get('g_period', 1.0))
        r_period = float(r_feats.get('r_period', 1.0))
        p_agree = float(np.abs(g_period - r_period) / max(1e-5, max(g_period, r_period)))
        p_ratio = float(g_period / max(1e-5, r_period))
        
        # Overall master period
        period = star_record.get('period', g_period if g_period > 0 else 1.0)
        if period is None or np.isnan(period) or period <= 0:
            period = g_period if g_period > 0 else 1.0
            
        feats['clr_mn'] = clr_mn
        feats['clr_md'] = clr_md
        feats['clr_sd'] = clr_sd
        feats['amp_r'] = amp_r
        feats['bnd_corr'] = bnd_corr
        feats['p_agree'] = p_agree
        feats['p_ratio'] = p_ratio
        feats['period'] = float(period)
        
        return feats

if __name__ == "__main__":
    extractor = KimiFeatureExtractor()
    dummy = {
        'source_id': 'test_star',
        'time_g': np.linspace(58000, 58100, 30), 'mag_g': np.sin(np.linspace(0, 10, 30)) + 15.0, 'err_g': np.full(30, 0.02),
        'time_r': np.linspace(58000, 58100, 30), 'mag_r': np.sin(np.linspace(0, 10, 30)) + 14.5, 'err_r': np.full(30, 0.02)
    }
    res = extractor.extract_features(dummy)
    cols = [k for k in sorted(res.keys()) if k != 'sourceid']
    print(f"Extracted {len(cols)} total Kimi features successfully!")
