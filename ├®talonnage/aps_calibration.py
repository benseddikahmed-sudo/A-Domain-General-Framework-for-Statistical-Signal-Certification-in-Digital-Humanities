import sys
sys.path.insert(0, '.')
import voynich_framework_v4 as vf4
from voynich_simpson_test import run_simpson_pipeline
from collections import defaultdict
from scipy import stats
import numpy as np


def build_gc9_map_for_records(records):
    """Build a GC9-style letter map from this corpus's own letter frequencies."""
    vf4.build_freq_rank_map(records)
    return dict(vf4._EVA_GC9)


def extract_pairs(records, values):
    lines = defaultdict(list)
    for r in records:
        lines[(r.folio_id, r.line_idx)].append(r)
    pairs = []
    for line_recs in lines.values():
        mid = [r for r in line_recs if r.position_in_line == 'MIDDLE']
        end = [r for r in line_recs if r.position_in_line == 'END']
        if not mid or not end:
            continue
        s = sum(values[r.token] for r in mid)
        g = values[end[-1].token]
        pairs.append((s, g))
    return pairs


def compute_full_aps(records, label, random_seed=42):
    """Full 5-component APS for the S_corps<->GC9_fin anticorrelation signal,
    computed end-to-end on an arbitrary corpus with a two-group split
    encoded in TokenRecord.language ('A' vs 'B')."""
    smap = build_gc9_map_for_records(records)
    values = {t: vf4.compute_voynich_value(t, vf4.VoynichNumericalSystem.GC9)
              for t in set(r.token for r in records)}

    recs_a = [r for r in records if r.language == 'A']
    recs_b = [r for r in records if r.language == 'B']

    data_A = extract_pairs(recs_a, values)
    data_B = extract_pairs(recs_b, values)

    # --- s2, s3, s5 via the Simpson diagnostic battery ---
    report = run_simpson_pipeline(data_A, data_B, n_bootstrap=1000,
                                   n_perm_diagnostic=2000, random_seed=random_seed)

    all_pairs = data_A + data_B
    S = np.array([p[0] for p in all_pairs]); G = np.array([p[1] for p in all_pairs])
    r_obs, p_obs = stats.pearsonr(S, G)

    n_tests_sim = report.diagnostic.n_tests_confirming
    s2 = n_tests_sim / 5.0

    r_bal = report.resampling.r_balanced_mean
    s3 = min(1.0, abs(r_obs - r_bal) / abs(r_obs)) if r_obs != 0 else 1.0

    Sa = np.array([p[0] for p in data_A]); Ga = np.array([p[1] for p in data_A])
    Sb = np.array([p[0] for p in data_B]); Gb = np.array([p[1] for p in data_B])
    _, pA = stats.pearsonr(Sa, Ga) if len(Sa) > 2 else (None, 1.0)
    _, pB = stats.pearsonr(Sb, Gb) if len(Sb) > 2 else (None, 1.0)
    s5 = (int(pA >= 0.05) + int(pB >= 0.05)) / 2

    # --- s1, s4 via the 37+1-system robustness battery ---
    rob = vf4.RobustnessAnalysis(records, random_seed=random_seed)
    res = rob.run()
    s1 = 1.0 - res['fraction_anti']
    p_bonf37 = min(1.0, p_obs * 37)
    s4 = 1 if p_bonf37 > 0.05 else 0

    weights = dict(s1=0.30, s2=0.25, s3=0.20, s4=0.15, s5=0.10)
    aps = 100 * (weights['s1']*s1 + weights['s2']*s2 + weights['s3']*s3 +
                 weights['s4']*s4 + weights['s5']*s5)

    if aps <= 20:   zone = "Robust"
    elif aps <= 50: zone = "Uncertain"
    elif aps <= 80: zone = "Probable artifact"
    else:           zone = "Near-certain artifact"

    print(f"=== {label} ===")
    print(f"  n_A={len(data_A)} lines, n_B={len(data_B)} lines")
    print(f"  r_obs={r_obs:.4f} p_obs={p_obs:.3e}")
    print(f"  GC9-original: r={res['gc9_original_r']:.4f} p={res['gc9_original_p']:.3e} "
          f"rank={res['rank_gc9_original']}/{res['n_systems']+1} "
          f"fraction_anti={res['fraction_anti']:.3f}")
    print(f"  s1={s1:.3f}  s2={s2:.3f}  s3={s3:.3f}  s4={s4}  s5={s5:.3f}")
    print(f"  APS = {aps:.1f}  --> {zone}")
    print()
    return dict(label=label, r_obs=r_obs, p_obs=p_obs, s1=s1, s2=s2, s3=s3, s4=s4, s5=s5,
                aps=aps, zone=zone, n_A=len(data_A), n_B=len(data_B))
