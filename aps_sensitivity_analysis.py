import pickle, numpy as np
np.random.seed(42)

anchors = pickle.load(open('anchors.pkl','rb'))
labels = list(anchors.keys())
S = np.array([anchors[l]['s_mean'] for l in labels])  # (14,5)
baseline_w = np.array([0.30,0.25,0.20,0.15,0.10])

def zone_original(aps):
    if aps <= 20: return "Robust"
    if aps <= 50: return "Uncertain"
    if aps <= 80: return "Probable artifact"
    return "Near-certain"

def zone_revised(aps):
    # revised, post-calibration scale proposed in the manuscript (Sec 8.1 note)
    if aps <= 20: return "Robust"
    if aps <= 75: return "Uncertain"
    return "Near-certain"

def aps_for(w):
    return 100 * S.dot(w)

base_aps = aps_for(baseline_w)
base_zone_o = [zone_original(a) for a in base_aps]
base_zone_r = [zone_revised(a) for a in base_aps]
base_rank = np.argsort(-base_aps)  # descending order indices

print("=== BASELINE ===")
for l, a, zo, zr in zip(labels, base_aps, base_zone_o, base_zone_r):
    print(f"{l:35s} APS={a:6.2f}  [original]={zo:18s} [revised]={zr}")

# --- 1) One-at-a-time (OAT) sensitivity: perturb each weight by -30%..+30%, renormalize others proportionally
print("\n=== ONE-AT-A-TIME SENSITIVITY (weight of s_i perturbed +-30%, others renormalized) ===")
deltas = np.linspace(-0.30, 0.30, 13)
oat_zone_changes = {l: set() for l in labels}
for i in range(5):
    for d in deltas:
        w = baseline_w.copy()
        w[i] = baseline_w[i] * (1+d)
        # renormalize the other 4 weights proportionally so sum stays 1
        remaining = 1 - w[i]
        others_idx = [j for j in range(5) if j != i]
        others_sum = baseline_w[others_idx].sum()
        for j in others_idx:
            w[j] = baseline_w[j] / others_sum * remaining
        aps = aps_for(w)
        for l, a in zip(labels, aps):
            oat_zone_changes[l].add(zone_revised(a))

print(f"{'Anchor':35s} {'baseline zone':18s} {'zones seen under OAT +-30%':s}")
for l in labels:
    print(f"{l:35s} {zone_revised(dict(zip(labels,base_aps))[l]):18s} {sorted(oat_zone_changes[l])}")

# --- 2) Global sensitivity: random weight vectors on the simplex, local perturbation around baseline
print("\n=== GLOBAL SENSITIVITY (N=5000 random weight vectors, +-40% per component then renormalized) ===")
N = 5000
draws = np.random.uniform(0.6, 1.4, size=(N,5)) * baseline_w
draws = draws / draws.sum(axis=1, keepdims=True)

zone_stability_revised = {l: 0 for l in labels}
zone_stability_original = {l: 0 for l in labels}
spearman_list = []
from scipy.stats import spearmanr

for w in draws:
    aps = aps_for(w)
    rank = np.argsort(-aps)
    rho, _ = spearmanr(base_aps, aps)
    spearman_list.append(rho)
    for idx, l in enumerate(labels):
        if zone_revised(aps[idx]) == base_zone_r[idx]:
            zone_stability_revised[l] += 1
        if zone_original(aps[idx]) == base_zone_o[idx]:
            zone_stability_original[l] += 1

print(f"\nSpearman rank correlation with baseline ranking: mean={np.mean(spearman_list):.4f}, "
      f"min={np.min(spearman_list):.4f}, 5th pct={np.percentile(spearman_list,5):.4f}")

print(f"\n{'Anchor':35s} {'baseline APS':>12s} {'%stable (revised zones)':>24s} {'%stable (original zones)':>25s}")
for l in labels:
    print(f"{l:35s} {dict(zip(labels,base_aps))[l]:12.2f} {100*zone_stability_revised[l]/N:23.1f}% {100*zone_stability_original[l]/N:24.1f}%")

import json
out = {
    "baseline_weights": baseline_w.tolist(),
    "baseline_aps": {l: float(a) for l,a in zip(labels, base_aps)},
    "baseline_zone_revised": dict(zip(labels, base_zone_r)),
    "baseline_zone_original": dict(zip(labels, base_zone_o)),
    "global_sensitivity": {
        "n_draws": N,
        "perturbation": "uniform +-40% per weight component, then renormalized to sum 1",
        "spearman_mean": float(np.mean(spearman_list)),
        "spearman_min": float(np.min(spearman_list)),
        "spearman_p5": float(np.percentile(spearman_list,5)),
        "zone_stability_pct_revised": {l: 100*zone_stability_revised[l]/N for l in labels},
        "zone_stability_pct_original": {l: 100*zone_stability_original[l]/N for l in labels},
    },
    "oat_zones_seen_revised": {l: sorted(oat_zone_changes[l]) for l in labels}
}
json.dump(out, open('sensitivity_results.json','w'), indent=2, ensure_ascii=False)
print("\nSaved sensitivity_results.json")
