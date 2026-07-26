import json, glob, os
import numpy as np

CAL = "/home/claude/osf_repo/osf_repo/calibration"

def load_replicates(path):
    d = json.load(open(path))
    if isinstance(d, dict):
        d = [d]
    return d

anchors = {}

# Multi-replicate anchors
rep_files = {
    "Shakespeare": f"{CAL}/shakespeare/results/shakespeare_result.json",
    "Copiale (decrypted)": f"{CAL}/copiale/results/copiale_replicates.json",
    "Lojban": f"{CAL}/lojban/results/lojban_replicates.json",
    "Coptic (Sahidic/Bohairic)": f"{CAL}/coptic/results/coptic_result.json",
    "Zonaras (Byzantine chronicle)": f"{CAL}/zonaras/results/zonaras_replicates.json",
    "Pride and Prejudice": f"{CAL}/pride_and_prejudice/results/pride_and_prejudice_replicates.json",
    "Patrologia Latina excerpt": f"{CAL}/patrologia/results/patrologia_replicates.json",
    "Anna Komnene, Alexiad": f"{CAL}/anna_komnene/results/anna_komnene_replicates.json",
    "Moby Dick": f"{CAL}/moby_dick/results/mobydick_replicates.json",
    "Arabic (Quran)": f"{CAL}/arabic/results/arabic_replicates.json",
    "Hebrew (Torah)": f"{CAL}/hebrew/results/hebrew_replicates.json",
    "Hoax (Markov+injected)": f"{CAL}/hoax/results/hoax_replicates.json",
    "Voynich, fully permuted (noise)": f"{CAL}/noise/results/noise_replicates.json",
}

for label, path in rep_files.items():
    reps = load_replicates(path)
    s = np.array([[r['s1'], r['s2'], r['s3'], r['s4'], r['s5']] for r in reps], dtype=float)
    aps_reported = np.array([r['aps'] for r in reps])
    anchors[label] = dict(s_mean=s.mean(axis=0), s_reps=s, n_reps=len(reps),
                           aps_reported_mean=aps_reported.mean(), aps_reported_sd=aps_reported.std())

# Voynich itself: use PUBLISHED s1-s5 from Table 9 (S_corps<->GC9_fin, ALL, p_perm<0.01)
# manuscript Sec 8.2 -- NOT reconstructed, taken directly from the docx table.
s_voynich = np.array([0.378, 1.000, 0.311, 1.000, 0.500])
anchors["Voynich Manuscript"] = dict(s_mean=s_voynich, s_reps=s_voynich.reshape(1,-1), n_reps=1,
                                      aps_reported_mean=62.6, aps_reported_sd=0.0)

print(f"{'Anchor':35s} {'s1':>7s} {'s2':>7s} {'s3':>7s} {'s4':>7s} {'s5':>7s} {'APS(recalc)':>12s} {'APS(article)':>13s}")
weights = np.array([0.30,0.25,0.20,0.15,0.10])
for label, dd in anchors.items():
    aps_recalc = 100*np.dot(weights, dd['s_mean'])
    print(f"{label:35s} " + " ".join(f"{v:7.3f}" for v in dd['s_mean']) +
          f" {aps_recalc:12.2f} {dd['aps_reported_mean']:13.2f}")

import pickle
pickle.dump(anchors, open('anchors.pkl','wb'))
print("\nn anchors:", len(anchors))
