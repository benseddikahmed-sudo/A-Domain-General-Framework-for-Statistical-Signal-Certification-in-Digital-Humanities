[README (12).md](https://github.com/user-attachments/files/31077960/README.12.md)
# A Domain-General Framework for Statistical Signal Certification in Digital Humanities

This repository hosts the calibration code, empirical data, and reproducibility metadata for the **Artifact Probability Score (APS)**, a five-component, empirically calibrated instrument for distinguishing genuine statistical signal from methodological artifact in computationally detected patterns.

APS was originally developed within a statistical-auditing study of the Voynich Manuscript (related OSF project: https://doi.org/10.17605/OSF.IO/Y6ZCD) and is presented here as a domain-general framework, calibrated against fourteen external reference corpora spanning eight languages and four script families, with an accompanying weight-sensitivity analysis establishing the robustness of its classification scale.

## Repository contents

- **`manuscrit/`** — the manuscript submitted for peer review (Word format).
- **`étalonnage/`** — calibration pipeline (`aps_calibration.py`) and per-anchor replicate results (13 external anchors + Shakespeare/Coptic single-split results). All filenames use the English anchor names used throughout the manuscript and code (e.g. `copiale_replicates.json`, `pride_and_prejudice_replicates.json`).
- **`sensibilité/`** — weight-sensitivity analysis: `aps_sensitivity_load_data.py` (loads per-anchor results into a single dataset), `aps_sensitivity_analysis.py` (one-at-a-time and global weight-perturbation sensitivity tests, N = 5,000 draws), their outputs (`aps_sensitivity_results.json`, `aps_weight_sensitivity.csv`), and reproducibility metadata (`reproducibility_metadata_sec6.json`: execution environment, random seed, SHA-256 checksums of all input files).

## Data structure

Calibration files in `étalonnage/` are stored flat (no per-anchor subdirectories). `aps_sensitivity_load_data.py` reads directly from this flat layout.

## Reproducing the analysis

```bash
cd étalonnage/
python aps_calibration.py          # regenerates per-anchor calibration results

cd ../sensibilité/
python aps_sensitivity_load_data.py   # aggregates per-anchor results into anchors.pkl
python aps_sensitivity_analysis.py    # runs the weight-sensitivity analysis
```

## Related repository

A companion OSF repository archives the same code and data with a permanent DOI: https://doi.org/10.17605/OSF.IO/H38UG

## License

MIT license for code; CC-BY 4.0 for data.

## Citation

If you use APS or this dataset, please cite the manuscript (full citation to be added upon publication).
