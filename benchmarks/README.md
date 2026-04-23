# Benchmarks

Pre-computed feature sets shipped with `quantum-ml-comparator` so reviewers can
reproduce published results without running the entire upstream pipeline.

## `protein_ligand_binding_220k.parquet` (3.0 MB, snappy-compressed)

A 220,471-residue sample of public protein-ligand binding data, drawn from
**BioLiP v2**. Every residue carries:

- **34 classical features** (`classical_00` … `classical_33`): sequence and
  geometric descriptors — mean-pooled ESM-2 embedding projections plus
  residue-level solvent-accessibility and backbone-dihedral statistics, all
  standard-scaled on the training split.
- **10 VQE-derived quantum descriptors** (`quantum_0_homo_lumo_gap_eV` …
  `quantum_9_correlation_energy_casci`): computed on the ligand-category
  active site via `qmc.molecules.vqe.run_vqe` with the QNP ansatz on 8 qubits
  (4 electrons / 4 orbitals), broadcast to every residue of every complex in
  that category.
- **`label`** (int8): binding-site flag (1 if the residue contacts a ligand
  heavy atom within 4.5 Å; 0 otherwise). Base rate ≈ 5.98 %.
- **`split`** ("train" / "val" / "test"): deterministic 70 / 15 / 15 split
  stratified by complex (no residues from the same complex appear in more
  than one split). Sizes: 154,329 / 33,071 / 33,071.

### Provenance and license

- Original structures: **BioLiP v2** (Zhang Lab, licensed for unrestricted
  academic use — [https://zhanggroup.org/BioLiP/](https://zhanggroup.org/BioLiP/)).
  See the BioLiP website for the complex-level citation list.
- Sampling: 1,000 complexes sampled with `numpy.random.default_rng(seed=42)`
  after filtering to complexes with at least one non-solvent ligand and ≤ 800
  residues.
- Feature extraction pipeline: the seven-step pipeline described in the
  quantum-binding-explorer repository README
  (<https://github.com/orbion-life/quantum-binding-explorer>). That pipeline
  uses Orbion-internal infrastructure for ESM-2 embedding extraction, but the
  **output** (this parquet) is a plain feature table — anyone can re-run the
  ML comparison from it without installing Orbion software.
- License on this feature file: **MIT** (same as the rest of the package).
  Original BioLiP entries retain their own terms; cite BioLiP and the ESM-2
  paper if you reuse the features.

### Reproducing the published "+55.6 %" result

```bash
pip install quantum-ml-comparator
python examples/05_protein_ligand_binding.py
```

Expected output at `n_train = 5,000`:

```
n_train      classical F1     classical + quantum     lift
  5,000           0.1053                  0.1639     +55.6 %
```

The full learning curve (100 → 50,000 training samples) reproduces the
panel shown in the ESA OSIP submission dashboard. Runtime ≈ 3 min on an
Apple M1 Pro.

### Citation

If you use this benchmark, please cite:

```
Orbion GmbH (2026). quantum-ml-comparator: a reproducible protein-ligand
binding benchmark with classical and VQE-derived features.
https://pypi.org/project/quantum-ml-comparator/

Zhang, Y. et al. BioLiP2: an updated structure database for biologically
relevant ligand–protein interactions. Nucleic Acids Research (2023).
```
