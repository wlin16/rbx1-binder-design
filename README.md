# RBX1 Binder Design with Mosaic + AlphaFold 3

Gradient-based protein binder design targeting the **RBX1 RING domain**, using [Mosaic](https://github.com/escalante-bio/mosaic)'s differentiable optimization pipeline with **AlphaFold 3** as the structure predictor.

This replaces Boltz-2 (used in the [original Mosaic pipeline](https://escalante.bio/posts/boltz-binder-design/)) with the official DeepMind AlphaFold 3, enabling gradient-based binder design through AF3's Evoformer trunk.

---

## Architecture

```
PSSM (uniform) ──► AF3 forward pass ──► ipTM / pLDDT / contact losses
      ▲                                              │
      └──────────── gradient (simplex_APGM) ◄────────┘
```

No backbone generation needed. The optimizer works directly in sequence space; structure is implicitly predicted by AF3 at every step.

**Key design choices vs. original Mosaic:**
| Component | Original Mosaic | This repo |
|---|---|---|
| Structure predictor | Boltz-2 | AlphaFold 3 |
| Optimizer | `simplex_APGM` | `simplex_APGM` (unchanged) |
| Inverse folding | ProteinMPNN (via joltz) | ProteinMPNN (unchanged) |
| Deployment | local | Modal (A100) |

---

## Target: RBX1 RING Domain

**RBX1** (Ring-Box 1, also ROC1) is the RING subunit of Cullin-RING Ligase (CRL) complexes — one of the largest families of E3 ubiquitin ligases. The RING domain recruits E2 ubiquitin-conjugating enzymes (UBE2D/UBCH5) to transfer ubiquitin to substrates.

**Targeted interface:** The E2-binding surface of the RING domain (residues 44–95, UniProt P46934):
- L2 loop (I53, D54, S55, E56, A57) — direct E2 contact
- RING loop (Q65, P66, N67, K68, I69) — E2 recruitment

Blocking this interface would inhibit CRL-mediated ubiquitination, with potential therapeutic relevance in cancers dependent on CRL substrates (e.g., p27, p21, CDT1).

**Why the RING domain fragment (52 aa) instead of full-length RBX1 (344 aa)?**
AF3's triangle self-attention has O(N²) pair representations. Gradient backprop through 48 Evoformer layers at N=424 (full RBX1 + 80 aa binder) requires ~393 GiB — physically impossible. With the RING domain (N=112), gradient optimization fits on an A100 (40–80 GB).

---

## Repository Structure

```
├── design_rbx1_binder.py          # Main design script (RBX1 RING domain target)
├── modal_test.py                  # Modal deployment (smoke test + full design run)
├── mosaic/src/mosaic/
│   ├── models/af3.py              # ← AlphaFold 3 wrapper (main contribution)
│   └── losses/af3.py             # AF3Output helper (pLDDT, PAE, ipTM accessors)
└── requirements.txt
```

> **Note:** The rest of the `mosaic` package is from [escalante-bio/mosaic](https://github.com/escalante-bio/mosaic). Install it separately: `pip install git+https://github.com/escalante-bio/mosaic --no-deps`, then replace `mosaic/src/mosaic/models/af3.py` and `mosaic/src/mosaic/losses/af3.py` with the files in this repo.

---

## Setup

### 1. Install dependencies

```bash
# Install mosaic base
pip install git+https://github.com/escalante-bio/mosaic --no-deps

# Install AF3
pip install git+https://github.com/google-deepmind/alphafold3.git

# Other deps
pip install jax[cuda12] dm-haiku equinox jaxtyping optax gemmi ml-collections einops
pip install pytorch_lightning
pip install git+https://github.com/nboyd/joltz --no-deps
pip install boltz --no-deps
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### 2. AlphaFold 3 weights

Download from [DeepMind's official repository](https://github.com/google-deepmind/alphafold3) and place `af3.bin` (or `af3.bin.zst`) in `~/.alphafold3/model/`.

### 3. AF3 CCD data

```bash
python -c "
import urllib.request, subprocess
from alphafold3.common import resources
conv_dir = resources.filename('constants/converters')
urllib.request.urlretrieve(
    'https://files.wwpdb.org/pub/pdb/data/monomers/components.cif.gz',
    '/tmp/components.cif.gz')
subprocess.run(['python', '-m', 'alphafold3.constants.converters.ccd_pickle_gen',
    '/tmp/components.cif.gz', conv_dir + '/ccd.pickle'], check=True)
subprocess.run(['python', '-m', 'alphafold3.constants.converters.chemical_component_sets_gen',
    conv_dir + '/chemical_component_sets.pickle'], check=True)
"
```

---

## Usage

### Local run

```bash
python design_rbx1_binder.py \
    --model_dir ~/.alphafold3/model \
    --binder_length 60 \
    --n_steps 200 \
    --n_candidates 8 \
    --output_dir results/rbx1
```

### Modal (cloud, A100)

```bash
# Upload weights once
modal volume put af3-weights af3.bin.zst /af3.bin.zst

# Smoke test (1 candidate, 3 steps, ~10 min)
modal run modal_test.py

# Full design (8 candidates, 200 steps, ~2h)
modal run modal_test.py --full
```

Results are saved to the `structure-results` Modal volume as `.cif` files + `results.json`.

---

## AF3 Wrapper: Key Engineering Notes

The main technical work is `mosaic/src/mosaic/models/af3.py`. Several non-obvious issues had to be solved:

**1. JIT boundary for prediction (`_predict_jit`)**
Using `@eqx.filter_jit` on the prediction function caused tracers to escape JIT scope — equinox's filter logic misclassified some values in AF3's nested result dict as static. Fix: store a separate `jax.jit(transformed.apply)` for inference only.

**2. Object-dtype numpy arrays**
AF3 features include `cleaned_struc` (a numpy object array holding an AF3 `Structure` object). JAX cannot trace object-dtype arrays. These are stripped before every JIT boundary.

**3. Gradient memory**
For gradient optimization (not inference), `apply_fn` is deliberately NOT pre-JIT'd. The outer `eqx.filter_jit` in Mosaic's optimizer compiles forward+backward as one XLA program, which is required for gradient checkpointing (`block_remat=True`) to be effective.

**4. Diffusion head**
AF3's losses (ipTM, pLDDT, PAE, distogram) come from the Evoformer trunk + confidence head, not the diffusion head. Default diffusion (5 samples × 200 steps) is reduced to 1×1 during gradient optimization, cutting ~1000× diffusion memory/compute.

---

## Citation

If you use this work, please also cite:

- **AlphaFold 3**: Abramson et al., *Nature* 2024
- **Mosaic**: escalante.bio/posts/boltz-binder-design/
- **ProteinMPNN**: Dauparas et al., *Science* 2022
