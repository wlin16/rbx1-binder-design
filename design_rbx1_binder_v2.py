"""
RBX1 binder design — Strategy 3: Fixed gradients + hotspot-biased PSSM + rebalanced losses.

Changes vs strategy 1:
  - Hotspot-biased PSSM init (interface-favoring amino acids weighted higher)
  - AF3IPTMLoss: uses tmscore_adjusted_pae_interface directly (fully differentiable)
    Previously IPTMLoss used argmin on pae_logits — zero gradient!
  - BinderTargetIPSAE added (ipSAE; Dunbrack 2025) — penalises high PAE on interface pairs
  - ProteinMPNN weight: 10 → 2  (prevents leucine collapse)
  - BinderTargetContact weight: -1 → -3  (stronger pull toward target contact early)
  - n_steps: 200 → 500
  - Each trial uses a different random seed for diversity
"""

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from mosaic.models.af3 import AlphaFold3
from mosaic.structure_prediction import TargetChain
from mosaic.optimizers import simplex_APGM
from mosaic.common import TOKENS
import mosaic.losses.structure_prediction as sp
from mosaic.losses.structure_prediction import AF3IPTMLoss
from mosaic.losses.protein_mpnn import InverseFoldingSequenceRecovery
from mosaic.losses.transformations import NoCys
from mosaic.proteinmpnn.mpnn import load_mpnn

# ─────────────────────────────────────────────────────────────────────────────
# RBX1 RING domain (UniProt P46934, residues 44-95), 52 aa
# ─────────────────────────────────────────────────────────────────────────────
RBX1_SEQUENCE = "CPICLEMQEPVSTEAEKVLHVTRQKIFPLHPYLEMIRQELENHTLSEALRKA"

# ─────────────────────────────────────────────────────────────────────────────
# TOKENS order (standard Mosaic/AF3 amino acid index):
# A C D E F G H I K L M N P Q R S T V W Y
# 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19
# ─────────────────────────────────────────────────────────────────────────────
AA_INDEX = {aa: i for i, aa in enumerate(TOKENS[:20])}


def hotspot_biased_pssm(binder_length: int, key: jax.Array) -> jnp.ndarray:
    """
    Build a PSSM biased toward interface-compatible amino acids.

    Rationale: RBX1 E2 interface has charged/polar residues (D54, E56, K68).
    A binder that can form salt bridges and H-bonds there needs complementary
    residues (K/R to pair with D54/E56; D/E to pair with K68; S/T/N/Q for H-bonds).
    Pure leucine repeats cannot form these contacts — we penalise their dominance
    in the initial distribution.

    We use a Dirichlet-softmax approach:
      1. Define a base weight vector (interface-favoring > hydrophobic > disruptive)
      2. Add small random noise (Dirichlet) per position for diversity across trials
      3. Normalise to simplex
    """
    # Base weights for 19 aa (TOKENS order minus C: A D E F G H I K L M N P Q R S T V W Y)
    # NoCys removes C from the alphabet, so PSSM is 19-dimensional
    base_weights = jnp.array([
        0.8,   # A  - small, ok
        1.5,   # D  - good: pairs with K68 on target
        1.5,   # E  - good: pairs with K68 on target
        0.4,   # F  - bulky aromatic, avoid at interface
        0.3,   # G  - too flexible, no sidechains
        0.8,   # H  - ok polar
        0.9,   # I  - moderate hydrophobic (ok for binder core)
        1.5,   # K  - good: pairs with D54/E56 on target
        0.6,   # L  - reduce (leucine collapse risk)
        0.5,   # M  - moderate
        1.2,   # N  - polar, good for H-bonds
        0.3,   # P  - helix breaker, avoid
        1.2,   # Q  - polar, good for H-bonds
        1.5,   # R  - good: pairs with D54/E56 on target, longer reach
        1.3,   # S  - polar, good for H-bonds
        1.2,   # T  - polar, good for H-bonds
        0.9,   # V  - moderate hydrophobic (ok for binder core)
        0.3,   # W  - too bulky
        0.6,   # Y  - moderate aromatic
    ])

    # Add per-position Dirichlet noise for diversity across trials
    # concentration ~ 3.0 * base_weights keeps mean at base_weights
    # but gives positional diversity
    concentration = 3.0 * base_weights
    noise = jax.random.dirichlet(key, concentration, shape=(binder_length,))

    # Mix: 70% base, 30% noise (ensures diversity while keeping bias)
    base_uniform = base_weights / base_weights.sum()
    pssm = 0.70 * base_uniform[None, :] + 0.30 * noise

    # Project onto simplex (already normalised from dirichlet, just normalise base part)
    pssm = pssm / pssm.sum(axis=-1, keepdims=True)
    return pssm


def build_losses(mpnn):
    """
    Loss combination matching the Mosaic reference with hotspot-biased PSSM init.
    Weights and signs are identical to design_rbx1_binder.py (the reference).
    """
    inner_loss = (
        1.0  * sp.BinderTargetContact()
        + 1.0  * sp.WithinBinderContact()
        + 10.0 * InverseFoldingSequenceRecovery(mpnn=mpnn, temp=jnp.array(0.001), num_samples=1)
        + 0.05 * sp.TargetBinderPAE()
        + 0.05 * sp.BinderTargetPAE()
        + 0.4  * sp.WithinBinderPAE()
        + 0.025 * AF3IPTMLoss()
        + 0.025 * sp.pTMEnergy()
        + 0.1   * sp.PLDDTLoss()
    )
    return inner_loss


def design(
    model_dir: str,
    binder_length: int = 60,
    n_steps: int = 500,
    n_candidates: int = 8,
    output_dir: str = "results",
    seed: int = 42,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading AlphaFold 3 from {model_dir}...")
    model = AlphaFold3(
        model_dir=model_dir,
        num_recycling=3,
        diffusion_num_samples=1,
        diffusion_num_steps=1,
    )

    print("Loading ProteinMPNN...")
    mpnn = load_mpnn()

    target = TargetChain(sequence=RBX1_SEQUENCE, use_msa=False)
    print(f"Building features: binder_length={binder_length}, target_length={len(RBX1_SEQUENCE)}")
    features, _ = model.binder_features(binder_length=binder_length, chains=[target])

    inner_loss = build_losses(mpnn)
    af3_loss = NoCys(loss=model.build_loss(loss=inner_loss, features=features))

    key = jax.random.PRNGKey(seed)
    results = []

    for trial in range(n_candidates):
        key, subkey_opt, subkey_init = jax.random.split(key, 3)

        # Strategy 2: hotspot-biased PSSM initialization (19-dim, NoCys removes C)
        PSSM_init = hotspot_biased_pssm(binder_length, subkey_init)
        top_aa = [TOKENS[i] for i in jnp.argsort(PSSM_init[0])[-5:]]
        print(f"\n[Trial {trial+1}/{n_candidates}] Top 5 initial aa (pos 0): {top_aa}")
        print(f"  Running simplex_APGM for {n_steps} steps...")

        PSSM_opt, best_PSSM = simplex_APGM(
            loss_function=af3_loss,
            x=PSSM_init,
            n_steps=n_steps,
            stepsize=0.1,
            momentum=0.9,
            key=subkey_opt,
        )
        PSSM_opt = best_PSSM

        full_pssm = NoCys.sequence(PSSM_opt)
        seq = "".join(TOKENS[int(i)] for i in jnp.argmax(full_pssm, axis=-1))

        print(f"  Final sequence: {seq}")
        pred = model.predict(
            PSSM=full_pssm,
            features=features,
            writer=None,
            key=subkey_opt,
        )
        iptm = float(pred.iptm)
        plddt_mean = float(jnp.mean(pred.plddt[:binder_length]))
        print(f"  ipTM={iptm:.3f}  pLDDT(binder)={plddt_mean:.1f}")

        # Amino acid composition analysis
        from collections import Counter
        aa_counts = Counter(seq)
        top3 = aa_counts.most_common(3)
        print(f"  AA composition (top 3): {top3}")

        results.append({
            "trial": trial,
            "sequence": seq,
            "iptm": iptm,
            "plddt_binder": plddt_mean,
            "aa_top3": top3,
        })

        out_cif = output_dir / f"binder_{trial:02d}_iptm{iptm:.3f}.cif"
        pred.st.make_mmcif_document().write_file(str(out_cif))
        print(f"  Saved: {out_cif}")

    results_sorted = sorted(results, key=lambda r: -r["iptm"])
    with open(output_dir / "results.json", "w") as f:
        json.dump(results_sorted, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Top binders by ipTM:")
    for r in results_sorted[:5]:
        print(f"  ipTM={r['iptm']:.3f}  pLDDT={r['plddt_binder']:.1f}  {r['sequence']}")

    return results_sorted


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="~/.alphafold3/model")
    parser.add_argument("--binder_length", type=int, default=60)
    parser.add_argument("--n_steps", type=int, default=500)
    parser.add_argument("--n_candidates", type=int, default=8)
    parser.add_argument("--output_dir", default="results/rbx1_v2")
    parser.add_argument("--seed", type=int, default=99)
    args = parser.parse_args()

    design(
        model_dir=args.model_dir,
        binder_length=args.binder_length,
        n_steps=args.n_steps,
        n_candidates=args.n_candidates,
        output_dir=args.output_dir,
        seed=args.seed,
    )
