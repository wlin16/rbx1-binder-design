"""
RBX1 binder design using Mosaic + AlphaFold 3.

Replicates the Mosaic pipeline (https://github.com/escalante-bio/mosaic)
but replaces Boltz-2 with official DeepMind AlphaFold 3.

Usage:
    python design_rbx1_binder.py \
        --model_dir ~/.alphafold3/model \
        --binder_length 60 \
        --n_steps 200 \
        --output_dir results/

Loss weights mirror the Mosaic reference (escalante-bio blog):
    1.0  * BinderTargetContact
    1.0  * WithinBinderContact
    10.0 * InverseFoldingSequenceRecovery (ProteinMPNN)
    0.05 * TargetBinderPAE
    0.05 * BinderTargetPAE
    0.025 * AF3IPTMLoss  (uses tmscore_adjusted_pae_interface natively)
    0.4  * WithinBinderPAE
    0.025 * pTMEnergy
    0.1  * PLDDTLoss
Wrapped in NoCys to exclude cysteine from binder.

num_recycling=10: AF3 default (paper uses 10+1 passes).
diffusion_num_samples/steps=1: minimised for gradient memory on A100-80GB.
"""

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

# Mosaic
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
# RBX1 RING domain (UniProt P46934, residues 44-95)
#
# This is the minimal functional unit: the RING domain that recruits E2
# ubiquitin-conjugating enzymes (UBE2D/UBCH5). Targeting this fragment:
#   1. Focuses on the therapeutically relevant interface (E2 recruitment)
#   2. Keeps total N (target + binder) within AF3 gradient memory limits
#      on A100 (~80 GB): N=52+60=112 vs full-length N=344+80=424 (infeasible)
#
# Key E2-binding interface residues (canonical RING numbering):
#   L2 loop: I53, D54, S55, E56, A57  — direct E2 contact
#   RING loop: Q65, P66, N67, K68, I69 — E2 recruitment
# ─────────────────────────────────────────────────────────────────────────────
RBX1_SEQUENCE = "CPICLEMQEPVSTEAEKVLHVTRQKIFPLHPYLEMIRQELENHTLSEALRKA"  # 52 aa RING domain


def build_losses(mpnn):
    """
    Loss combination matching the Mosaic reference (escalante-bio blog).

    All weights and signs are taken directly from the reference:
        - Each LossTerm already returns a value with the correct sign for
          minimisation (negative for things to maximise, positive for things
          to minimise). NEVER apply a negative multiplier.
        - AF3IPTMLoss replaces IPTMLoss: uses AF3's native
          tmscore_adjusted_pae_interface (fully differentiable from trunk).
        - NoCys wrapper: zeros out Cys probability before passing to the loss,
          preventing cysteines in the optimised binder.
    """
    inner_loss = (
        # Contact losses (return negative values; minimise → more contact)
        1.0  * sp.BinderTargetContact()
        + 1.0  * sp.WithinBinderContact()

        # Inverse folding: moves PSSM toward sequences ProteinMPNN predicts
        # for the current predicted structure (continuous AF2-Cycler analogue)
        + 10.0 * InverseFoldingSequenceRecovery(mpnn=mpnn, temp=jnp.array(0.001), num_samples=4)

        # PAE losses (return positive PAE values; minimise → lower PAE)
        + 0.05 * sp.TargetBinderPAE()
        + 0.05 * sp.BinderTargetPAE()
        + 0.4  * sp.WithinBinderPAE()

        # Confidence (return negative values; minimise → higher confidence)
        + 0.025 * AF3IPTMLoss()
        + 0.025 * sp.pTMEnergy()
        + 0.1   * sp.PLDDTLoss()
    )
    # Return inner_loss separately — NoCys must wrap the entire AF3Loss,
    # not just the inner combination, so that set_binder_sequence receives
    # the full 20-dim PSSM (NoCys converts 19→20 before calling AF3Loss).
    return inner_loss


def design(
    model_dir: str,
    binder_length: int = 60,
    n_steps: int = 200,
    n_candidates: int = 8,
    output_dir: str = "results",
    seed: int = 42,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading AlphaFold 3 from {model_dir}...")
    # num_recycling=10: AF3 paper default (10 recycles + 1 = 11 trunk passes).
    # With block_remat=True (gradient checkpointing), backprop memory is
    # proportional to sqrt(num_recycling) rather than linear, making this
    # feasible on A100-80GB at N=112 (52 target + 60 binder).
    # diffusion_num_samples/steps=1: confidence losses (distogram, PAE, ipTM)
    # come from the trunk, not the diffusion head, so 1 sample/step suffices.
    model = AlphaFold3(
        model_dir=model_dir,
        num_recycling=10,
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

    # ── Optimization ──────────────────────────────────────────────────────
    key = jax.random.PRNGKey(seed)
    results = []

    for trial in range(n_candidates):
        key, subkey = jax.random.split(key)

        # NoCys operates on a 19-dimensional PSSM (C excluded)
        PSSM_init = jnp.ones((binder_length, 19)) / 19.0

        print(f"\n[Trial {trial+1}/{n_candidates}] Running simplex_APGM for {n_steps} steps...")

        PSSM_opt, best_PSSM = simplex_APGM(
            loss_function=af3_loss,
            x=PSSM_init,
            n_steps=n_steps,
            stepsize=0.1,
            momentum=0.9,
            key=subkey,
        )
        PSSM_opt = best_PSSM

        # Decode: re-insert zero Cys probability, then argmax
        full_pssm = NoCys.sequence(PSSM_opt)
        seq = "".join(TOKENS[int(i)] for i in jnp.argmax(full_pssm, axis=-1))

        # Final prediction
        print(f"  Final sequence: {seq}")
        pred = model.predict(
            PSSM=full_pssm,
            features=features,
            writer=None,
            key=subkey,
        )
        iptm = float(pred.iptm)
        plddt_mean = float(jnp.mean(pred.plddt[:binder_length]))
        print(f"  ipTM={iptm:.3f}  pLDDT(binder)={plddt_mean:.1f}")

        results.append({
            "trial": trial,
            "sequence": seq,
            "iptm": iptm,
            "plddt_binder": plddt_mean,
        })

        # Save structure
        out_cif = output_dir / f"binder_{trial:02d}_iptm{iptm:.3f}.cif"
        pred.st.make_mmcif_document().write_file(str(out_cif))
        print(f"  Saved: {out_cif}")

    # Save summary
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
    parser.add_argument("--model_dir", default="~/.alphafold3/model",
                        help="Directory with AF3 weights (af3.bin.zst)")
    parser.add_argument("--binder_length", type=int, default=60)
    parser.add_argument("--n_steps", type=int, default=200,
                        help="APGM optimization steps per candidate")
    parser.add_argument("--n_candidates", type=int, default=8)
    parser.add_argument("--output_dir", default="results/rbx1")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    design(
        model_dir=args.model_dir,
        binder_length=args.binder_length,
        n_steps=args.n_steps,
        n_candidates=args.n_candidates,
        output_dir=args.output_dir,
        seed=args.seed,
    )
