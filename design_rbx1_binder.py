"""
RBX1 binder design using Mosaic + AlphaFold 3.

Replicates the Mosaic pipeline (https://github.com/escalante-bio/mosaic)
but replaces Boltz-2 with official DeepMind AlphaFold 3.

Usage:
    python design_rbx1_binder.py \
        --model_dir ~/.alphafold3/model \
        --binder_length 80 \
        --n_steps 200 \
        --output_dir results/

Architecture used (confirmed from AF3 weight file):
    c_s=384, c_z=128, c_atom=128
    48 evoformer trunk layers, 6 diffusion layers
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
from mosaic.losses.protein_mpnn import ProteinMPNNLoss
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
    Construct the Mosaic loss combination for binder design.

    Mirrors the Adaptyv competition setup from the blog post:
        BinderTargetContact + WithinBinderContact
        + IPTMLoss + BinderTargetPAE + PLDDTLoss
        + 10 * ProteinMPNNLoss (inverse folding sequence recoverability)
    """
    loss = (
        # Structural contact losses
        (-1.0) * sp.BinderTargetContact()
        + (-1.0) * sp.WithinBinderContact()

        # Confidence — AF3IPTMLoss uses tmscore_adjusted_pae_interface (fully differentiable)
        # instead of reconstructed pae_logits. Falls back to IPTMLoss for non-AF3 models.
        + (-1.0) * AF3IPTMLoss()
        + sp.BinderTargetPAE()
        + (-1.0) * sp.PLDDTLoss()

        # ipSAE: Interface Predicted Score-Aligned Error (Dunbrack 2025).
        # Complementary to ipTM — penalises high PAE on interface pairs.
        # Uses differentiable Gaussian pae_logits approximation.
        + (-1.0) * sp.BinderTargetIPSAE()

        # Inverse folding (ProteinMPNN) — reduced from 10x to 2x to prevent
        # leucine-collapse (ProteinMPNN alone pushes toward hydrophobic repeats)
        + 2.0 * ProteinMPNNLoss(mpnn=mpnn, num_samples=4)
    )
    return loss


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
    # num_recycling=1 for gradient optimization: backprop through 10 recycling steps
    # would require ~10x more memory (168 GiB at N=112). Recycling=1 gives ~17 GiB.
    # Final prediction uses a separate forward pass (predict()) which could use more.
    model = AlphaFold3(
        model_dir=model_dir,
        num_recycling=1,
        diffusion_num_samples=1,
        diffusion_num_steps=1,
    )

    print("Loading ProteinMPNN...")
    mpnn = load_mpnn()

    target = TargetChain(sequence=RBX1_SEQUENCE, use_msa=False)
    print(f"Building features: binder_length={binder_length}, target_length={len(RBX1_SEQUENCE)}")
    features, _ = model.binder_features(binder_length=binder_length, chains=[target])

    loss_fn = build_losses(mpnn)
    af3_loss = model.build_loss(loss=loss_fn, features=features)

    # ── Optimization ──────────────────────────────────────────────────────
    key = jax.random.PRNGKey(seed)
    results = []

    for trial in range(n_candidates):
        key, subkey = jax.random.split(key)

        # Initialize PSSM uniformly on simplex
        PSSM_init = jnp.ones((binder_length, 20)) / 20.0

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

        # Decode best sequence
        seq = "".join(TOKENS[int(i)] for i in jnp.argmax(PSSM_opt, axis=-1))

        # Final prediction
        print(f"  Final sequence: {seq}")
        pred = model.predict(
            PSSM=PSSM_opt,
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
        # pred.st is a gemmi.Structure
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
