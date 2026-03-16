"""
RBX1 binder design using Mosaic + Boltz-2 (joltz2).

Replaces AlphaFold3 with Boltz-2, which is the model Mosaic was originally
designed around. Advantages over AF3 for gradient-based hallucination:
  - Much lower memory (~8 GB vs ~70 GB for AF3 at N=112)
  - Faster per step (~0.5s vs ~3.5s with recycling=3)
  - Auto-downloads weights (no manual upload needed)
  - Better suited for diffusion-based structure generation

Loss weights follow the Mosaic reference (escalante-bio blog):
    1.0  * BinderTargetContact
    1.0  * WithinBinderContact
    10.0 * InverseFoldingSequenceRecovery (ProteinMPNN)
    0.05 * TargetBinderPAE
    0.05 * BinderTargetPAE
    0.025 * IPTMLoss
    0.4  * WithinBinderPAE
    0.025 * pTMEnergy
    0.1  * PLDDTLoss
Wrapped in NoCys to exclude cysteine from binder.
"""

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp

from mosaic.models.boltz2 import Boltz2
from mosaic.structure_prediction import TargetChain
from mosaic.optimizers import simplex_APGM
from mosaic.common import TOKENS
import mosaic.losses.structure_prediction as sp
from mosaic.losses.structure_prediction import IPTMLoss
from mosaic.losses.protein_mpnn import InverseFoldingSequenceRecovery
from mosaic.losses.transformations import NoCys
from mosaic.proteinmpnn.mpnn import load_mpnn


# RBX1 RING domain (UniProt P46934, residues 44-95), 52 aa
RBX1_SEQUENCE = "CPICLEMQEPVSTEAEKVLHVTRQKIFPLHPYLEMIRQELENHTLSEALRKA"


def build_losses(mpnn):
    inner_loss = (
        1.0  * sp.BinderTargetContact()
        + 1.0  * sp.WithinBinderContact()
        + 10.0 * InverseFoldingSequenceRecovery(mpnn=mpnn, temp=jnp.array(0.001), num_samples=4)
        + 0.05 * sp.TargetBinderPAE()
        + 0.05 * sp.BinderTargetPAE()
        + 0.4  * sp.WithinBinderPAE()
        + 0.025 * IPTMLoss()
        + 0.025 * sp.pTMEnergy()
        + 0.1   * sp.PLDDTLoss()
    )
    return inner_loss


def design(
    cache_path=None,
    binder_length: int = 60,
    n_steps: int = 200,
    n_candidates: int = 8,
    recycling_steps: int = 3,
    sampling_steps: int = 25,
    output_dir: str = "results",
    seed: int = 42,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading Boltz-2 (auto-downloads weights if needed)...")
    model = Boltz2(cache_path=Path(cache_path) if cache_path else None)

    print("Loading ProteinMPNN...")
    mpnn = load_mpnn()

    target = TargetChain(sequence=RBX1_SEQUENCE, use_msa=False)
    print(f"Building features: binder_length={binder_length}, target_length={len(RBX1_SEQUENCE)}")
    features, writer = model.binder_features(binder_length=binder_length, chains=[target])

    inner_loss = build_losses(mpnn)
    # NoCys wraps the full Boltz2Loss so set_binder_sequence gets 20-dim PSSM
    boltz2_loss = NoCys(loss=model.build_loss(
        loss=inner_loss,
        features=features,
        recycling_steps=recycling_steps,
        sampling_steps=sampling_steps,
    ))

    key = jax.random.PRNGKey(seed)
    results = []

    for trial in range(n_candidates):
        key, subkey = jax.random.split(key)

        # 19-dim PSSM: NoCys removes C from alphabet
        PSSM_init = jnp.ones((binder_length, 19)) / 19.0

        print(f"\n[Trial {trial+1}/{n_candidates}] Running simplex_APGM for {n_steps} steps...")
        PSSM_opt, best_PSSM = simplex_APGM(
            loss_function=boltz2_loss,
            x=PSSM_init,
            n_steps=n_steps,
            stepsize=0.1,
            momentum=0.9,
            key=subkey,
        )
        PSSM_opt = best_PSSM

        full_pssm = NoCys.sequence(PSSM_opt)
        seq = "".join(TOKENS[int(i)] for i in jnp.argmax(full_pssm, axis=-1))

        print(f"  Final sequence: {seq}")
        pred = model.predict(
            PSSM=full_pssm,
            features=features,
            writer=writer,
            recycling_steps=recycling_steps,
            sampling_steps=sampling_steps,
            key=subkey,
        )
        iptm = float(pred.iptm)
        plddt_mean = float(jnp.mean(pred.plddt[:binder_length]))
        print(f"  ipTM={iptm:.3f}  pLDDT(binder)={plddt_mean:.1f}")

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
    print("Top binders by ipTM:")
    for r in results_sorted[:5]:
        print(f"  ipTM={r['iptm']:.3f}  pLDDT={r['plddt_binder']:.1f}  {r['sequence']}")

    return results_sorted


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_path", default=None,
                        help="Path to boltz2_conf.ckpt (default: ~/.boltz/)")
    parser.add_argument("--binder_length", type=int, default=60)
    parser.add_argument("--n_steps", type=int, default=200)
    parser.add_argument("--n_candidates", type=int, default=8)
    parser.add_argument("--recycling_steps", type=int, default=3)
    parser.add_argument("--sampling_steps", type=int, default=25)
    parser.add_argument("--output_dir", default="results/rbx1_boltz2")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    design(
        cache_path=args.cache_path,
        binder_length=args.binder_length,
        n_steps=args.n_steps,
        n_candidates=args.n_candidates,
        recycling_steps=args.recycling_steps,
        sampling_steps=args.sampling_steps,
        output_dir=args.output_dir,
        seed=args.seed,
    )
