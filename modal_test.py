"""
Modal smoke-test for RBX1 binder design (Mosaic + AlphaFold 3).

Usage:
    # First-time only: upload weights
    modal volume put af3-weights af3.bin.zst /af3.bin.zst

    # Run smoke test (1 candidate, 3 steps)
    modal run modal_test.py

    # Run full design (8 candidates, 200 steps) — takes ~1-2h on A100
    modal run modal_test.py --full
"""

import os
import sys
from pathlib import Path
import modal

vol_weights = modal.Volume.from_name("af3-weights")
vol_results = modal.Volume.from_name("structure-results")

mosaic_path = Path(__file__).parent / "mosaic"
design_script = Path(__file__).parent / "design_rbx1_binder.py"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("zstd", "git", "zlib1g-dev", "libboost-regex-dev")
    .pip_install("jax[cuda12]")
    .pip_install(
        "git+https://github.com/google-deepmind/alphafold3.git",
        "dm-haiku>=0.0.13",
        "equinox>=0.11.0",
        "gemmi>=0.6.5",
        "jaxtyping>=0.2.28",
        "optax>=0.2.4",
        "ml-collections>=1.0.0",
        "einops>=0.8.0",
    )
    .pip_install("pytorch_lightning")
    .pip_install(
        "git+https://github.com/nboyd/joltz",
        "boltz",
        extra_options="--no-deps",
    )
    .pip_install(
        "torch",
        "torchvision",
        extra_options="--index-url https://download.pytorch.org/whl/cpu",
    )
    .run_commands(
        "python -c \""
        "import urllib.request, os, subprocess; "
        "from alphafold3.common import resources; "
        "conv_dir = resources.filename('constants/converters'); "
        "ccd_pkl = conv_dir + '/ccd.pickle'; "
        "ccd_sets_pkl = conv_dir + '/chemical_component_sets.pickle'; "
        "print('Downloading CCD...'); "
        "urllib.request.urlretrieve("
        "  'https://files.wwpdb.org/pub/pdb/data/monomers/components.cif.gz',"
        "  '/tmp/components.cif.gz'); "
        "print('Building ccd.pickle...'); "
        "subprocess.run(['python', '-m', 'alphafold3.constants.converters.ccd_pickle_gen',"
        "  '/tmp/components.cif.gz', ccd_pkl], check=True); "
        "print('Building chemical_component_sets.pickle...'); "
        "subprocess.run(['python', '-m', 'alphafold3.constants.converters.chemical_component_sets_gen',"
        "  ccd_sets_pkl], check=True); "
        "os.remove('/tmp/components.cif.gz'); "
        "print('AF3 CCD data ready')\""
    )
    .add_local_dir(str(mosaic_path), remote_path="/mosaic", copy=True)
    .run_commands("pip install -e /mosaic --no-deps")
    .add_local_file(str(design_script), remote_path="/app/design_rbx1_binder.py", copy=True)
)

app = modal.App("rbx1-binder-design", image=image)


@app.function(
    volumes={"/weights": vol_weights},
    cpu=4,
    memory=8192,
    timeout=600,
)
def decompress_weights():
    """Decompress af3.bin.zst → af3.bin if not already done."""
    import subprocess, os
    src = "/weights/af3.bin.zst"
    dst = "/weights/af3.bin"
    if os.path.exists(dst):
        size = os.path.getsize(dst)
        print(f"af3.bin already exists ({size/1e9:.1f} GB), skipping decompression.")
        return dst
    print(f"Decompressing {src} ...")
    subprocess.run(["zstd", "-d", src, "-o", dst, "--rm", "-f"], check=True)
    vol_weights.commit()
    size = os.path.getsize(dst)
    print(f"Done. af3.bin = {size/1e9:.1f} GB")
    return dst


@app.function(
    gpu="A100",
    volumes={
        "/weights": vol_weights,
        "/results": vol_results,
    },
    cpu=8,
    memory=65536,
    timeout=3600,
)
def smoke_test():
    """Quick end-to-end test: 1 candidate, 3 optimization steps, binder_length=5."""
    sys.path.insert(0, "/mosaic/src")

    import jax
    print(f"JAX version: {jax.__version__}")
    print(f"Devices: {jax.devices()}")

    import jax.numpy as jnp
    from mosaic.models.af3 import AlphaFold3
    from mosaic.structure_prediction import TargetChain
    from mosaic.optimizers import simplex_APGM
    from mosaic.common import TOKENS
    import mosaic.losses.structure_prediction as sp

    MODEL_DIR = "/weights"
    BINDER_LENGTH = 5
    N_STEPS = 3
    OUTPUT_DIR = "/results/rbx1_smoketest"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    RBX1_SEQ = "CPICLEMQEPVSTEAEKVLHVTRQKIFPLHPYLEMIRQELENHTLSEALRKA"  # 52 aa RING core

    print("Loading AlphaFold 3 ...")
    model = AlphaFold3(
        model_dir=MODEL_DIR,
        num_recycling=1,
        diffusion_num_samples=1,
        diffusion_num_steps=1,
    )

    target = TargetChain(sequence=RBX1_SEQ, use_msa=False)
    print(f"Building features: binder={BINDER_LENGTH} aa, target={len(RBX1_SEQ)} aa")
    features, _ = model.binder_features(binder_length=BINDER_LENGTH, chains=[target])

    loss_fn = (
        (-1.0) * sp.BinderTargetContact()
        + (-1.0) * sp.IPTMLoss()
        + (-1.0) * sp.PLDDTLoss()
    )
    af3_loss = model.build_loss(loss=loss_fn, features=features)

    key = jax.random.PRNGKey(42)
    key, subkey = jax.random.split(key)
    PSSM_init = jnp.ones((BINDER_LENGTH, 20)) / 20.0

    print(f"Running simplex_APGM for {N_STEPS} steps ...")
    PSSM_opt, best_PSSM = simplex_APGM(
        loss_function=af3_loss,
        x=PSSM_init,
        n_steps=N_STEPS,
        stepsize=0.1,
        momentum=0.9,
        key=subkey,
    )
    PSSM_opt = best_PSSM

    seq = "".join(TOKENS[int(i)] for i in jnp.argmax(PSSM_opt, axis=-1))
    print(f"Optimized sequence: {seq}")

    print("Running final prediction ...")
    pred = model.predict(PSSM=PSSM_opt, features=features, key=subkey)
    iptm = float(pred.iptm)
    plddt = float(jnp.mean(pred.plddt[:BINDER_LENGTH]))
    print(f"ipTM={iptm:.3f}  pLDDT(binder)={plddt:.1f}")

    out_cif = f"{OUTPUT_DIR}/smoketest_iptm{iptm:.3f}.cif"
    pred.st.make_mmcif_document().write_file(out_cif)
    print(f"Saved structure: {out_cif}")

    vol_results.commit()
    return {"sequence": seq, "iptm": iptm, "plddt": plddt}


@app.function(
    gpu="A100",
    volumes={
        "/weights": vol_weights,
        "/results": vol_results,
    },
    cpu=16,
    memory=65536,
    timeout=7200,
)
def full_design(
    binder_length: int = 60,
    n_steps: int = 200,
    n_candidates: int = 8,
    seed: int = 42,
):
    """Full RBX1 binder design run — 8 candidates x 200 steps on A100."""
    sys.path.insert(0, "/mosaic/src")
    sys.path.insert(0, "/app")

    import importlib
    import design_rbx1_binder as drb
    importlib.reload(drb)

    results = drb.design(
        model_dir="/weights",
        binder_length=binder_length,
        n_steps=n_steps,
        n_candidates=n_candidates,
        output_dir="/results/rbx1_full",
        seed=seed,
    )
    vol_results.commit()
    return results


@app.local_entrypoint()
def main(full: bool = False):
    weight_path = decompress_weights.remote()
    print(f"Weights ready at: {weight_path}")

    if full:
        print("\nLaunching full design (A100, ~1-2h) ...")
        results = full_design.remote()
    else:
        print("\nRunning smoke test (~10 min) ...")
        result = smoke_test.remote()
        print(f"\n\u2713 Smoke test passed: {result}")
