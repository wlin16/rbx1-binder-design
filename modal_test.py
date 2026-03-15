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

# ─── Volume ───────────────────────────────────────────────────────────────────
vol_weights = modal.Volume.from_name("af3-weights")
vol_results = modal.Volume.from_name("structure-results")

# ─── Image ────────────────────────────────────────────────────────────────────
# Start from an official JAX+CUDA image; install AF3 and mosaic on top.

mosaic_path = Path(__file__).parent / "mosaic"
design_script = Path(__file__).parent / "design_rbx1_binder.py"
design_script_v2 = Path(__file__).parent / "design_rbx1_binder_v2.py"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("zstd", "git", "zlib1g-dev", "libboost-regex-dev")
    # Step 1: JAX with full deps (installs jaxlib, numpy, etc.)
    .pip_install("jax[cuda12]")
    # Step 2: AF3 + Mosaic deps
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
    # joltz + boltz installed without deps to avoid conflicting JAX pins
    # pytorch_lightning needed by joltz/boltz at import time
    # Install boltz WITH all its deps, then joltz --no-deps (avoids joltz pulling in boltz again)
    .pip_install("boltz")
    .pip_install("git+https://github.com/nboyd/joltz", extra_options="--no-deps")
    # Reinstall jax[cuda12] last so it overrides any jax version boltz may have pinned
    .pip_install("jax[cuda12]", extra_options="--upgrade")
    # Step 2b: PyTorch CPU (ProteinMPNN uses it for inverse folding)
    .pip_install(
        "torch",
        "torchvision",
        extra_options="--index-url https://download.pytorch.org/whl/cpu",
    )
    # Step 3: Build AF3 CCD data files (required at import time)
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
    .add_local_file(str(design_script_v2), remote_path="/app/design_rbx1_binder_v2.py", copy=True)
)

app = modal.App("rbx1-binder-design", image=image)


# ─── Weight decompression (run once, cache result on volume) ──────────────────
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


# ─── Smoke test ───────────────────────────────────────────────────────────────
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
    """Quick end-to-end test: 1 candidate, 3 optimization steps, binder_length=10."""
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
    from mosaic.losses.structure_prediction import AF3IPTMLoss

    MODEL_DIR = "/weights"
    BINDER_LENGTH = 5    # tiny for speed
    N_STEPS = 3
    OUTPUT_DIR = "/results/rbx1_smoketest"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # For the smoke test: use only the RBX1 RING domain core (~50 aa, residues 44-93)
    # so that N_total = 55 and the backward pass fits in GPU memory.
    # The triangle attention intermediate is O(N^3); for N=55 this is ~1 GB vs ~393 GB for N=354.
    # Full-sequence gradient optimization requires either Boltz-2 or gradient-free methods.
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

    # Smoke test: AF3-only losses (no ProteinMPNN to avoid heavy joltz/boltz deps)
    # Uses AF3IPTMLoss (tmscore_adjusted_pae_interface) and BinderTargetIPSAE
    # to verify gradient flow through all confidence metrics
    loss_fn = (
        (-1.0) * sp.BinderTargetContact()
        + (-1.0) * AF3IPTMLoss()
        + (-1.0) * sp.PLDDTLoss()
        + (-1.0) * sp.BinderTargetIPSAE()
    )
    af3_loss = model.build_loss(loss=loss_fn, features=features)

    # Diagnostic: check cleaned_struc in features
    if 'cleaned_struc' in features:
        cs = features['cleaned_struc']
        print(f"cleaned_struc: type={type(cs)}, dtype={getattr(cs,'dtype','N/A')}, shape={getattr(cs,'shape','N/A')}, value={cs}")
    # Diagnostic: find numpy arrays in af3_loss
    import equinox as eqx
    leaves = jax.tree_util.tree_leaves(eqx.filter(af3_loss, eqx.is_array))
    numpy_leaves = [(i, type(l), getattr(l, 'dtype', None)) for i, l in enumerate(leaves) if isinstance(l, __import__('numpy').ndarray)]
    if numpy_leaves:
        print(f"WARNING: {len(numpy_leaves)} numpy arrays still in af3_loss at indices: {numpy_leaves[:5]}")
    else:
        print(f"All {len(leaves)} array leaves are JAX arrays ✓")

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

    # Debug: print AF3 result dict keys so we can verify all expected outputs are present
    if hasattr(pred, '_result') and isinstance(pred._result, dict):
        print(f"AF3 result dict keys: {list(pred._result.keys())}")
        for k, v in pred._result.items():
            shape = getattr(v, 'shape', None) or (type(v).__name__)
            print(f"  {k}: {shape}")
    elif hasattr(pred, 'result') and isinstance(pred.result, dict):
        print(f"AF3 result dict keys: {list(pred.result.keys())}")
        for k, v in pred.result.items():
            shape = getattr(v, 'shape', None) or (type(v).__name__)
            print(f"  {k}: {shape}")

    iptm = float(pred.iptm)
    plddt = float(jnp.mean(pred.plddt[:BINDER_LENGTH]))
    print(f"ipTM={iptm:.3f}  pLDDT(binder)={plddt:.1f}")

    out_cif = f"{OUTPUT_DIR}/smoketest_iptm{iptm:.3f}.cif"
    pred.st.make_mmcif_document().write_file(out_cif)
    print(f"Saved structure: {out_cif}")

    vol_results.commit()
    return {"sequence": seq, "iptm": iptm, "plddt": plddt}


# ─── Full design run ──────────────────────────────────────────────────────────
@app.function(
    gpu="A100-80GB",
    volumes={
        "/weights": vol_weights,
        "/results": vol_results,
    },
    cpu=16,
    memory=131072,
    timeout=7200,
)
def full_design(
    binder_length: int = 60,
    n_steps: int = 200,
    n_candidates: int = 8,
    seed: int = 42,
):
    """Full RBX1 binder design run — 8 candidates × 200 steps on A100."""
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


# ─── Strategy 2: hotspot-biased PSSM + rebalanced losses ──────────────────────
@app.function(
    gpu="A100-80GB",
    volumes={
        "/weights": vol_weights,
        "/results": vol_results,
    },
    cpu=16,
    memory=131072,
    timeout=14400,  # 4h for 500 steps × 8 candidates
)
def full_design_v2(
    binder_length: int = 60,
    n_steps: int = 500,
    n_candidates: int = 8,
    seed: int = 99,
):
    """Strategy 2: hotspot-biased PSSM init + rebalanced losses (500 steps)."""
    sys.path.insert(0, "/mosaic/src")
    sys.path.insert(0, "/app")

    import importlib
    import design_rbx1_binder_v2 as drb2
    importlib.reload(drb2)

    results = drb2.design(
        model_dir="/weights",
        binder_length=binder_length,
        n_steps=n_steps,
        n_candidates=n_candidates,
        output_dir="/results/rbx1_v2",
        seed=seed,
    )
    vol_results.commit()
    return results


# ─── Local entrypoint ─────────────────────────────────────────────────────────
@app.local_entrypoint()
def main(full: bool = False, v2: bool = False):
    # Step 1: make sure weights are decompressed
    weight_path = decompress_weights.remote()
    print(f"Weights ready at: {weight_path}")

    if v2:
        print("\nLaunching Strategy 2 design (A100-80GB, ~3-4h) ...")
        results = full_design_v2.remote()
        print(f"\n✓ Strategy 2 done: {results}")
    elif full:
        print("\nLaunching full design (A100, ~1-2h) ...")
        results = full_design.remote()
    else:
        print("\nRunning smoke test (A10G, ~5-10 min) ...")
        result = smoke_test.remote()
        print(f"\n✓ Smoke test passed: {result}")
