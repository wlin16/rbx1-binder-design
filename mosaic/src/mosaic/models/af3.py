"""
AlphaFold 3 wrapper for Mosaic binder design pipeline.

Drop-in replacement for Boltz2 / AlphaFold2 in Mosaic's StructurePredictionModel interface.

Usage:
    model = AlphaFold3(model_dir="~/.alphafold3/model")
    features, writer = model.binder_features(binder_length=80, chains=[TargetChain(rbx1_seq)])
    loss = model.build_loss(loss=my_loss_combo, features=features)
    # ... run simplex_APGM ...

Architecture (confirmed from weight file):
    c_s=384, c_z=128, c_atom=128
    48 evoformer trunk layers, 6 diffusion layers, 4 confidence layers
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import equinox as eqx
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

from jaxtyping import Array, Float, PyTree

from mosaic.structure_prediction import (
    StructurePredictionModel,
    TargetChain,
    StructurePrediction,
)
from mosaic.losses.structure_prediction import IPTMLoss, AbstractStructureOutput
from mosaic.losses.af3 import AF3Output
from mosaic.common import LossTerm, LinearCombination, tokenize

# AF3 imports (must have alphafold3 installed)
from alphafold3.common import folding_input
from alphafold3.model import model as af3_model
from alphafold3.model import params as af3_params
from alphafold3.data import featurisation
from alphafold3.constants import chemical_components, residue_names


# ─────────────────────────────────────────────────────────────────────────────
# AA ordering
# ─────────────────────────────────────────────────────────────────────────────

# Mosaic TOKENS = "ARNDCQEGHILKMFPSTWYV" == AF3 PROTEIN_TYPES_ONE_LETTER
# → no permutation needed

_AF3_ONE_LETTER = residue_names.PROTEIN_TYPES_ONE_LETTER  # tuple, 20 elements
_AF3_AA_TO_IDX = {aa: i for i, aa in enumerate(_AF3_ONE_LETTER)}
_MOSAIC_TOKENS = "ARNDCQEGHILKMFPSTWYV"


def _make_empty_msa(sequence: str) -> str:
    """Return an a3m string with a single sequence and no homologs."""
    return f">{sequence}\n{sequence}\n"


# ─────────────────────────────────────────────────────────────────────────────
# Feature preparation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_folding_input(
    chains: list[TargetChain],
    chain_ids: str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
    rng_seed: int = 0,
) -> folding_input.Input:
    """Convert Mosaic TargetChain list → AF3 folding_input.Input."""
    af3_chains = []
    for cid, chain in zip(chain_ids, chains):
        msa = _make_empty_msa(chain.sequence) if not chain.use_msa else None
        af3_chains.append(
            folding_input.ProteinChain(
                id=cid,
                sequence=chain.sequence,
                ptms=[],
                paired_msa=msa,
                unpaired_msa=msa,
                templates=[],
            )
        )
    return folding_input.Input(chains=af3_chains, rng_seeds=[rng_seed], name="binder_design")


def _featurise(
    fold_input: folding_input.Input,
    ccd,
) -> dict:
    """Run AF3 data pipeline, return a single BatchDict."""
    batches = featurisation.featurise_input(
        fold_input=fold_input,
        ccd=ccd,
        buckets=None,
    )
    assert len(batches) == 1
    return batches[0]


# ─────────────────────────────────────────────────────────────────────────────
# Soft PSSM injection
# ─────────────────────────────────────────────────────────────────────────────

def set_binder_sequence(
    PSSM: Float[Array, "N 20"],
    features: dict,
) -> dict:
    """
    Inject a soft PSSM into AF3 features for the binder (first N residues).

    Strategy (same as Mosaic AF2):
    - aatype ← argmax(PSSM)  [hard, stop-gradient for discrete ops]
    - profile ← soft PSSM   [differentiable signal]
    Straight-through estimator so gradients flow through PSSM.

    AF3 profile shape: (total_len, 32) — first 20 dims are protein AAs.
    """
    L = PSSM.shape[0]  # binder length

    # Straight-through argmax
    hard_aatype = jax.lax.stop_gradient(jnp.argmax(PSSM, axis=-1))
    soft_aatype = hard_aatype  # used for discrete ops; gradient via profile

    # Patch aatype for binder positions
    new_aatype = features["aatype"].at[:L].set(soft_aatype)

    # Patch MSA profile: AF3 profile is (N_padded, 32)
    # Protein AAs are indices 0-19; pad PSSM to 32 dims with zeros
    pssm_padded = jnp.pad(PSSM, [[0, 0], [0, features["profile"].shape[-1] - 20]])
    new_profile = features["profile"].at[:L].set(pssm_padded)

    return {**features, "aatype": new_aatype, "profile": new_profile}


# ─────────────────────────────────────────────────────────────────────────────
# AF3 loss wrapper (eqx.Module so it's a valid LossTerm pytree)
# ─────────────────────────────────────────────────────────────────────────────

class AF3Loss(LossTerm):
    """
    Wraps AF3 forward pass + a user-provided loss function.

    Calling:
        value, aux = af3_loss(PSSM, dummy_output, key=key)
        # PSSM: (L_binder, 20) jax array
    """

    apply_fn: Any          # hk.Transformed.apply (jax.jit-wrapped, treated as static by eqx)
    params: Any            # haiku params pytree (JAX arrays)
    # features contains object-dtype arrays (AF3 Structure objects) that equinox
    # would try to trace as JAX arrays, causing errors. Mark as static so equinox
    # passes them through without tracing.
    features: PyTree = eqx.field(static=True)
    loss: LossTerm | LinearCombination

    def __call__(
        self,
        sequence: Float[Array, "N 20"],
        key=None,
    ):
        if key is None:
            key = jax.random.PRNGKey(0)
        feats = set_binder_sequence(sequence, self.features)
        # Strip object-dtype numpy arrays (e.g. cleaned_struc: AF3 Structure objects)
        # before passing to JIT-compiled apply_fn — JAX cannot trace them.
        feats_jit = {k: v for k, v in feats.items()
                     if not (isinstance(v, np.ndarray) and v.dtype.kind == 'O')}
        # jax.checkpoint: during backward pass, recompute AF3 intermediate activations
        # instead of storing them. This reduces peak memory from ~393 GiB (storing all
        # 48 Evoformer + diffusion activations) to ~30-40 GB (only store I/O at boundary).
        result = jax.checkpoint(self.apply_fn)(self.params, key, feats_jit)
        output = AF3Output(batch=feats, result=result)
        v, aux = self.loss(sequence=sequence, output=output, key=key)
        return v, {"af3": aux}


# ─────────────────────────────────────────────────────────────────────────────
# Main model class
# ─────────────────────────────────────────────────────────────────────────────

class AlphaFold3(StructurePredictionModel):
    """
    AlphaFold 3 wrapped as a Mosaic StructurePredictionModel.

    Args:
        model_dir: Directory containing af3.bin.zst (or af3.bin).
        num_recycling: Number of Evoformer recycling steps (default 10 per AF3 paper).
        return_distogram: Whether to return raw distogram logits (needed for contact losses).

    Note: diffusion sampling steps are controlled by AF3's internal config (default 200).
    """

    _apply_fn: Any
    _predict_jit: Any
    _params: Any
    _ccd: Any
    num_recycling: int
    return_distogram: bool

    def __init__(
        self,
        model_dir: str | Path = "~/.alphafold3/model",
        num_recycling: int = 10,
        return_distogram: bool = True,
        diffusion_num_samples: int = 1,
        diffusion_num_steps: int = 1,
    ):
        model_dir = Path(model_dir).expanduser()

        # Load weights
        haiku_params = af3_params.get_model_haiku_params(model_dir)

        # Build model config
        cfg = af3_model.Model.Config(
            num_recycles=num_recycling,
            return_distogram=return_distogram,
        )
        cfg.heads.diffusion.eval.num_samples = diffusion_num_samples
        cfg.heads.diffusion.eval.steps = diffusion_num_steps
        cfg.evoformer.pairformer.block_remat = True
        cfg.heads.diffusion.transformer.block_remat = True

        def _forward(batch):
            m = af3_model.Model(cfg, name="diffuser")
            return m(batch)

        transformed = hk.transform(_forward)

        # _apply_fn: NOT jax.jit-wrapped. Used in AF3Loss (optimizer loop) where the outer
        # eqx.filter_jit compiles forward+backward as one program. Pre-JIT here would cause
        # nested-JIT: XLA sees jvp(jit(apply_fn)) and materializes ~393 GiB at once.
        self._apply_fn = transformed.apply
        # _predict_jit: jax.jit-wrapped. Used ONLY in predict() (no gradient needed).
        # Plain jax.jit avoids equinox's filter logic, which can misclassify AF3's nested
        # result dict values as "static" and let them escape as tracers.
        self._predict_jit = jax.jit(transformed.apply)
        self._params = jax.device_put(haiku_params)
        self._ccd = chemical_components.Ccd()
        self.num_recycling = num_recycling
        self.return_distogram = return_distogram

    @staticmethod
    def _target_features(
        chains: list[TargetChain],
        ccd: Any,
        rng_seed: int = 0,
    ) -> dict:
        fi = _build_folding_input(chains, rng_seed=rng_seed)
        return _featurise(fi, ccd)

    def target_only_features(self, chains: list[TargetChain]):
        feats = self._target_features(chains, self._ccd)
        return feats, None

    def binder_features(self, binder_length: int, chains: list[TargetChain]):
        binder_chain = TargetChain(sequence="A" * binder_length, use_msa=False)
        all_chains = [binder_chain] + list(chains)
        fi = _build_folding_input(all_chains, rng_seed=0)
        feats = _featurise(fi, self._ccd)
        return feats, None

    def build_loss(
        self,
        *,
        loss: LossTerm | LinearCombination,
        features: PyTree,
    ) -> AF3Loss:
        def _to_jax(v):
            if isinstance(v, np.ndarray) and v.dtype.kind != 'O':
                return jnp.array(v)
            if isinstance(v, dict):
                return {kk: _to_jax(vv) for kk, vv in v.items()}
            return v
        jax_features = {k: _to_jax(v) for k, v in features.items()}
        return AF3Loss(
            apply_fn=self._apply_fn,
            params=self._params,
            features=jax_features,
            loss=loss,
        )

    def model_output(
        self,
        *,
        PSSM: Float[Array, "N 20"] | None = None,
        features: PyTree,
        key: jax.Array,
    ) -> AF3Output:
        feats = features
        if PSSM is not None:
            feats = set_binder_sequence(PSSM, features)
        feats_jit = {k: v for k, v in feats.items()
                     if not (isinstance(v, np.ndarray) and v.dtype.kind == 'O')}
        result = self._apply_fn(self._params, key, feats_jit)
        return AF3Output(batch=feats, result=result)

    def _run(self, PSSM, features, key):
        feats = set_binder_sequence(PSSM, features) if PSSM is not None else features
        result = self._predict_jit(self._params, key, feats)
        seq = PSSM if PSSM is not None else jnp.zeros((0, 20))
        iptm = -IPTMLoss()(seq, AF3Output(batch=feats, result=result), key=jax.random.key(0))[0]
        return result, feats, iptm

    def predict(
        self,
        *,
        PSSM: Float[Array, "N 20"] | None = None,
        features: PyTree,
        writer: Any = None,
        key: jax.Array,
    ) -> StructurePrediction:
        features_jit = {
            k: (jnp.array(v) if isinstance(v, np.ndarray) and v.dtype.kind != 'O' else v)
            for k, v in features.items()
            if not (isinstance(v, np.ndarray) and v.dtype.kind == 'O')
        }
        result, feats, iptm = self._run(PSSM, features_jit, key)
        output = AF3Output(batch=feats, result=result)

        result_np = jax.tree_util.tree_map(
            lambda v: np.array(v) if isinstance(v, jax.Array) else v,
            result,
        )
        from alphafold3.model.model import get_predicted_structure
        from alphafold3.model import feat_batch

        batch_obj = feat_batch.Batch.from_data_dict(features)
        pred_struc = get_predicted_structure(result=result_np, batch=batch_obj)

        import gemmi
        mmcif_str = pred_struc.to_mmcif()
        st = gemmi.cif.read_string(mmcif_str)
        gemmi_st = gemmi.make_structure_from_block(st[0])

        return StructurePrediction(
            st=gemmi_st,
            plddt=output.plddt,
            pae=output.pae,
            iptm=iptm,
        )
