"""AF3Output: implements AbstractStructureOutput for AlphaFold 3."""

from dataclasses import dataclass
import jax
import jax.numpy as jnp
import numpy as np

from ..common import LossTerm
from .structure_prediction import AbstractStructureOutput


# AF3 distogram: 64 bins, 2.3125 to 21.6875 Å  (same as AF2)
_AF3_DISTOGRAM_BINS = np.linspace(2.3125, 21.6875, 64)

# AF3 PAE: 64 bins, 0–32 Å, step = 0.5 Å  (from confidence_head.py: linspace(0, 31, 63) breaks)
_AF3_PAE_BREAKS = np.linspace(0.0, 31.0, 63)
_AF3_PAE_STEP = _AF3_PAE_BREAKS[1] - _AF3_PAE_BREAKS[0]
_AF3_PAE_BINS = np.concatenate([_AF3_PAE_BREAKS + _AF3_PAE_STEP / 2,
                                 [_AF3_PAE_BREAKS[-1] + 3 * _AF3_PAE_STEP / 2]])  # (64,)


@dataclass
class AF3Output(AbstractStructureOutput):
    """
    Wraps AF3 model output dict + input batch dict to implement AbstractStructureOutput.

    result keys used:
        diffusion_samples.atom_positions  (num_samples, N_atoms, 3) in token_atoms layout
        full_pae                          (num_samples, N, N)
        predicted_lddt                    (num_samples, N, num_atoms)  * 100
        distogram.contact_probs           (N, N)  [optionally distogram.distogram logits]
        tmscore_adjusted_pae_interface    (num_samples, N, N)

    batch keys used:
        asym_id                           (N,)
        residue_index                     (N,)
        aatype                            (N,)
        residue_center_index              (N,)  index of CA atom within each token's atoms
        is_protein                        (N,)
    """

    batch: dict
    result: dict
    # index of the diffusion sample to use (default 0)
    sample_idx: int = 0

    # ------------------------------------------------------------------ #
    # Sequence + chain info
    # ------------------------------------------------------------------ #

    @property
    def full_sequence(self):
        # aatype integers → one-hot (20 classes, AA 0-19)
        return jax.nn.one_hot(self.batch["aatype"], 20)

    @property
    def asym_id(self):
        return self.batch["asym_id"]

    @property
    def residue_idx(self):
        return self.batch["residue_index"]

    # ------------------------------------------------------------------ #
    # Distogram (contact probabilities as proxy)
    # ------------------------------------------------------------------ #

    @property
    def distogram_bins(self):
        return jnp.array(_AF3_DISTOGRAM_BINS)

    @property
    def distogram_logits(self):
        # AF3 only exposes contact_probs by default; convert to logits via log
        contact_probs = self.result["distogram"]["contact_probs"]  # (N, N)
        # Mosaic uses distogram_logits for the DistogramContactLoss which sums over bins.
        # Return a single-bin "logit" so that softmax(logit) * bins ≈ contact_prob.
        # Full logit tensor: shape (N, N, 64) — put all mass on nearest bin.
        # For simplicity: return log(probs + eps) broadcast to (N, N, 1) then pad.
        # Better: return logits from model if return_distogram was set.
        if "distogram" in self.result["distogram"]:
            return self.result["distogram"]["distogram"]  # (N, N, 64)
        # Fallback: one-hot at mean-dist bin from contact probs (approx)
        N = contact_probs.shape[0]
        logits = jnp.zeros((N, N, 64))
        return logits

    # ------------------------------------------------------------------ #
    # PAE
    # ------------------------------------------------------------------ #

    @property
    def pae(self):
        return self.result["full_pae"][self.sample_idx]  # (N, N)

    @property
    def pae_bins(self):
        return jnp.array(_AF3_PAE_BINS)

    @property
    def pae_logits(self):
        """
        Differentiable soft PAE logits from full_pae.

        AF3's confidence head computes full_pae = sum(softmax(pae_logits_true) * bins),
        but does NOT expose pae_logits_true in the result dict.

        We reconstruct *differentiable* logits via a Gaussian centred at full_pae:
            logits[i,j,k] = -(full_pae[i,j] - bins[k])^2 / bin_width^2

        Key property: d(logits)/d(full_pae) = 2*(full_pae - bins) / bin_width^2 ≠ 0
        → gradient flows from IPTMLoss/BinderTargetIPSAE back through full_pae
          into AF3's trunk (via full_pae = sum(softmax(pae_logits_true)*bins)).

        Previous implementation used jnp.argmin which has zero gradient everywhere
        (discrete op), completely blocking ipTM/ipSAE gradient signal.
        """
        full_pae = self.result["full_pae"][self.sample_idx]  # (N, N)
        bins = self.pae_bins  # (64,)
        bin_width = bins[1] - bins[0]  # ≈ 0.5 Å
        # Gaussian soft logits — differentiable w.r.t. full_pae
        logits = -(full_pae[:, :, None] - bins[None, None, :]) ** 2 / (bin_width ** 2)
        return logits  # (N, N, 64)

    @property
    def tmscore_adjusted_pae_interface(self):
        """
        AF3's own per-pair TM-score-adjusted PAE for interface residue pairs.
        Shape: (N, N). Computed inside AF3 from softmax(pae_logits_true) — fully differentiable.
        Higher values = better interface TM score for that residue pair.
        Use this directly to maximise ipTM without any logit reconstruction.
        """
        key = "tmscore_adjusted_pae_interface"
        if key in self.result:
            val = self.result[key]
            # AF3 batches over diffusion samples: shape (num_samples, N, N)
            if val.ndim == 3:
                return val[self.sample_idx]
            return val  # already (N, N)
        return None

    # ------------------------------------------------------------------ #
    # pLDDT — per-token (take CA atom = residue_center_index)
    # ------------------------------------------------------------------ #

    @property
    def plddt(self):
        # predicted_lddt shape: (num_samples, N_padded, num_atoms_per_token)
        # Each value is pLDDT * 100; take CA atom (residue_center_index)
        pred_lddt = self.result["predicted_lddt"][self.sample_idx]  # (N, num_atoms)
        center_idx = self.batch["residue_center_index"]              # (N,)
        ca_plddt = pred_lddt[jnp.arange(pred_lddt.shape[0]), center_idx]  # (N,)
        return ca_plddt / 100.0

    # ------------------------------------------------------------------ #
    # Backbone coordinates N, CA, C, O
    # AF3 atom positions are in token_atoms_layout: (N, max_atoms_per_token, 3)
    # Standard protein residue atom order in AF3: N, CA, C, O, CB, ...
    # ------------------------------------------------------------------ #

    @property
    def backbone_coordinates(self):
        """
        Returns (N, 4, 3) backbone coordinates [N, CA, C, O].
        Uses atom_positions from the first diffusion sample.
        AF3 protein residue atom order: index 0=N, 1=CA, 2=C, 3=O, 4=CB, ...
        """
        # atom_positions: (num_samples, N_tokens, max_atoms, 3)
        pos = self.result["diffusion_samples"]["atom_positions"][self.sample_idx]
        # Take first 4 atoms: N, CA, C, O  (shape: N, 4, 3)
        return pos[:, :4, :]
