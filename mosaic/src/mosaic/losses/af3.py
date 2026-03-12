"""AF3Output: implements AbstractStructureOutput for AlphaFold 3."""

from dataclasses import dataclass
import jax
import jax.numpy as jnp
import numpy as np

from ..common import LossTerm
from .structure_prediction import AbstractStructureOutput


# AF3 distogram: 64 bins, 2.3125 to 21.6875 Å  (same as AF2)
_AF3_DISTOGRAM_BINS = np.linspace(2.3125, 21.6875, 64)

# AF3 PAE: 64 bins, 0–32 Å, step = 0.5 Å
_AF3_PAE_BREAKS = np.linspace(0.0, 31.0, 63)
_AF3_PAE_STEP = _AF3_PAE_BREAKS[1] - _AF3_PAE_BREAKS[0]
_AF3_PAE_BINS = np.concatenate([_AF3_PAE_BREAKS + _AF3_PAE_STEP / 2,
                                 [_AF3_PAE_BREAKS[-1] + 3 * _AF3_PAE_STEP / 2]])  # (64,)


@dataclass
class AF3Output(AbstractStructureOutput):
    """
    Wraps AF3 model output dict + input batch dict to implement AbstractStructureOutput.

    result keys used:
        diffusion_samples.atom_positions  (num_samples, N_atoms, 3)
        full_pae                          (num_samples, N, N)
        predicted_lddt                    (num_samples, N, num_atoms) * 100
        distogram.contact_probs           (N, N)
        tmscore_adjusted_pae_interface    (num_samples, N, N)

    batch keys used:
        asym_id, residue_index, aatype, residue_center_index, is_protein
    """

    batch: dict
    result: dict
    sample_idx: int = 0

    @property
    def full_sequence(self):
        return jax.nn.one_hot(self.batch["aatype"], 20)

    @property
    def asym_id(self):
        return self.batch["asym_id"]

    @property
    def residue_idx(self):
        return self.batch["residue_index"]

    @property
    def distogram_bins(self):
        return jnp.array(_AF3_DISTOGRAM_BINS)

    @property
    def distogram_logits(self):
        contact_probs = self.result["distogram"]["contact_probs"]
        if "distogram" in self.result["distogram"]:
            return self.result["distogram"]["distogram"]
        N = contact_probs.shape[0]
        return jnp.zeros((N, N, 64))

    @property
    def pae(self):
        return self.result["full_pae"][self.sample_idx]

    @property
    def pae_bins(self):
        return jnp.array(_AF3_PAE_BINS)

    @property
    def pae_logits(self):
        full_pae = self.result["full_pae"][self.sample_idx]
        bins = self.pae_bins
        bin_idx = jnp.argmin(jnp.abs(full_pae[:, :, None] - bins[None, None, :]), axis=-1)
        return jax.nn.one_hot(bin_idx, len(bins)) * 1e6

    @property
    def plddt(self):
        pred_lddt = self.result["predicted_lddt"][self.sample_idx]
        center_idx = self.batch["residue_center_index"]
        ca_plddt = pred_lddt[jnp.arange(pred_lddt.shape[0]), center_idx]
        return ca_plddt / 100.0

    @property
    def backbone_coordinates(self):
        pos = self.result["diffusion_samples"]["atom_positions"][self.sample_idx]
        return pos[:, :4, :]
