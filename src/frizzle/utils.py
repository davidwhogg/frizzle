import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from typing import Dict, Optional, Tuple
from sklearn.neighbors import KDTree


def get_modes(
    n_modes: Optional[int],
    λ_out: npt.ArrayLike,
    mask: npt.ArrayLike,
) -> int:
    """
    Choose the number of Fourier modes to use.

    If ``n_modes`` is provided it is returned unchanged. Otherwise the value is
    set to the smaller of ``len(λ_out)`` and the number of unmasked pixels, and
    rounded down to the nearest odd integer.

    :param n_modes:
        The user-specified number of modes, or ``None`` to choose automatically.

    :param λ_out:
        The wavelengths to sample the combined spectrum on.

    :param mask:
        A boolean array where ``True`` marks masked pixels.

    :returns:
        The number of Fourier modes to use.
    """
    if n_modes is not None:
        return n_modes

    n_modes = min([len(λ_out), int(np.sum(~mask))])
    if n_modes % 2 == 0:
        n_modes -= 1

    return n_modes


def check_inputs(
    λ_out: npt.ArrayLike,
    λ: npt.ArrayLike,
    flux: npt.ArrayLike,
    ivar: Optional[npt.ArrayLike],
    mask: Optional[npt.ArrayLike],
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Validate and normalize the inputs for combining spectra.

    Concatenates per-spectrum arrays into single 1D arrays, fills in defaults
    for ``ivar`` and ``mask``, sorts ``λ_out``, and extends the mask to cover
    pixels that fall outside the resampling range.

    :param λ_out:
        The wavelengths to sample the combined spectrum on.

    :param λ:
        The input wavelengths, either a single array or a sequence of arrays
        that will be concatenated.

    :param flux:
        The input fluxes, with the same shape conventions as ``λ``.

    :param ivar:
        The input inverse variances, with the same shape conventions as ``λ``.
        If ``None``, an array of ones is used.

    :param mask:
        A boolean mask where ``True`` marks pixels to ignore, with the same
        shape conventions as ``λ``. If ``None``, no pixels are masked.

    :returns:
        A tuple ``(λ_out, λ, flux, ivar, mask)`` of normalized arrays.
    """
    λ, flux = map(jnp.hstack, (λ, flux))
    if mask is None:
        mask = jnp.zeros(flux.size, dtype=bool)
    else:
        mask = jnp.hstack(mask).astype(bool)

    λ_out = jnp.array(jnp.sort(λ_out))
    # It is important to mask things outside of the resampling range
    mask += (λ < λ_out[0]) | (λ > λ_out[-1])

    if ivar is None:
        ivar = jnp.ones_like(flux)
    else:
        ivar = jnp.hstack(ivar)

    return (λ_out, λ, flux, ivar, mask)


def separate_flags(flags: Optional[npt.ArrayLike] = None) -> Dict[int, npt.NDArray[np.bool_]]:
    """
    Separate flags into a dictionary of flags for each bit.

    :param flags:
        An ``M``-length array of flag values.

    :returns:
        A dictionary of flags, where each key is a bit and each value is an array of 0s and 1s.
    """
    separated = {}
    if flags is not None:
        for q in range(1 + int(np.log2(np.max(flags)))):
            is_set = (flags & np.uint64(2**q)) > 0
            if any(is_set):
                separated[q] = is_set.astype(bool)
    return separated


def combine_flags(
    λ_out: npt.ArrayLike,
    λ: npt.ArrayLike,
    flags: Optional[npt.NDArray[np.integer]],
) -> npt.NDArray[np.integer]:
    """
    Combine flags from input spectra.

    For each set bit, a flag is propagated to an output pixel when the nearest
    flagged input wavelength is within one output pixel width.

    :param λ_out:
        The wavelengths to sample the combined spectrum on.

    :param λ:
        The input wavelengths.

    :param flags:
        An array of integer flags, or ``None`` if no flags are provided.

    :returns:
        An array of combined integer flags with shape ``λ_out.shape``.
    """
    flags_star = np.zeros(λ_out.size, dtype=np.uint64 if flags is None else flags.dtype)
    λ_out_T = λ_out.reshape((-1, 1))
    diff_λ_out = np.diff(λ_out)
    for bit, flag in separate_flags(flags).items():
        tree = KDTree(λ[flag].reshape((-1, 1)))
        distances, indices = tree.query(λ_out_T, k=1)
        within_pixel = np.hstack([distances[:-1, 0] <= diff_λ_out, False])
        flags_star[within_pixel] += 2**bit
    return flags_star
