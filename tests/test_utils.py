import jax.numpy as jnp
import numpy as np
import pytest

from frizzle.utils import (
    check_inputs,
    combine_flags,
    get_modes,
    separate_flags,
)


def test_get_modes_returns_explicit_value():
    assert get_modes(7, jnp.linspace(0, 1, 10), jnp.zeros(20, dtype=bool)) == 7


def test_get_modes_returns_zero_when_user_passes_zero():
    # An explicit 0 should be passed straight through (it is not None).
    assert get_modes(0, jnp.linspace(0, 1, 10), jnp.zeros(20, dtype=bool)) == 0


def test_get_modes_auto_uses_smaller_of_lambda_out_and_unmasked():
    λ_out = jnp.linspace(0.0, 1.0, 11)
    mask = jnp.zeros(50, dtype=bool)
    # min(11, 50) = 11; 11 is odd, returned as-is.
    assert get_modes(None, λ_out, mask) == 11


def test_get_modes_auto_rounds_even_value_down_to_odd():
    λ_out = jnp.linspace(0.0, 1.0, 10)
    mask = jnp.zeros(50, dtype=bool)
    # min(10, 50) = 10; even -> 9.
    assert get_modes(None, λ_out, mask) == 9


def test_get_modes_auto_uses_unmasked_count_when_smaller():
    λ_out = jnp.linspace(0.0, 1.0, 100)
    mask = jnp.array([False] * 7 + [True] * 50)
    # 7 unmasked, < 100 — and 7 is odd.
    assert get_modes(None, λ_out, mask) == 7


def test_check_inputs_defaults_for_ivar_and_mask():
    λ_out = jnp.array([0.5, 1.0, 1.5, 2.0])
    λ = jnp.array([0.6, 0.8, 1.2, 1.7])
    flux = jnp.array([1.0, 2.0, 3.0, 4.0])

    λ_out_o, λ_o, flux_o, ivar_o, mask_o = check_inputs(λ_out, λ, flux, None, None)

    assert λ_out_o.shape == (4,)
    assert λ_o.shape == (4,)
    assert flux_o.shape == (4,)
    # default ivar is ones.
    assert jnp.allclose(ivar_o, jnp.ones_like(flux))
    # No pixels outside the [λ_out[0], λ_out[-1]] range, mask is all False.
    assert mask_o.dtype == bool
    assert not bool(jnp.any(mask_o))


def test_check_inputs_sorts_lambda_out():
    λ_out = jnp.array([2.0, 1.0, 1.5])
    λ = jnp.array([1.2])
    flux = jnp.array([1.0])
    λ_out_o, *_ = check_inputs(λ_out, λ, flux, None, None)
    assert jnp.allclose(λ_out_o, jnp.array([1.0, 1.5, 2.0]))


def test_check_inputs_masks_out_of_range_pixels():
    λ_out = jnp.array([1.0, 2.0])
    λ = jnp.array([0.5, 1.5, 2.5])  # only middle pixel is in-range
    flux = jnp.array([1.0, 1.0, 1.0])

    _, _, _, _, mask = check_inputs(λ_out, λ, flux, None, None)
    assert bool(mask[0])
    assert not bool(mask[1])
    assert bool(mask[2])


def test_check_inputs_concatenates_sequence_of_arrays():
    λ_out = jnp.array([0.0, 1.0, 2.0])
    λ = [jnp.array([0.5, 0.7]), jnp.array([1.2, 1.6])]
    flux = [jnp.array([10.0, 20.0]), jnp.array([30.0, 40.0])]
    ivar = [jnp.array([1.0, 1.0]), jnp.array([2.0, 2.0])]
    mask = [jnp.array([0, 1]), jnp.array([0, 0])]

    λ_out_o, λ_o, flux_o, ivar_o, mask_o = check_inputs(λ_out, λ, flux, ivar, mask)
    assert λ_o.shape == (4,)
    assert flux_o.shape == (4,)
    assert ivar_o.shape == (4,)
    assert mask_o.dtype == bool
    # The user-provided mask bit should still be set.
    assert bool(mask_o[1])


def test_separate_flags_with_none_returns_empty_dict():
    assert separate_flags(None) == {}


def test_separate_flags_returns_per_bit_boolean_arrays():
    flags = np.array([0, 1, 2, 3, 4], dtype=np.uint64)
    sep = separate_flags(flags)
    # bit 0 is set on 1 and 3; bit 1 on 2 and 3; bit 2 on 4.
    assert set(sep.keys()) == {0, 1, 2}
    assert sep[0].tolist() == [False, True, False, True, False]
    assert sep[1].tolist() == [False, False, True, True, False]
    assert sep[2].tolist() == [False, False, False, False, True]


def test_separate_flags_skips_bits_with_no_set_values():
    flags = np.array([1, 1, 1], dtype=np.uint64)  # only bit 0
    sep = separate_flags(flags)
    assert list(sep.keys()) == [0]


def test_combine_flags_with_no_flags_returns_zeros():
    λ_out = jnp.linspace(0.0, 1.0, 5)
    λ = jnp.linspace(0.0, 1.0, 11)
    out = combine_flags(λ_out, λ, None)
    assert out.shape == (5,)
    assert out.dtype == np.uint64
    assert np.all(out == 0)


def test_combine_flags_propagates_set_bits_to_nearby_pixels():
    λ_out = jnp.array([0.0, 0.25, 0.5, 0.75, 1.0])
    λ = jnp.array([0.24, 0.5, 0.9])
    flags = np.array([1, 2, 4], dtype=np.uint64)

    out = combine_flags(λ_out, λ, flags)
    assert out.shape == (5,)
    # bit 0 (value 1) → only at output index where flagged input is within one
    # pixel width (0.25). 0.24 is within [0, 0.5] (pixel width = 0.25) of
    # output index 1.
    assert (out[1] & 1) > 0
    # bit 1 (value 2) → input at 0.5, nearest output index 2.
    assert (out[2] & 2) > 0
    # bit 2 (value 4) → input at 0.9, nearest output index 4 (last is forced
    # False by hstack).
    assert (out[3] & 4) > 0
