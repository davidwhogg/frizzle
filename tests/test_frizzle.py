import jax.numpy as jnp
import numpy as np
import pytest

from frizzle import (
    ATCinvA,
    _post_rmatvec,
    _pre_matvec,
    frizzle,
    matvec,
    rmatvec,
)


def _simple_inputs(n_in=64, n_out=33, λ_min=0.1, λ_max=1.0, seed=0):
    rng = np.random.default_rng(seed)
    λ_out = jnp.linspace(λ_min, λ_max, n_out)
    λ = jnp.linspace(λ_min, λ_max, n_in)
    flux = jnp.array(1.0 + 0.1 * rng.standard_normal(n_in))
    ivar = jnp.full(n_in, 100.0)
    return λ_out, λ, flux, ivar


def test_frizzle_basic_call_recovers_constant_signal():
    λ_out, λ, _, ivar = _simple_inputs()
    flux = jnp.ones_like(λ)

    y_star, C_star, flags_star, meta = frizzle(λ_out, λ, flux, ivar)

    assert y_star.shape == λ_out.shape
    assert C_star.shape == (λ_out.size, λ_out.size)
    assert flags_star.shape == λ_out.shape
    # The combined spectrum of a flat line should also be flat.
    assert jnp.allclose(y_star, 1.0, atol=5e-2)
    # Without flags supplied, all output flags should be zero.
    assert int(jnp.sum(flags_star)) == 0
    # censor_missing_regions=True by default → meta has the no_data_mask key.
    assert "no_data_mask" in meta


def test_frizzle_defaults_for_ivar_and_mask():
    λ_out, λ, flux, _ = _simple_inputs()
    y_star, C_star, _, _ = frizzle(λ_out, λ, flux, ivar=None, mask=None)
    assert y_star.shape == λ_out.shape
    # With ivar=None (all ones) the diagonal of C_star should be finite and
    # positive in regions with data.
    assert jnp.all(jnp.isfinite(C_star))


def test_frizzle_with_explicit_n_modes():
    λ_out, λ, flux, ivar = _simple_inputs(n_in=80, n_out=41)
    y_star, _, _, _ = frizzle(λ_out, λ, flux, ivar, n_modes=11)
    assert y_star.shape == λ_out.shape


def test_frizzle_propagates_flags():
    λ_out = jnp.linspace(0.0, 1.0, 9)
    λ = jnp.linspace(0.0, 1.0, 33)
    flux = jnp.ones_like(λ)
    ivar = jnp.ones_like(λ)
    flags = np.zeros(λ.shape, dtype=np.uint64)
    flags[16] = 1  # bit 0 set near the middle
    flags[5] = 4  # bit 2 set near a quarter

    _, _, flags_star, _ = frizzle(λ_out, λ, flux, ivar, flags=flags)
    assert flags_star.dtype == flags.dtype
    # at least some output pixel got the flag.
    assert int(np.sum(flags_star & 1)) >= 1
    assert int(np.sum(flags_star & 4)) >= 1


def test_frizzle_censor_missing_regions_true_inserts_nans():
    # Carve a gap in the input wavelengths so the closest input is far from
    # the corresponding output pixel.
    λ_out = jnp.linspace(0.0, 1.0, 21)
    λ = jnp.concatenate([
        jnp.linspace(0.0, 0.3, 30),
        jnp.linspace(0.7, 1.0, 30),
    ])
    flux = jnp.ones_like(λ)
    ivar = jnp.ones_like(λ)

    y_star, C_star, _, meta = frizzle(
        λ_out, λ, flux, ivar, censor_missing_regions=True
    )
    # The middle of the spectrum has no data → at least one NaN.
    assert jnp.any(jnp.isnan(y_star))
    # And those positions get inf variance.
    assert jnp.any(jnp.isinf(C_star))
    assert "no_data_mask" in meta
    assert bool(jnp.any(meta["no_data_mask"]))


def test_frizzle_censor_missing_regions_false_returns_finite_meta():
    λ_out, λ, flux, ivar = _simple_inputs()
    y_star, C_star, _, meta = frizzle(
        λ_out, λ, flux, ivar, censor_missing_regions=False
    )
    # When censoring is off, no_data_mask should not be added to meta.
    assert "no_data_mask" not in meta
    assert jnp.all(jnp.isfinite(y_star))


def test_frizzle_with_user_mask_drops_pixels():
    λ_out, λ, flux, ivar = _simple_inputs(n_in=64)
    mask = np.zeros(λ.size, dtype=bool)
    mask[::2] = True  # mask every other pixel
    y_star, _, _, _ = frizzle(λ_out, λ, flux, ivar, mask=mask)
    assert y_star.shape == λ_out.shape


def test_pre_matvec_produces_hermitian_symmetric_output():
    p = 7
    rng = np.random.default_rng(1)
    c = jnp.asarray(rng.standard_normal(p))
    f = _pre_matvec(c, p)
    assert f.shape == (p,)
    # Hermitian symmetry: f == conj(flip(f)).
    assert jnp.allclose(f, jnp.conj(jnp.flip(f)))


def test_post_rmatvec_extracts_real_and_imag_parts():
    p = 7
    rng = np.random.default_rng(2)
    f = jnp.asarray(rng.standard_normal(p) + 1j * rng.standard_normal(p))
    out = _post_rmatvec(f, p)
    assert out.shape == (p,)
    # First half is the real parts; second half is the imag parts of the tail.
    assert jnp.allclose(out[: p // 2 + 1], jnp.real(f[: p // 2 + 1]))
    assert jnp.allclose(out[p // 2 + 1 :], jnp.imag(f[-(p - p // 2 - 1) :]))


def test_matvec_and_rmatvec_shapes():
    p = 9
    rng = np.random.default_rng(2)
    c = jnp.asarray(rng.standard_normal(p))
    x = jnp.linspace(-jnp.pi, jnp.pi, 17)
    y = matvec(c, x, p)
    assert y.shape == x.shape
    assert jnp.all(jnp.isreal(y))

    f = jnp.asarray(rng.standard_normal(x.size))
    out = rmatvec(f, x, p)
    assert out.shape == (p,)


def test_atcinva_is_self_consistent():
    # ATCinvA(c) should equal rmatvec(matvec(c)).
    p = 5
    c = jnp.asarray(np.random.default_rng(3).standard_normal(p))
    x = jnp.linspace(-jnp.pi, jnp.pi, 13)
    expected = rmatvec(matvec(c, x, p), x, p)
    got = ATCinvA(c, x, p)
    assert jnp.allclose(got, expected)
