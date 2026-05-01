import numpy as np
import pytest

from frizzle.test_utils import (
    _badify,
    _ivar,
    _noisy_true_spectrum,
    get_profile_function,
    make_one_dataset,
    oned_gaussian,
    oned_lorentzian,
    oned_voigt,
    true_spectrum,
)


def test_oned_gaussian_normalizes_to_one():
    sigma = 0.5
    xs = np.linspace(-10 * sigma, 10 * sigma, 5001)
    ys = oned_gaussian(xs, sigma)
    integral = np.trapezoid(ys, xs)
    assert pytest.approx(1.0, rel=1e-3) == integral
    # Peak is at zero and equals 1 / (sqrt(2 pi) sigma).
    assert ys.argmax() == len(xs) // 2


def test_oned_lorentzian_normalizes_to_one():
    gamma = 0.3
    xs = np.linspace(-1000 * gamma, 1000 * gamma, 100_001)
    ys = oned_lorentzian(xs, gamma)
    integral = np.trapezoid(ys, xs)
    assert pytest.approx(1.0, abs=2e-3) == integral


def test_oned_voigt_is_positive_and_peaks_at_zero():
    sigma, gamma = 0.4, 0.2
    xs = np.linspace(-5.0, 5.0, 1001)
    ys = oned_voigt(xs, sigma, gamma)
    assert np.all(ys > 0)
    # Peaks at zero.
    assert xs[ys.argmax()] == pytest.approx(0.0, abs=1e-2)
    # Symmetric around zero.
    np.testing.assert_allclose(ys, ys[::-1])


def test_get_profile_function_gaussian_matches_oned_gaussian():
    sigma = 0.5
    pf = get_profile_function("gaussian", sigma)
    xs = np.linspace(-2.0, 2.0, 100)
    np.testing.assert_allclose(pf(xs), oned_gaussian(xs, sigma))


def test_get_profile_function_lorentzian_matches_oned_lorentzian():
    gamma = 0.2
    pf = get_profile_function("LORENTZIAN", gamma)  # case-insensitive
    xs = np.linspace(-2.0, 2.0, 50)
    np.testing.assert_allclose(pf(xs), oned_lorentzian(xs, gamma))


def test_get_profile_function_voigt_requires_tuple():
    with pytest.raises(ValueError):
        get_profile_function("voigt", 0.5)


def test_get_profile_function_voigt_with_tuple_works():
    pf = get_profile_function("voigt", (0.3, 0.1))
    xs = np.linspace(-1.0, 1.0, 50)
    np.testing.assert_allclose(pf(xs), oned_voigt(xs, 0.3, 0.1))


def test_get_profile_function_unknown_type_raises():
    with pytest.raises(ValueError):
        get_profile_function("triangle", 0.5)


def test_true_spectrum_returns_unity_with_no_lines():
    xs = np.linspace(0.0, 1.0, 11)
    pf = get_profile_function("gaussian", 0.01)
    out = true_spectrum(xs, doppler=0.0, line_xs=np.array([]), line_ews=np.array([]), profile_func=pf)
    np.testing.assert_allclose(out, np.ones_like(xs))


def test_true_spectrum_with_line_dips_below_unity():
    xs = np.linspace(-0.5, 0.5, 101)
    pf = get_profile_function("gaussian", 0.05)
    out = true_spectrum(xs, doppler=0.0, line_xs=np.array([0.0]), line_ews=np.array([0.05]), profile_func=pf)
    assert out.min() < 1.0
    assert out.max() <= 1.0
    # Minimum should be near x=0.
    assert abs(xs[out.argmin()]) < 0.02


def test_ivar_helper():
    ys = np.array([0.5, 1.0, 2.0])
    out = _ivar(ys, continuum_ivar=10.0)
    np.testing.assert_allclose(out, np.array([20.0, 10.0, 5.0]))


def test_noisy_true_spectrum_returns_arrays_of_same_shape():
    rng = np.random.default_rng(0)
    np.random.seed(0)  # _noisy_true_spectrum uses np.random
    xs = np.linspace(0.0, 0.01, 200)
    pf = get_profile_function("gaussian", 1e-4)
    ys, y_ivars = _noisy_true_spectrum(
        xs, doppler=0.0,
        line_xs=np.array([0.005]),
        line_ews=np.array([1e-4]),
        profile_func=pf,
        continuum_ivar=100.0,
    )
    assert ys.shape == xs.shape
    assert y_ivars.shape == xs.shape
    # ivar is positive everywhere.
    assert np.all(y_ivars > 0)


def test_badify_returns_mask_and_perturbed_flux():
    np.random.seed(42)
    yy = np.ones(500)
    bs, bady = _badify(yy, badfrac=0.1)
    assert bs.shape == yy.shape
    assert bady.shape == yy.shape
    # Some pixels should be marked bad.
    assert np.any(bs == 0)
    # Bad pixels get inflated values; good pixels are untouched.
    np.testing.assert_allclose(bady[bs == 1], yy[bs == 1])
    assert np.all(bady[bs == 0] >= yy[bs == 0])


def test_badify_with_zero_badfrac_keeps_data_unchanged():
    np.random.seed(0)
    yy = np.linspace(0.5, 1.5, 100)
    bs, bady = _badify(yy.copy(), badfrac=0.0)
    np.testing.assert_allclose(bady, yy)
    assert np.all(bs == 1)


def test_make_one_dataset_returns_expected_shapes():
    dx = 1.0 / 1.35e5
    xs, ys, y_ivars, bs, Delta_xs, line_args = make_one_dataset(
        dx=dx, snr=20.0, N=4, random_seed=7,
    )
    assert ys.shape == (4, xs.size)
    assert y_ivars.shape == ys.shape
    assert bs.shape == ys.shape
    assert Delta_xs.shape == (4,)
    line_xs, line_ews, profile_func = line_args
    assert line_xs.shape == line_ews.shape
    assert callable(profile_func)


def test_make_one_dataset_seed_is_deterministic():
    kwargs = dict(dx=1.0 / 1.35e5, snr=20.0, N=2, random_seed=123)
    a = make_one_dataset(**kwargs)
    b = make_one_dataset(**kwargs)
    np.testing.assert_allclose(a[0], b[0])
    np.testing.assert_allclose(a[1], b[1])
    np.testing.assert_allclose(a[4], b[4])


def test_make_one_dataset_with_user_supplied_delta_xs():
    dx = 1.0 / 1.35e5
    Delta = np.array([0.0, 1e-5, -1e-5])
    _, _, _, _, Delta_out, _ = make_one_dataset(
        dx=dx, snr=10.0, N=3, random_seed=0, Delta_xs=Delta,
    )
    np.testing.assert_allclose(Delta_out, Delta)


def test_make_one_dataset_with_voigt_profile():
    dx = 1.0 / 1.35e5
    xs, ys, *_ = make_one_dataset(
        dx=dx, snr=20.0, N=2, random_seed=0,
        profile_type="voigt",
    )
    assert ys.shape == (2, xs.size)


def test_make_one_dataset_with_explicit_profile_width():
    dx = 1.0 / 1.35e5
    xs, ys, *_ = make_one_dataset(
        dx=dx, snr=20.0, N=2, random_seed=0,
        profile_type="lorentzian",
        profile_width=1e-5,
    )
    assert ys.shape == (2, xs.size)
