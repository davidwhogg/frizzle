"""
Synthetic data helpers used by the tests, examples, and documentation.

These functions generate fake multi-epoch spectra of a star with absorption
lines drawn from a Poisson line list, with optional Doppler shifts, noise,
and bad pixels.
"""

import numpy as np
from scipy.special import wofz

c = 299792458.0  # m / s
sqrt2pi = np.sqrt(2.0 * np.pi)


def oned_gaussian(dxs, sigma):
    return np.exp(-0.5 * dxs ** 2 / sigma ** 2) / (sqrt2pi * sigma)


def oned_lorentzian(dxs, gamma):
    return gamma / (np.pi * (dxs ** 2 + gamma ** 2))


def oned_voigt(dxs, sigma, gamma):
    z = (dxs + 1j * gamma) / (sigma * np.sqrt(2))
    return np.real(wofz(z)) / (sigma * sqrt2pi * np.sqrt(2))


def get_profile_function(profile_type, profile_width):
    if profile_type.lower() == "gaussian":
        return lambda dxs: oned_gaussian(dxs, profile_width)
    if profile_type.lower() == "lorentzian":
        return lambda dxs: oned_lorentzian(dxs, profile_width)
    if profile_type.lower() == "voigt":
        if not isinstance(profile_width, (tuple, list)) or len(profile_width) != 2:
            raise ValueError("voigt profile_width must be a (sigma, gamma) tuple")
        sigma, gamma = profile_width
        return lambda dxs: oned_voigt(dxs, sigma, gamma)
    raise ValueError("profile_type must be 'gaussian', 'lorentzian', or 'voigt'")


def true_spectrum(xs, doppler, line_xs, line_ews, profile_func):
    """The noise-free template at log-wavelengths ``xs``, Doppler-shifted by ``doppler``."""
    return np.exp(
        -1.0
        * np.sum(
            line_ews[None, :] * profile_func(xs[:, None] - doppler - line_xs[None, :]),
            axis=1,
        )
    )


def _ivar(ys, continuum_ivar):
    return continuum_ivar / ys


def _noisy_true_spectrum(xs, doppler, line_xs, line_ews, profile_func, continuum_ivar):
    ys_true = true_spectrum(xs, doppler, line_xs, line_ews, profile_func)
    y_ivars = _ivar(ys_true, continuum_ivar)
    ys_noisy = ys_true + np.random.normal(size=xs.shape) / np.sqrt(y_ivars)
    return (ys_noisy, y_ivars)


def _badify(yy, badfrac):
    bady = 1.0 * yy
    bs = (np.random.uniform(size=len(bady)) > badfrac).astype(int)
    bs = np.minimum(bs, np.roll(bs, 1))
    bs = np.minimum(bs, np.roll(bs, -1))
    nbad = int(np.sum(bs < 0.5))
    if nbad > 0:
        bady[bs < 0.5] += 2.0 * np.random.uniform(size=nbad)
    return bs, bady


def make_one_dataset(
    dx,
    snr,
    N=8,
    x_min=8.7000,
    x_max=8.7025,
    R=1.35e5,
    lines_per_x=2.0e4,
    ew_max_x=3.0e-5,
    ew_power=5.0,
    badfrac=0.01,
    profile_type="gaussian",
    profile_width=None,
    random_seed=None,
    Delta_xs=None,
):
    """
    Generate one synthetic multi-epoch dataset.

    :param dx: Pixel spacing in log-wavelength.
    :param snr: Continuum signal-to-noise ratio per pixel.
    :param N: Number of epochs.
    :param x_min, x_max: Log-wavelength range.
    :param R: Spectral resolving power.
    :param lines_per_x: Mean line density per unit log-wavelength.
    :param ew_max_x: Max equivalent width.
    :param ew_power: Power-law index for the EW distribution.
    :param badfrac: Fraction of pixels marked bad.
    :param profile_type: 'gaussian', 'lorentzian', or 'voigt'.
    :param profile_width: Profile width(s); defaults to ``1/R``.
    :param random_seed: Seed for reproducibility.
    :param Delta_xs: Per-epoch Doppler shifts (defaults to a sinusoid).

    :returns: ``(xs, ys, y_ivars, bs, Delta_xs, line_args)`` where ``line_args``
        is a tuple ``(line_xs, line_ews, profile_func)`` accepted by
        :func:`true_spectrum`.
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    if profile_width is None:
        sigma_x = 1.0 / R
        if profile_type.lower() == "voigt":
            profile_width = (sigma_x, sigma_x)
        else:
            profile_width = sigma_x

    profile_func = get_profile_function(profile_type, profile_width)

    x_margin = 1.0e6 / c
    x_range = x_max - x_min + 2.0 * x_margin
    nlines = np.random.poisson(x_range * lines_per_x)
    line_xs = (x_min - x_margin) + x_range * np.random.uniform(size=nlines)
    line_ews = ew_max_x * np.random.uniform(size=nlines) ** ew_power

    if Delta_xs is None:
        Delta_xs = (3.0e4 / c) * np.cos(np.arange(N) / 3.0)

    continuum_ivar = snr ** 2

    xs = np.arange(x_min - 0.5 * dx, x_max + dx, dx)
    ys = np.zeros((N, len(xs)))
    y_ivars = np.zeros_like(ys)
    bs = np.zeros_like(ys).astype(int)

    for j in range(N):
        ys[j], y_ivars[j] = _noisy_true_spectrum(
            xs, Delta_xs[j], line_xs, line_ews, profile_func, continuum_ivar
        )
        bs[j], ys[j] = _badify(ys[j], badfrac)

    return xs, ys, y_ivars, bs, Delta_xs, (line_xs, line_ews, profile_func)
