.. |jax| replace:: ``jax``
.. _jax: https://github.com/jax-ml/jax
.. |finufft| replace:: ``finufft``
.. _finufft: https://github.com/flatironinstitute/finufft
.. |jax-finufft| replace:: ``jax-finufft``
.. _jax-finufft: https://github.com/flatironinstitute/jax-finufft

``frizzle``
===========

.. rst-class:: lead

   **Combine spectra by forward modeling.**

``frizzle`` takes spectra from different epochs and redshifts, and produces a
combined spectrum on a user-specified wavelength grid, without ever interpolating
the data. The combined spectra has *uncorrelated noise* and *honest uncertainties*,
even when the input spectra have bad pixels, gaps, or are undersampled. It
leverages |jax|_ and |finufft|_ (through the |jax-finufft|_ bindings) for speed,
automatic differentiation, and GPU acceleration.


.. code-block:: bash

   uv add frizzle


A minimal example
-----------------

The snippet below generates eight synthetic, Doppler-shifted spectra with
partial bad pixels, and combines them onto a uniform output grid.

First, let's generate some fake data:

.. raw:: html

    <details class="setup-details">
    <summary> Show data-generation code</summary>

.. plot::
    :context: reset
    :include-source:
    :nofigs:

    import numpy as np
    import matplotlib.pyplot as plt
    from frizzle.test_utils import make_one_dataset, true_spectrum

    R = 1.35e5  # resolving power (lambda / delta_lambda)
    x_min, x_max = 8.7000, 8.7025 # log-wavelength range

    # eight synthetic spectra at slightly different Doppler shifts
    xs, ys, ivars, bs, delta_xs, line_args = make_one_dataset(
        dx=1 / R, snr=12, random_seed=17, x_min=x_min, x_max=x_max,
    )

    # output wavelength grid
    λ_out = np.arange(x_min + 1 / R, x_max, 1 / R)

    # concatenate everything into 1-D arrays for frizzle
    λ    = np.hstack([xs - dx for dx in delta_xs])
    flux  = np.hstack(ys)
    ivar = np.hstack(ivars)
    mask = ~np.hstack(bs).astype(bool)   # True = drop this pixel

.. raw:: html

    </details>

Then combine the spectra:

.. plot::
    :context:
    :include-source:
    :nofigs:

    from frizzle import frizzle

    y_star, C_star, flags, meta = frizzle(λ_out, λ, flux, ivar, mask)

And plot the result:

.. raw:: html

    <details class="setup-details">
    <summary> Show plotting code</summary>

.. plot::
    :context:
    :include-source:
    :nofigs:

    fig, ax = plt.subplots(figsize=(10, 4))
    for j in range(len(ys)):
        ax.step(
            xs - delta_xs[j],
            ys[j],
            where="mid", color="0.7", lw=0.6, alpha=0.6
        )
    sigma = np.sqrt(np.diag(C_star))
    ax.fill_between(
        λ_out,
        y_star - sigma,
        y_star + sigma,
        color="C3", alpha=0.25, ec="none"
    )
    ax.step(
        λ_out,
        y_star,
        where="mid", color="C3", lw=1.2, label=r"$\mathtt{frizzle}$"
    )
    ax.plot(
        λ_out,
        true_spectrum(λ_out, 0., *line_args),
        color="k", lw=0.7, ls="--", label="Truth"
    )
    ax.set_xlim(np.min(λ), np.max(λ))
    ax.set_xlabel(r"$\ln\,\lambda$")
    ax.set_ylabel(r"flux, $y$")
    ax.set_ylim(-0.1, 1.2)
    ax.legend(frameon=False, loc="lower left")

.. raw:: html

    </details>

.. plot::
    :context:
    :include-source: false

    plt.close("all")
    fig, ax = plt.subplots(figsize=(10, 4))
    for j in range(len(ys)):
        ax.step(
            xs - delta_xs[j],
            ys[j],
            where="mid", color="0.7", lw=0.6, alpha=0.6
        )
    sigma = np.sqrt(np.diag(C_star))
    ax.fill_between(
        λ_out,
        y_star - sigma,
        y_star + sigma,
        color="C3", alpha=0.25, ec="none"
    )
    ax.step(
        λ_out,
        y_star,
        where="mid", color="C3", lw=1.2, label=r"$\mathtt{frizzle}$"
    )
    ax.plot(
        λ_out,
        true_spectrum(λ_out, 0., *line_args),
        color="k", lw=0.7, ls="--", label="Truth"
    )
    ax.set_xlim(np.min(λ), np.max(λ))
    ax.set_xlabel(r"$\ln\,\lambda$")
    ax.set_ylabel(r"flux, $y$")
    ax.set_ylim(-0.1, 1.2)
    ax.legend(frameon=False, loc="lower left")

The combined spectrum (red) tracks the underlying truth (dashed) within its
uncertainty envelope, even though every input epoch (gray) is noisy and
shifted.


A closer look at the residuals
------------------------------

The residuals between the combined spectrum and truth, normalized by the
returned uncertainty, follow a unit Gaussian — confirming that the reported
covariance is honest:

.. plot::
    :context: close-figs
    :include-source:

    from scipy.stats import norm

    z = (
        (y_star - true_spectrum(λ_out, 0., *line_args))
    /   np.sqrt(np.diag(C_star))
    )

    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(-3, 3, 30)
    ax.hist(z, bins=bins, color="0.3", alpha=0.8)
    ax.plot(
        bins,
        norm.pdf(bins) * len(z) * (bins[1] - bins[0]),
        color="C3", lw=1.5, label=r"$\mathcal{N}(0, 1)$"
    )
    ax.set_xlabel(r"$z = (y_\star - y_\mathrm{true}) / \sigma$")
    ax.set_ylabel("count")
    ax.legend(frameon=False)

Empirical covariance
--------------------

A more demanding test: simulate many realizations of the same scene, and create
two combined spectra: one with ``frizzle``, and one with cubic interpolation.

Let's compare how correlated the residuals are between neighbouring pixels.
Here you want to compare the filled markers (``frizzle``) to the open markers
(cubic interpolation).

.. raw:: html

    <details class="setup-details">
    <summary> Show covariance code</summary>

.. plot::
    :context: close-figs
    :include-source:
    :nofigs:

    import scipy.interpolate as interp

    def standard_practice(xs, ys, bs, dxs, λ_out, kind="cubic"):
        """Interpolate each epoch onto λ_out and average."""
        N = len(ys)
        yprimes = np.full((N, len(λ_out)), np.nan)
        for j in range(N):
            use = bs[j] > 0.5
            f = interp.interp1d(xs[use] - dxs[j], ys[j][use],
                                kind=kind, fill_value=np.nan, bounds_error=False)
            yprimes[j] = f(λ_out)
        return np.nanmean(yprimes, axis=0)

    def covariances(resids, n_lags=8):
        lags = np.arange(n_lags)
        var = np.full(n_lags, np.nan)
        var[0] = np.nanmean(resids * resids)
        for lag in lags[1:]:
            var[lag] = np.nanmean(resids[lag:] * resids[:-lag])
        return lags, var

    def average_covars(n_trials, dx, snr, n_lags=8):
        cov_friz = np.zeros(n_lags)
        cov_std = np.zeros(n_lags)
        for trial in range(n_trials):
            xs_t, ys_t, ivars_t, bs_t, dxs_t, line_args_t = make_one_dataset(
                dx=dx, snr=snr, x_min=x_min, x_max=x_max, random_seed=trial,
            )
            mask_t = ~np.hstack(bs_t).astype(bool)
            y_friz, *_ = frizzle(
                λ_out,
                np.hstack([xs_t - d for d in dxs_t]),
                np.hstack(ys_t), np.hstack(ivars_t), mask_t,
            )
            y_std = standard_practice(xs_t, ys_t, bs_t, dxs_t, λ_out)
            y_true = true_spectrum(λ_out, 0., *line_args_t)
            _, c1 = covariances(y_friz - y_true, n_lags)
            _, c2 = covariances(y_std  - y_true, n_lags)
            cov_friz += c1
            cov_std  += c2
        return np.arange(n_lags), cov_friz / n_trials, cov_std / n_trials

    np.random.seed(42)
    n_trials = 32
    lags, c_friz_os, c_std_os = average_covars(n_trials, dx=1 / R, snr=12)
    lags, c_friz_us, c_std_us = average_covars(n_trials, dx=2 / R, snr=18)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.axhline(0., color="k", lw=0.5)
    ax.plot(lags, c_friz_os, "o", color="C3", ms=6,
            label=r"$\mathtt{frizzle}$, over-sampled")
    ax.plot(lags, c_std_os, "o", color="C3", ms=6, mfc="none",
            label="cubic interp, over-sampled")
    ax.plot(lags, c_friz_us, "s", color="C0", ms=6,
            label=r"$\mathtt{frizzle}$, under-sampled")
    ax.plot(lags, c_std_us, "s", color="C0", ms=6, mfc="none",
            label="cubic interp, under-sampled")
    ax.set_xlabel("lag (output pixels)", fontsize=20)
    ax.set_ylabel("covariance", fontsize=20)
    ax.tick_params(labelsize=16)
    ax.legend(frameon=False, fontsize=16)

.. raw:: html

    </details>

.. plot::
    :context: close-figs
    :include-source: false

    plt.close("all")
    np.random.seed(42)
    n_trials = 32
    lags, c_friz_os, c_std_os = average_covars(n_trials, dx=1 / R, snr=12)
    lags, c_friz_us, c_std_us = average_covars(n_trials, dx=2 / R, snr=18)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.axhline(0., color="k", lw=0.5)
    ax.plot(lags, c_friz_os, "o", color="C3", ms=6,
            label=r"$\mathtt{frizzle}$, over-sampled")
    ax.plot(lags, c_std_os, "o", color="C3", ms=6, mfc="none",
            label="cubic interp, over-sampled")
    ax.plot(lags, c_friz_us, "s", color="C0", ms=6,
            label=r"$\mathtt{frizzle}$, under-sampled")
    ax.plot(lags, c_std_us, "s", color="C0", ms=6, mfc="none",
            label="cubic interp, under-sampled")
    ax.set_xlabel("lag (output pixels)", fontsize=20)
    ax.set_ylabel("covariance", fontsize=20)
    ax.tick_params(labelsize=16)
    ax.legend(frameon=False, fontsize=16)

The spectra combined with cubic interpolation have correlated residuals, which
appear as very weak absorption lines in the output spectrum (see `Figure 5
<https://arxiv.org/abs/2403.11011>`_) -- exactly the thing
that astronomers are often trying to measure. It also means a standard chi-squared
analysis using the diagonal of the covariance matrix will be biased low, because
it doesn't account for those correlations.
The situation gets worse for under-sampled spectra (e.g., JWST, APOGEE).

The combined spectrum with ``frizzle`` have essentially uncorrelated uncertainties:
the covariance drops to zero at non-zero lags. That means that a chi-squared
analysis using the diagonal of the covariance matrix is actually correct, and
the uncertainties are honest. ``frizzle`` performs equally well for both
under-sampled and over-sampled input spectra.


Common options
--------------

The four keyword arguments you'll reach for most often:

**Bad pixel masks.** Pass ``mask=...`` as a boolean array where ``True`` marks
pixels to ignore when combining spectra. Those masked pixels won't contribute to
the combined spectrum, but flags from those pixels will be propagated.

.. code-block:: python

   y_star, C_star, _, _ = frizzle(x_out, lam, flux, ivar, mask=mask)

**Bitwise flags.** Pass ``flags=...`` to propagate per-pixel bitwise flags
(cosmic rays, saturation, etc.) onto the output grid. An output pixel
inherits a bit if any input pixel within one output-pixel width has it set:

.. code-block:: python

   y_star, C_star, flags_out, _ = frizzle(
       x_out, lam, flux, ivar, mask=mask, flags=flags_in,
   )

**Number of Fourier modes.** ``frizzle`` fits a Fourier series internally;
``n_modes`` controls how flexible the model is. The default
(``min(len(x_out), n_unmasked_pixels)``) is almost always what you want, but
you can dial it down for speed:

.. code-block:: python

   y_star, *_ = frizzle(x_out, lam, flux, ivar, mask=mask, n_modes=501)

**Censoring gaps.** By default, output pixels with no nearby input data are
set to ``NaN`` (with infinite variance) to prevent the model from
extrapolating into the void. These extrapolations are merely cosmetic: the
combined covariance matrix will have large uncertainties in those regions.
However, astronomers appreciate cosmetics, so by default we censor those extrapolations.
To see the raw model output instead, pass
``censor_missing_regions=False``.

Next steps
----------

.. toctree::
    :maxdepth: 1

    api

Reference
---------

See the paper (`arxiv <https://arxiv.org/abs/2403.11011>`_ / `ADS <https://ui.adsabs.harvard.edu/abs/2024arXiv240311011H/abstract>`_) for the mathematical details
and performance benchmarks.

.. raw:: html

    <details class="setup-details">
    <summary> Show bibtex entry</summary>


.. code-block:: bibtex

    @ARTICLE{frizzle,
           author = {{Hogg}, David W. and {Casey}, Andrew R.},
            title = "{Frizzle: Combining spectra or images by forward modeling}",
          journal = {arXiv e-prints},
         keywords = {Astrophysics - Instrumentation and Methods for Astrophysics},
             year = 2024,
            month = mar,
              eid = {arXiv:2403.11011},
            pages = {arXiv:2403.11011},
              doi = {10.48550/arXiv.2403.11011},
    archivePrefix = {arXiv},
           eprint = {2403.11011},
     primaryClass = {astro-ph.IM},
           adsurl = {https://ui.adsabs.harvard.edu/abs/2024arXiv240311011H},
          adsnote = {Provided by the SAO/NASA Astrophysics Data System}
    }

.. raw:: html

    </details>


Authors
-------

- **David W. Hogg** — NYU, MPIA, Flatiron
- **Andy Casey** — Monash, Flatiron
