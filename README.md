<div align="Center">

# frizzle 

Combine spectra by forward modeling [(Hogg & Casey, 20xx)](https://arxiv.org/abs/2403.11011).

[![Test Status](https://github.com/davidwhogg/frizzle/actions/workflows/ci.yml/badge.svg)](https://github.com/davidwhogg/frizzle/actions/workflows/ci.yml)
[![Coverage Status](https://coveralls.io/repos/github/davidwhogg/frizzle/badge.svg?branch=main&service=github)](https://coveralls.io/github/davidwhogg/frizzle?branch=main)
[![Documentation Status](https://readthedocs.org/projects/frizzle/badge/?version=latest)](https://frizzle.readthedocs.io/en/latest/?badge=latest)

</div>

# Install

To get frizzy wit it:
```
uv add frizzle
```

# Getting Started

Combine eight Doppler-shifted spectra of the same source onto a common
output grid:

```python
import numpy as np
from frizzle import frizzle
from frizzle.test_utils import make_one_dataset

R = 1.35e5
x_min, x_max = 8.7000, 8.7025

# generate eight synthetic spectra at slightly different Doppler shifts
xs, ys, ivars, bs, delta_xs, _ = make_one_dataset(
    dx=1 / R, snr=12, random_seed=17, x_min=x_min, x_max=x_max,
)

# output wavelength grid
λ_out = np.arange(x_min + 1 / R, x_max, 1 / R)

# concatenate everything into 1-D arrays for frizzle
λ    = np.hstack([xs - dx for dx in delta_xs])
flux = np.hstack(ys)
ivar = np.hstack(ivars)
mask = ~np.hstack(bs).astype(bool)         # True = drop this pixel

# combine
y_star, C_star, flags, meta = frizzle(λ_out, λ, flux, ivar, mask)
```

See the [documentation](https://frizzle.readthedocs.io/) for a worked
example with plots, the most useful kwargs, and why forward modeling
beats interpolation.

# Authors
- **David W Hogg** (NYU) (MPIA) (Flatiron)
- **Andy Casey** (Monash) (Flatiron)


With contributions from:
- **Matt Daunt** (NYU);
- **Thomas Hilder** (Monash);
- **Adrian Price-Whelan** (Flatiron);
- the **Astronomical Data Group** at the Flatiron Institute; and 
- the **Inference Group** at Monash University.

