API reference
=============

The user-facing API is a single function, :func:`frizzle.frizzle`. The helpers
in :mod:`frizzle.utils` are documented below for completeness.

.. currentmodule:: frizzle

frizzle
-------

.. autofunction:: frizzle

frizzle.utils
-------------

.. currentmodule:: frizzle.utils

.. autofunction:: get_modes
.. autofunction:: check_inputs
.. autofunction:: combine_flags
.. autofunction:: separate_flags

frizzle.test_utils
------------------

Synthetic-data helpers used throughout the documentation and tests.

.. currentmodule:: frizzle.test_utils

.. autofunction:: make_one_dataset
.. autofunction:: true_spectrum
