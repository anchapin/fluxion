"""measures — root package for Fluxion AOT measures (Issue #1814).

Drop ``*.py`` files into a sub-directory of this package (typically
``measures/examples/``) and they will be picked up by
``fluxion apply-measures --measures <dir>``. Each ``.py`` file may declare
any number of :class:`fluxion.FluxionMeasure` subclasses; only the
concrete (non-abstract) ones are discovered.

The root package itself does not export any measures — it is a namespace.
"""
