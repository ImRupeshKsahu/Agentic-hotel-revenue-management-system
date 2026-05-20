"""Compatibility wrapper for pricing_core.engine."""

import sys

from pricing_core import engine as _impl

sys.modules[__name__] = _impl
