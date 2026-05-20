"""Compatibility wrapper for pricing_core.scenario."""

import sys

from pricing_core import scenario as _impl

sys.modules[__name__] = _impl
