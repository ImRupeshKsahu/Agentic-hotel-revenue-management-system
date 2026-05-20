"""Compatibility wrapper for market_core.competitor_simulator."""

import sys

from market_core import competitor_simulator as _impl

sys.modules[__name__] = _impl
