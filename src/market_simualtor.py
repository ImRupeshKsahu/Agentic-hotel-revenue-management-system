"""Compatibility wrapper for market_core.simulator."""

import sys

from market_core import simulator as _impl

sys.modules[__name__] = _impl
