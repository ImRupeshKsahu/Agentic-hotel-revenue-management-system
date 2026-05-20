"""Compatibility wrapper for market_core.feed."""

import sys

from market_core import feed as _impl

sys.modules[__name__] = _impl
