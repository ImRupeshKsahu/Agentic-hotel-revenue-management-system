"""Compatibility wrapper for pricing_core.cancellation."""

import sys

from pricing_core import cancellation as _impl

sys.modules[__name__] = _impl
