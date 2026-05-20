"""Compatibility wrapper for pricing_core.local_intel."""

import sys

from pricing_core import local_intel as _impl

sys.modules[__name__] = _impl
