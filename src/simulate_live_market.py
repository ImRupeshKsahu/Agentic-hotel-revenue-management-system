"""Compatibility wrapper for pms_core.live_ledger."""

import sys

from pms_core import live_ledger as _impl

sys.modules[__name__] = _impl
