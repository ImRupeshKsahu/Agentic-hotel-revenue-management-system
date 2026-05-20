"""Compatibility wrapper for pms_core.snapshot."""

import sys

from pms_core import snapshot as _impl

sys.modules[__name__] = _impl
