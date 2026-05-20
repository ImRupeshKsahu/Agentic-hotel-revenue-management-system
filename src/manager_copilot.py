"""Compatibility wrapper for copilot_core.manager."""

import sys

from copilot_core import manager as _impl

sys.modules[__name__] = _impl
