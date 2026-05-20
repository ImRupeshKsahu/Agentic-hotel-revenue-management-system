"""Compatibility wrapper for copilot_core.pricing_agent."""

import sys

from copilot_core import pricing_agent as _impl

sys.modules[__name__] = _impl
