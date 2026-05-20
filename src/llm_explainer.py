"""Compatibility wrapper for copilot_core.llm_explainer."""

import sys

from copilot_core import llm_explainer as _impl

sys.modules[__name__] = _impl
