"""Compatibility wrapper for pms_core.data_pipeline."""

import sys

from pms_core import data_pipeline as _impl

sys.modules[__name__] = _impl
