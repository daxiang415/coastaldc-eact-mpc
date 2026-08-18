import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from coastaldc_env import CoastalDCContinuousEnv  # noqa: E402


@pytest.fixture
def env():
    return CoastalDCContinuousEnv(country="JPN", seed=0)
