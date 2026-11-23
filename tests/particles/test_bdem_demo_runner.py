"""Manual high-fidelity BDEM demo driver; excluded from normal test collection."""
import os
import sys
from pathlib import Path

import pytest

import genesis  # Load native dependencies through the normal test-harness order.


@pytest.mark.skipif(not os.environ.get("BDEM_DEMO_ARM"), reason="manual BDEM demo driver")
def test_bdem_demo_arm():
    diagnosis = Path(__file__).resolve().parents[3] / "AGSim-Genesis-053" / "diagnosis"
    sys.path.insert(0, str(diagnosis))
    import bdem_demo

    os.environ["BDEM_PYTEST_DRIVER"] = "1"
    bdem_demo.run(os.environ["BDEM_DEMO_ARM"])
