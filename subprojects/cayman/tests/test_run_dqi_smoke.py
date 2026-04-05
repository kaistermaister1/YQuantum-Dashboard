"""Smoke tests for ``run_dqi`` (default backend: Nexus Selene).

Run: ``python -m unittest tests.test_run_dqi_smoke -v``

Overrides: ``DQI_SMOKE_EXECUTION=local`` (on-machine Selene), ``nexus_helios`` +
``DQI_NEXUS_HELIOS_SYSTEM``; ``DQI_NEXUS_TIMEOUT`` (seconds, default 600).
"""

from __future__ import annotations

import os
import sys
import unittest
import uuid
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dqi_core import import_guppylang_with_workaround
from src.run_dqi import run_dqi, run_dqi_with_details

try:
    from qnexus.exceptions import AuthenticationError as _QnxAuthenticationError
except ImportError:  # pragma: no cover
    _QnxAuthenticationError = None

try:
    import_guppylang_with_workaround()
    _guppy_ok = True
except (ImportError, OSError, RuntimeError):
    _guppy_ok = False


def _skip_if_selene_env_missing(exc: BaseException) -> None:
    """Selene uses zig to compile; Windows needs MSVC + Windows SDK in PATH."""
    msg = str(exc)
    if "WindowsSdkNotFound" in msg or "zig command failed" in msg:
        raise unittest.SkipTest(
            "Selene emulator build skipped: install Visual Studio Build Tools with "
            "'Desktop development with C++' (includes Windows SDK), then retry."
        ) from exc


def _maybe_skip_from_nexus_or_selene(exc: BaseException) -> None:
    if _QnxAuthenticationError is not None and isinstance(exc, _QnxAuthenticationError):
        raise unittest.SkipTest("Nexus authentication failed.") from exc
    if isinstance(exc, ImportError):
        msg = str(exc).lower()
        if "qnexus" in msg or "nexus" in msg:
            raise unittest.SkipTest("qnexus import failed (check cayman requirements.txt).") from exc
    if isinstance(exc, RuntimeError):
        _skip_if_selene_env_missing(exc)


def _smoke_execution() -> str:
    raw = os.environ.get("DQI_SMOKE_EXECUTION", "nexus_selene").strip().lower().replace("-", "_")
    if raw in ("local", "selene", "nexus_selene", "nexus_helios"):
        return raw if raw != "selene" else "local"
    raise unittest.SkipTest(
        f"Invalid DQI_SMOKE_EXECUTION={raw!r}; use local, nexus_selene, or nexus_helios"
    )


def _smoke_run_kwargs() -> dict:
    execution = _smoke_execution()
    if execution == "local":
        return {"execution": "local"}
    tag = uuid.uuid4().hex[:12]
    timeout = float(os.environ.get("DQI_NEXUS_TIMEOUT", "600"))
    kw: dict = {
        "execution": execution,
        "nexus_hugr_name": f"dqi-smoke-{tag}",
        "nexus_job_name": f"dqi-smoke-{tag}",
        "nexus_timeout": timeout,
    }
    if execution == "nexus_helios":
        kw["nexus_helios_system"] = os.environ.get("DQI_NEXUS_HELIOS_SYSTEM", "Helios-1")
    return kw


@unittest.skipIf(
    not _guppy_ok,
    "guppylang unavailable (not installed or wasmtime failed to load)",
)
class TestRunDqiSmoke(unittest.TestCase):
    def test_run_dqi_returns_solution_and_value(self) -> None:
        Q = np.array([[1.1, -0.25], [-0.25, -0.4]], dtype=float)
        try:
            x, value = run_dqi(
                Q,
                p=1,
                shots=128,
                seed=3,
                max_qubits=8,
                **_smoke_run_kwargs(),
            )
        except Exception as e:
            _maybe_skip_from_nexus_or_selene(e)
            raise
        self.assertEqual(x.shape, (2,))
        self.assertTrue(np.all((x == 0.0) | (x == 1.0)))
        self.assertTrue(np.isfinite(value))

    def test_run_dqi_with_quboblock_metadata(self) -> None:
        from src.qubo_block import QuboBlock

        q = np.array([[0.3, 0.2], [0.2, -0.5]], dtype=float)
        block = QuboBlock(
            package_index=0,
            Q=q,
            n_coverage=2,
            n_slack=0,
            coverage_offset=0,
            penalty_weight=1.0,
            constant_offset=1.25,
        )
        try:
            x, value, meta = run_dqi_with_details(
                block,
                p=1,
                shots=96,
                seed=2,
                max_qubits=8,
                **_smoke_run_kwargs(),
            )
        except Exception as e:
            _maybe_skip_from_nexus_or_selene(e)
            raise
        self.assertEqual(x.shape, (2,))
        self.assertIsNotNone(meta.coverage_bits)
        self.assertEqual(len(meta.coverage_bits or ""), 2)
        self.assertAlmostEqual(meta.constant_offset, 1.25)
        self.assertTrue(np.isfinite(value))


if __name__ == "__main__":
    unittest.main()
