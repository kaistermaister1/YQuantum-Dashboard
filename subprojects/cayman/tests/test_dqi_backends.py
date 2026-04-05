"""Tests for DQI execution backend name normalization."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dqi_backends import normalize_execution


class TestNormalizeExecution(unittest.TestCase):
    def test_local_aliases(self) -> None:
        self.assertEqual(normalize_execution("local"), "local")
        self.assertEqual(normalize_execution("selene"), "local")
        self.assertEqual(normalize_execution("SELENE"), "local")

    def test_nexus(self) -> None:
        self.assertEqual(normalize_execution("nexus_selene"), "nexus_selene")
        self.assertEqual(normalize_execution("nexus-selene"), "nexus_selene")
        self.assertEqual(normalize_execution("nexus_helios"), "nexus_helios")

    def test_unknown(self) -> None:
        with self.assertRaises(ValueError):
            normalize_execution("ibm")


if __name__ == "__main__":
    unittest.main()
