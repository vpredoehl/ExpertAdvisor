#!/usr/bin/env python3
"""Offline regressions for Phase 24X preflight safety and input evidence."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location(
    "preflight", Path(__file__).with_name("AnalyzeResourceQualificationPreflight.py"))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class PreflightTests(unittest.TestCase):
    def test_public_copy_only_and_preserves_null(self):
        dump = ("COPY other.experiment (id, path) FROM stdin;\n9\toutside\n\\.\n"
                "COPY public.experiment (id, path) FROM stdin;\n1\t\\N\n\\.\n"
                "SELECT 'not executed';\n")
        self.assertEqual(m.copy_rows(dump, "experiment"), [{"id": "1", "path": "\\N"}])
        self.assertEqual(m.copy_rows(dump, "model"), [])

    def test_malformed_copy_rejected(self):
        with self.assertRaises(ValueError):
            m.copy_rows("COPY public.experiment (id, path) FROM stdin;\n1\n", "experiment")

    def test_logs_outside_checkout_never_opened(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "checkout"
            root.mkdir()
            outside = Path(directory) / "production.log"
            outside.write_text("not a qualification input")
            (root / "alias.log").symlink_to(outside)
            (root / "private.log").write_text("real input")
            with patch.object(m, "ROOT", root):
                inventory = m.private_log_inventory([
                    {"train_log_path": str(outside), "infer_log_path": "alias.log"},
                    {"train_log_path": "private.log", "infer_log_path": "absent.log"}])
            self.assertEqual(inventory["counts"], {
                "outside_checkout_rejected": 2, "present_in_checkout": 1,
                "missing_in_checkout": 1})
            self.assertEqual(inventory["total_bytes"], 10)

    def test_existing_evidence_and_output_escape_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            area = Path(directory) / "area"
            with patch.object(m, "AREA", area), patch.object(m, "ROOT", Path(directory)):
                with self.assertRaises(ValueError):
                    m.Collector(Path(directory) / "outside")
                collector = m.Collector(area / "fresh")
                self.assertFalse(any(key.startswith(("PG", "LSTM", "FOREX"))
                                     for key in collector.env))
                with self.assertRaises(FileExistsError):
                    m.Collector(area / "fresh")

    def test_evidence_area_symlink_escape_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "checkout"
            outside = Path(directory) / "production"
            root.mkdir()
            outside.mkdir()
            area = root / "evidence"
            area.symlink_to(outside)
            with patch.object(m, "AREA", area), patch.object(m, "ROOT", root):
                with self.assertRaises(ValueError):
                    m.Collector(area / "fresh")
            self.assertFalse((outside / "fresh").exists())

    def test_swap_counters_not_pageins(self):
        self.assertEqual(m.vm_counters("Swapins: 123.\nSwapouts: 456.\nPageins: 99.\n"
                                       "Pageouts: 8.\n"),
                         {"Swapins": 123, "Swapouts": 456, "Pageouts": 8})


if __name__ == "__main__":
    unittest.main()
