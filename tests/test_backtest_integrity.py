"""Stage 8 tests: the pre-Test lock and the gated evaluation-only Test loader (temporary files only)."""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stage8_fixtures  # noqa: E402,F401

from regime_radar import backtest_integrity as bi  # noqa: E402
from regime_radar import regime_exploration as rx  # noqa: E402


class TestGatedLoader(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        (self.tmp / "src" / "regime_radar").mkdir(parents=True)
        (self.tmp / "scripts").mkdir()
        (self.tmp / "src" / "regime_radar" / "m.py").write_text("x = 1\n")
        (self.tmp / "scripts" / "10_backtest_portfolios.py").write_text("print(1)\n")
        self.splits = self.tmp / "splits"
        self.splits.mkdir()
        idx = pd.bdate_range("2020-01-02", periods=5)
        hashes = {}
        for name, cols in (("test_features.csv", ["f1"]), ("test_returns.csv", ["SPY"])):
            pd.DataFrame(np.ones((5, 1)), index=pd.Index(idx, name="date"), columns=cols).to_csv(self.splits / name)
            hashes[name] = bi.sha256(self.splits / name)
        (self.splits / "split_manifest.json").write_text(json.dumps({"split_files": hashes}))
        self.cfg = self.tmp / "cfg.json"
        self.cfg.write_text("{}")
        self.artifacts = {"cfg": self.cfg}
        snap = bi.snapshot(self.tmp, self.artifacts)
        self.lock = self.tmp / bi.LOCK_NAME
        bi.write_json(bi.build_lock({"hashes": snap, "protocol": {"x": 1}}, {}, True, "ok"), self.lock)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def snap(self):
        return bi.snapshot(self.tmp, self.artifacts)

    def test_valid_lock_opens_test(self):
        t = bi.load_test_split(self.splits, self.lock, self.snap())
        self.assertEqual(len(t.features), 5)
        self.assertEqual(t.files_read, list(bi.TEST_FILES))

    def test_missing_lock_refused(self):
        with self.assertRaises(bi.TestAccessError):
            bi.load_test_split(self.splits, self.tmp / "nope.json", self.snap())

    def test_unauthorized_lock_refused(self):
        bi.write_json(bi.build_lock({"hashes": self.snap()}, {}, False, "preflight failed"), self.lock)
        with self.assertRaises(bi.TestAccessError):
            bi.load_test_split(self.splits, self.lock, self.snap())

    def test_code_change_after_lock_refused(self):
        (self.tmp / "src" / "regime_radar" / "m.py").write_text("x = 2\n")
        with self.assertRaises(bi.TestAccessError):
            bi.load_test_split(self.splits, self.lock, self.snap())

    def test_config_change_after_lock_refused(self):
        self.cfg.write_text('{"changed": true}')
        with self.assertRaises(bi.TestAccessError):
            bi.load_test_split(self.splits, self.lock, self.snap())

    def test_edited_lock_refused(self):
        lock = json.loads(self.lock.read_text())
        lock["frozen"]["protocol"]["x"] = 2
        self.lock.write_text(json.dumps(lock))
        with self.assertRaises(bi.TestAccessError):
            bi.load_test_split(self.splits, self.lock, self.snap())

    def test_changed_test_file_refused(self):
        (self.splits / "test_returns.csv").write_text("date,SPY\n2020-01-02,9\n")
        with self.assertRaises(bi.TestAccessError):
            bi.load_test_split(self.splits, self.lock, self.snap())

    def test_access_log_append_only(self):
        log = self.tmp / bi.ACCESS_LOG_NAME
        bi.append_access_log(log, {"mode": "first_evaluation"})
        bi.append_access_log(log, {"mode": "reproduction"})
        self.assertEqual([e["mode"] for e in bi.read_access_log(log)], ["first_evaluation", "reproduction"])

    def test_fingerprint_stable(self):
        a = bi.fingerprint({"b": 1, "a": [1.0, 2]})
        self.assertEqual(a, bi.fingerprint({"a": [1.0, 2], "b": 1}))

    def test_earlier_loaders_still_refuse_test(self):
        with self.assertRaises(ValueError):
            rx.load_stage4_data(self.splits, ("train", "validation", "test"))


if __name__ == "__main__":
    unittest.main()
