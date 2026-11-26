"""Deterministic unit & adversarial tests for DISPATCH-06 kinematics_auth + synth_traces."""
from __future__ import annotations

import os
import statistics
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kinematics_auth import (
    FEATURE_NAMES,
    evaluate_seed,
    extract_features,
    fit_and_freeze_model,
)
from synth_traces import (
    CHANNELS,
    EDIT_KINDS,
    PROFILES,
    TRIGGER_KINDS,
    build_split_datasets,
    generate_trace,
)


class TestKinematicsAuth(unittest.TestCase):
    def test_01_trace_coordinate_and_schema_bounds(self) -> None:
        ds = build_split_datasets(seed=42323)
        for pid, splits in ds.items():
            for tr in splits["hold_gen"][:2] + splits["hold_mim"][:2]:
                self.assertGreater(len(tr.ticks), 100)
                self.assertGreater(len(tr.bursts), 40)
                prev_t = -1.0
                for tk in tr.ticks:
                    self.assertGreaterEqual(tk.x, 0.0)
                    self.assertLessEqual(tk.x, 1.0)
                    self.assertGreaterEqual(tk.y, 0.0)
                    self.assertLessEqual(tk.y, 1.0)
                    self.assertIn(tk.trigger_kind, TRIGGER_KINDS)
                    self.assertGreaterEqual(tk.ms, prev_t)
                    prev_t = tk.ms
                prev_b = -1.0
                for b in tr.bursts:
                    self.assertIn(b.channel, CHANNELS)
                    self.assertIn(b.edit_kind, EDIT_KINDS)
                    self.assertGreaterEqual(b.symbol_count, 1)
                    self.assertGreater(b.dwell_ms, 0.0)
                    self.assertGreater(b.ms, prev_b)
                    prev_b = b.ms

    def test_02_distinct_multi_channel_regimes_present(self) -> None:
        fast = generate_trace(PROFILES["OP_FAST_CONT"], seed=101, session_id="f1")
        slow = generate_trace(PROFILES["OP_SLOW_CHUNK"], seed=102, session_id="s1")
        f_direct = sum(1 for b in fast.bursts if b.channel == "DIRECT_KEY") / len(fast.bursts)
        s_chunk = sum(1 for b in slow.bursts if b.channel == "BATCH_CHUNK") / len(slow.bursts)
        self.assertGreater(f_direct, 0.60)
        self.assertGreater(s_chunk, 0.35)
        self.assertGreater(slow.bursts[-1].ms, fast.bursts[-1].ms * 2.0)

    def test_03_backspace_and_replace_edits_present(self) -> None:
        tr = generate_trace(PROFILES["OP_SLOW_CHUNK"], seed=202, session_id="e1", n_bursts=120)
        kinds = {b.edit_kind for b in tr.bursts}
        self.assertIn("EMIT", kinds)
        self.assertIn("BACKSPACE", kinds)
        self.assertIn("REPLACE", kinds)

    def test_04_feature_vector_dimensionality_and_finiteness(self) -> None:
        tr = generate_trace(PROFILES["OP_HYBRID_GRID"], seed=303, session_id="v1")
        vec = extract_features(tr)
        self.assertEqual(len(vec), len(FEATURE_NAMES))
        self.assertEqual(len(vec), 19)
        for v in vec:
            self.assertEqual(v, v)  # not NaN

    def test_05_pre_holdout_threshold_strictly_independent_of_holdout(self) -> None:
        ds1 = build_split_datasets(seed=42323)
        m1 = fit_and_freeze_model("OP_FAST_CONT", ds1["OP_FAST_CONT"]["enroll"], ds1["OP_FAST_CONT"]["cal_pos"], ds1["OP_FAST_CONT"]["cal_imp"])
        # Mutating holdout lists cannot alter frozen model hash or tau
        ds1["OP_FAST_CONT"]["hold_gen"] = []
        ds1["OP_FAST_CONT"]["hold_mim"] = []
        m2 = fit_and_freeze_model("OP_FAST_CONT", ds1["OP_FAST_CONT"]["enroll"], ds1["OP_FAST_CONT"]["cal_pos"], ds1["OP_FAST_CONT"]["cal_imp"])
        self.assertEqual(m1.threshold_tau, m2.threshold_tau)
        self.assertEqual(m1.frozen_sha256, m2.frozen_sha256)

    def test_06_param_mimic_matches_first_order_speed_yet_is_rejected(self) -> None:
        ds = build_split_datasets(seed=42323)
        prof_id = "OP_FAST_CONT"
        m = fit_and_freeze_model(prof_id, ds[prof_id]["enroll"], ds[prof_id]["cal_pos"], ds[prof_id]["cal_imp"])
        gen_mean_iki = statistics.mean(
            (tr.bursts[-1].ms - tr.bursts[0].ms) / (len(tr.bursts) - 1) for tr in ds[prof_id]["hold_gen"]
        )
        mim_mean_iki = statistics.mean(
            (tr.bursts[-1].ms - tr.bursts[0].ms) / (len(tr.bursts) - 1) for tr in ds[prof_id]["hold_mim"]
        )
        # Verify mimic genuinely matches 1st-order mean IKI within 25%
        self.assertLess(abs(mim_mean_iki - gen_mean_iki) / gen_mean_iki, 0.25)
        # Yet classifier rejects 100% of parameter-matched mimics!
        for tr in ds[prof_id]["hold_mim"]:
            ok, _ = m.verify(tr)
            self.assertFalse(ok)

    def test_07_fatigued_self_is_slower_yet_accepted(self) -> None:
        ds = build_split_datasets(seed=42323)
        prof_id = "OP_SLOW_CHUNK"
        m = fit_and_freeze_model(prof_id, ds[prof_id]["enroll"], ds[prof_id]["cal_pos"], ds[prof_id]["cal_imp"])
        gen_dur = statistics.mean(tr.bursts[-1].ms for tr in ds[prof_id]["hold_gen"])
        fat_dur = statistics.mean(tr.bursts[-1].ms for tr in ds[prof_id]["hold_fat"])
        self.assertGreater(fat_dur, gen_dur * 1.15)
        for tr in ds[prof_id]["hold_fat"]:
            ok, dist = m.verify(tr)
            self.assertTrue(ok, f"fatigued_self falsely rejected: d={dist} > tau={m.threshold_tau}")

    def test_08_seed_42323_exact_zero_far_frr_and_swap_matrix(self) -> None:
        res = evaluate_seed(42323)
        ov = res["overall"]
        self.assertEqual(ov["frr"], "0/48")
        self.assertEqual(ov["far"], "0/48")
        self.assertEqual(ov["swap_diagonal_accept"], "48/48")
        self.assertEqual(ov["swap_offdiag_far"], "0/96")
        self.assertGreater(ov["min_holdout_margin"], 1.0)

    def test_09_deterministic_byte_identical_across_repeated_runs(self) -> None:
        r1 = evaluate_seed(42323)
        r2 = evaluate_seed(42323)
        r1.pop("wall_s")
        r2.pop("wall_s")
        self.assertEqual(r1, r2)

    def test_10_five_seed_multi_regime_zero_far_frr(self) -> None:
        for s in (42323, 42420, 42517, 42614, 42711):
            r = evaluate_seed(s)
            self.assertEqual(r["overall"]["frr"], "0/48", f"seed {s} frr")
            self.assertEqual(r["overall"]["far"], "0/48", f"seed {s} far")
            self.assertEqual(r["overall"]["swap_offdiag_far"], "0/96", f"seed {s} swap")


if __name__ == "__main__":
    unittest.main(verbosity=2)
