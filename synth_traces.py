#!/usr/bin/env python3
"""Deterministic Multi-Channel Operator Kinematics & Symbol-Burst Trace Synthesizer (DISPATCH-06).

Zero external dependencies (Python 3.12 stdlib only). Generates realistic normalized 2D
trajectory ticks (ms, x in [0,1], y in [0,1], trigger_kind) plus discrete symbol-emission
bursts across channels (DIRECT_KEY, VIRTUAL_GRID, BATCH_CHUNK) including BACKSPACE/REPLACE
edits across 3 distinct operator regimes, plus genuine, fatigued-self, random-stranger, and
parameter-matched mimic traces.
"""
from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass
from typing import Any

CHANNELS = ("DIRECT_KEY", "VIRTUAL_GRID", "BATCH_CHUNK")
TRIGGER_KINDS = ("MOVE", "DOWN", "UP", "DWELL", "DRAG")
EDIT_KINDS = ("EMIT", "BACKSPACE", "REPLACE")


@dataclass(frozen=True)
class TrajectoryTick:
    ms: float
    x: float
    y: float
    trigger_kind: str


@dataclass(frozen=True)
class SymbolBurst:
    ms: float
    channel: str
    symbol_count: int
    dwell_ms: float
    edit_kind: str


@dataclass(frozen=True)
class OperatorTrace:
    session_id: str
    claimed_profile: str
    true_source: str
    condition: str  # genuine | fatigued_self | stranger | param_mimic | swap_impostor
    ticks: tuple[TrajectoryTick, ...]
    bursts: tuple[SymbolBurst, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "claimed_profile": self.claimed_profile,
            "true_source": self.true_source,
            "condition": self.condition,
            "ticks": [asdict(t) for t in self.ticks],
            "bursts": [asdict(b) for b in self.bursts],
        }


@dataclass(frozen=True)
class OperatorProfile:
    profile_id: str
    regime_label: str
    # 2D trajectory biomechanics
    stroke_dur_ms: float       # typical point-to-point stroke duration
    curvature_bias: float      # signed lateral arc bow (+clockwise, -counter-clockwise)
    peak_skew: float           # normalized time of velocity peak in (0.25..0.75)
    submove_amp: float         # secondary corrective oscillation amplitude
    preclick_dwell_ms: float   # pre-trigger dwell time
    end_decel_ratio: float     # terminal deceleration damping (Fitts's law)
    # Multi-channel symbol burst rhythm
    channel_probs: tuple[float, float, float]  # (DIRECT_KEY, VIRTUAL_GRID, BATCH_CHUNK)
    chunk_mean_syms: float     # symbols per BATCH_CHUNK burst
    iki_median_ms: float       # median inter-event interval
    iki_log_sigma: float       # right-skewed lognormal spread
    iki_ar1: float             # lag-1 autocorrelation in log-IKI
    dwell_frac: float          # key/grid dwell relative to IKI
    backspace_prob: float
    replace_prob: float
    post_edit_mult: float      # hesitation multiplier right after BACKSPACE/REPLACE


PROFILES: dict[str, OperatorProfile] = {
    "OP_FAST_CONT": OperatorProfile(
        profile_id="OP_FAST_CONT",
        regime_label="fast_continuous",
        stroke_dur_ms=210.0,
        curvature_bias=+0.115,
        peak_skew=0.34,
        submove_amp=0.008,
        preclick_dwell_ms=42.0,
        end_decel_ratio=0.18,
        channel_probs=(0.82, 0.12, 0.06),
        chunk_mean_syms=2.2,
        iki_median_ms=98.0,
        iki_log_sigma=0.34,
        iki_ar1=0.58,
        dwell_frac=0.46,
        backspace_prob=0.055,
        replace_prob=0.015,
        post_edit_mult=1.55,
    ),
    "OP_SLOW_CHUNK": OperatorProfile(
        profile_id="OP_SLOW_CHUNK",
        regime_label="slow_trajectory_chunked_bursts",
        stroke_dur_ms=640.0,
        curvature_bias=-0.125,
        peak_skew=0.61,
        submove_amp=0.036,
        preclick_dwell_ms=178.0,
        end_decel_ratio=0.42,
        channel_probs=(0.14, 0.28, 0.58),
        chunk_mean_syms=5.4,
        iki_median_ms=345.0,
        iki_log_sigma=0.42,
        iki_ar1=0.64,
        dwell_frac=0.34,
        backspace_prob=0.075,
        replace_prob=0.145,
        post_edit_mult=2.25,
    ),
    "OP_HYBRID_GRID": OperatorProfile(
        profile_id="OP_HYBRID_GRID",
        regime_label="medium_virtual_grid_hybrid",
        stroke_dur_ms=385.0,
        curvature_bias=+0.018,
        peak_skew=0.47,
        submove_amp=0.019,
        preclick_dwell_ms=96.0,
        end_decel_ratio=0.28,
        channel_probs=(0.34, 0.54, 0.12),
        chunk_mean_syms=3.3,
        iki_median_ms=192.0,
        iki_log_sigma=0.30,
        iki_ar1=0.46,
        dwell_frac=0.62,
        backspace_prob=0.125,
        replace_prob=0.045,
        post_edit_mult=1.92,
    ),
}


def _clamp01(v: float) -> float:
    return 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)


def _skewed_progress(u: float, peak_skew: float, end_decel: float) -> float:
    """Monotonic minimum-jerk-like progress s(u) in [0,1] with tunable velocity peak skew."""
    # Warp u through beta-like power mapping so velocity peaks near peak_skew
    gamma = math.log(0.5) / math.log(max(0.15, min(0.85, peak_skew)))
    w = pow(max(0.0, min(1.0, u)), gamma)
    # 5th-order minimum-jerk polynomial + subtle terminal deceleration blend
    mj = w * w * w * (10.0 - 15.0 * w + 6.0 * w * w)
    smooth = (1.0 - end_decel * 0.35) * mj + (end_decel * 0.35) * (1.0 - (1.0 - w) ** 3)
    return max(0.0, min(1.0, smooth))


def _sample_channel(rng: random.Random, probs: tuple[float, float, float]) -> str:
    r = rng.random()
    if r < probs[0]:
        return "DIRECT_KEY"
    if r < probs[0] + probs[1]:
        return "VIRTUAL_GRID"
    return "BATCH_CHUNK"


def generate_trace(
    profile: OperatorProfile,
    *,
    seed: int,
    session_id: str,
    claimed_profile: str | None = None,
    condition: str = "genuine",
    n_strokes: int = 16,
    n_bursts: int = 64,
) -> OperatorTrace:
    """Synthesize one multi-channel trace for genuine, fatigued_self, stranger, or param_mimic."""
    rng = random.Random(seed)
    claimed = claimed_profile or profile.profile_id

    # Fatigue scales global tempo/IKI & mild jitter while preserving invariant biomechanics
    if condition == "fatigued_self":
        tempo_scale = rng.uniform(1.22, 1.34)
        noise_scale = 1.25
        bs_boost = 1.18
    else:
        tempo_scale = rng.uniform(0.95, 1.06)
        noise_scale = 1.0
        bs_boost = 1.0

    mimic = (condition == "param_mimic")

    # --- 1. Generate 2D normalized trajectory ticks ---
    ticks: list[TrajectoryTick] = []
    t_ms = 0.0
    cx = rng.uniform(0.18, 0.82)
    cy = rng.uniform(0.18, 0.82)

    for s_idx in range(n_strokes):
        # Target point at realistic distance [0.18..0.55]
        for _ in range(10):
            tx = rng.uniform(0.10, 0.90)
            ty = rng.uniform(0.10, 0.90)
            dist = math.hypot(tx - cx, ty - cy)
            if 0.18 <= dist <= 0.62:
                break
        dx, dy = tx - cx, ty - cy
        dist = max(1e-6, math.hypot(dx, dy))
        # Unit perpendicular vector for lateral arc curvature
        nx, ny = -dy / dist, dx / dist

        dur = profile.stroke_dur_ms * tempo_scale * rng.uniform(0.86, 1.16)
        steps = max(10, min(26, int(round(dur / 18.0))))
        dt = dur / steps

        if mimic:
            # Parameter-matched mimic matches mean stroke duration & distance,
            # but uses robotic linear/symmetric velocity, inverted/zero curvature, and no Fitts tail!
            curv = rng.gauss(0.0, 0.025)
            p_skew = 0.50 + rng.uniform(-0.03, 0.03)
            sub_amp = 0.002
            end_dec = 0.02
            dwell_ms = max(8.0, profile.preclick_dwell_ms * rng.uniform(0.85, 1.15))
        else:
            curv = profile.curvature_bias * rng.uniform(0.82, 1.18)
            p_skew = max(0.22, min(0.76, profile.peak_skew + rng.gauss(0.0, 0.02 * noise_scale)))
            sub_amp = profile.submove_amp * rng.uniform(0.80, 1.25) * noise_scale
            end_dec = profile.end_decel_ratio
            dwell_ms = max(12.0, profile.preclick_dwell_ms * tempo_scale * rng.uniform(0.84, 1.18))

        drag_stroke = (s_idx % 4 == 0)
        ticks.append(TrajectoryTick(round(t_ms, 2), round(_clamp01(cx), 5), round(_clamp01(cy), 5), "DOWN"))
        for k in range(1, steps + 1):
            u = k / steps
            if mimic:
                # Constant-speed / simple cosine ease lacks authentic skewed minimum-jerk
                prog = 0.5 - 0.5 * math.cos(math.pi * u)
                lat = curv * math.sin(math.pi * u) * dist
                jit_x = rng.gauss(0.0, 0.004)
                jit_y = rng.gauss(0.0, 0.004)
            else:
                prog = _skewed_progress(u, p_skew, end_dec)
                lat = (curv * math.sin(math.pi * u) + sub_amp * math.sin(3.0 * math.pi * u)) * dist
                jit_x = rng.gauss(0.0, 0.0014 * noise_scale)
                jit_y = rng.gauss(0.0, 0.0014 * noise_scale)
            px = _clamp01(cx + dx * prog + nx * lat + jit_x)
            py = _clamp01(cy + dy * prog + ny * lat + jit_y)
            t_ms += dt
            kind = "DRAG" if drag_stroke and k < steps else "MOVE"
            ticks.append(TrajectoryTick(round(t_ms, 2), round(px, 5), round(py, 5), kind))

        # Pre-click dwell tick before UP
        t_ms += dwell_ms
        ticks.append(TrajectoryTick(round(t_ms, 2), round(_clamp01(tx), 5), round(_clamp01(ty), 5), "DWELL"))
        t_ms += 6.0
        ticks.append(TrajectoryTick(round(t_ms, 2), round(_clamp01(tx), 5), round(_clamp01(ty), 5), "UP"))
        t_ms += rng.uniform(25.0, 65.0) * tempo_scale
        cx, cy = tx, ty

    # --- 2. Generate multi-channel symbol-emission bursts ---
    bursts: list[SymbolBurst] = []
    b_ms = 0.0
    prev_z = rng.gauss(0.0, 1.0)
    prev_was_edit = False

    # Target mean IKI for mimic to match 1st-order mean!
    expected_lognorm_mean = profile.iki_median_ms * math.exp(0.5 * profile.iki_log_sigma ** 2)

    for _ in range(n_bursts):
        ch = _sample_channel(rng, profile.channel_probs)
        r_edit = rng.random()
        p_bs = min(0.28, profile.backspace_prob * bs_boost)
        p_rep = min(0.28, profile.replace_prob * bs_boost)
        if r_edit < p_bs:
            edit_kind = "BACKSPACE"
        elif r_edit < p_bs + p_rep:
            edit_kind = "REPLACE"
        else:
            edit_kind = "EMIT"

        if ch == "BATCH_CHUNK":
            sym_cnt = max(2, int(round(rng.gauss(profile.chunk_mean_syms, 1.1))))
        else:
            sym_cnt = 1

        if mimic:
            # Mimic matches mean IKI & channel frequencies,
            # BUT generates memoryless symmetric Gaussian IKI (ar1 ~ 0, zero log-skew)
            # and forgets post-edit hesitation and operator dwell-to-flight coupling!
            iki = max(18.0, rng.gauss(expected_lognorm_mean, expected_lognorm_mean * 0.22))
            dwell = max(10.0, expected_lognorm_mean * profile.dwell_frac * rng.uniform(0.55, 1.45))
        else:
            ar1 = profile.iki_ar1
            innov = math.sqrt(max(1e-6, 1.0 - ar1 * ar1))
            z = ar1 * prev_z + innov * rng.gauss(0.0, 1.0)
            prev_z = z
            sig = profile.iki_log_sigma * (1.08 if condition == "fatigued_self" else 1.0)
            base_iki = profile.iki_median_ms * tempo_scale * math.exp(sig * z)
            if prev_was_edit:
                base_iki *= profile.post_edit_mult * rng.uniform(0.90, 1.12)
            iki = max(16.0, base_iki)
            dwell = max(10.0, profile.iki_median_ms * tempo_scale * profile.dwell_frac * rng.uniform(0.84, 1.18))

        b_ms += iki
        bursts.append(SymbolBurst(
            ms=round(b_ms, 2),
            channel=ch,
            symbol_count=sym_cnt,
            dwell_ms=round(dwell, 2),
            edit_kind=edit_kind,
        ))
        prev_was_edit = (edit_kind != "EMIT")

    return OperatorTrace(
        session_id=session_id,
        claimed_profile=claimed,
        true_source=profile.profile_id if not mimic else f"MIMIC({profile.profile_id})",
        condition=condition,
        ticks=tuple(ticks),
        bursts=tuple(bursts),
    )


def generate_random_stranger_profile(rng: random.Random, exclude_id: str) -> OperatorProfile:
    """Create a synthetic out-of-distribution stranger operator profile."""
    c1 = rng.uniform(0.1, 0.8)
    c2 = rng.uniform(0.1, 0.9 - c1)
    c3 = max(0.05, 1.0 - c1 - c2)
    s = c1 + c2 + c3
    return OperatorProfile(
        profile_id=f"STRANGER_{rng.randrange(10000, 99999)}",
        regime_label="random_stranger",
        stroke_dur_ms=rng.uniform(160.0, 760.0),
        curvature_bias=rng.choice([-1.0, 1.0]) * rng.uniform(0.04, 0.18),
        peak_skew=rng.uniform(0.26, 0.72),
        submove_amp=rng.uniform(0.005, 0.045),
        preclick_dwell_ms=rng.uniform(25.0, 230.0),
        end_decel_ratio=rng.uniform(0.10, 0.48),
        channel_probs=(c1 / s, c2 / s, c3 / s),
        chunk_mean_syms=rng.uniform(2.0, 6.5),
        iki_median_ms=rng.uniform(75.0, 460.0),
        iki_log_sigma=rng.uniform(0.22, 0.50),
        iki_ar1=rng.uniform(0.15, 0.75),
        dwell_frac=rng.uniform(0.22, 0.75),
        backspace_prob=rng.uniform(0.02, 0.18),
        replace_prob=rng.uniform(0.01, 0.18),
        post_edit_mult=rng.uniform(1.15, 2.55),
    )


def build_split_datasets(
    seed: int = 42323,
    *,
    n_enroll: int = 24,
    n_cal_pos: int = 8,
    n_cal_fat: int = 6,
    n_cal_imp: int = 12,
    n_hold_gen: int = 8,
    n_hold_fat: int = 8,
    n_hold_str: int = 8,
    n_hold_mim: int = 8,
) -> dict[str, dict[str, list[OperatorTrace]]]:
    """Build strictly disjoint enrollment, calibration, and holdout trace sets per profile.

    Crucially, holdout seeds are separated by a large stride from enroll/calib seeds so
    zero holdout traces are ever observed before decision thresholds are frozen.
    """
    out: dict[str, dict[str, list[OperatorTrace]]] = {}
    profile_list = list(PROFILES.values())

    for p_idx, prof in enumerate(profile_list):
        base = seed * 1009 + p_idx * 100_003
        enroll: list[OperatorTrace] = []
        for i in range(n_enroll):
            enroll.append(generate_trace(prof, seed=base + 10 + i, session_id=f"{prof.profile_id}_enr_{i}", condition="genuine"))

        cal_pos: list[OperatorTrace] = []
        for i in range(n_cal_pos):
            cal_pos.append(generate_trace(prof, seed=base + 1_000 + i, session_id=f"{prof.profile_id}_calg_{i}", condition="genuine"))
        for i in range(n_cal_fat):
            cal_pos.append(generate_trace(prof, seed=base + 2_000 + i, session_id=f"{prof.profile_id}_calf_{i}", condition="fatigued_self"))

        cal_imp: list[OperatorTrace] = []
        cal_rng = random.Random(base + 3_000)
        for i in range(n_cal_imp):
            if i % 2 == 0:
                sp = generate_random_stranger_profile(cal_rng, prof.profile_id)
                cal_imp.append(generate_trace(sp, seed=base + 3_100 + i, session_id=f"{prof.profile_id}_calstr_{i}", claimed_profile=prof.profile_id, condition="stranger"))
            else:
                cal_imp.append(generate_trace(prof, seed=base + 3_500 + i, session_id=f"{prof.profile_id}_calmim_{i}", claimed_profile=prof.profile_id, condition="param_mimic"))

        # Strictly disjoint holdout namespace (+50_000 stride)
        hold_gen: list[OperatorTrace] = []
        for i in range(n_hold_gen):
            hold_gen.append(generate_trace(prof, seed=base + 50_000 + i, session_id=f"{prof.profile_id}_hg_{i}", condition="genuine"))

        hold_fat: list[OperatorTrace] = []
        for i in range(n_hold_fat):
            hold_fat.append(generate_trace(prof, seed=base + 60_000 + i, session_id=f"{prof.profile_id}_hf_{i}", condition="fatigued_self"))

        hold_str: list[OperatorTrace] = []
        h_rng = random.Random(base + 70_000)
        others = [op for op in profile_list if op.profile_id != prof.profile_id]
        for i in range(n_hold_str):
            if i < len(others):
                sp = others[i]
            else:
                sp = generate_random_stranger_profile(h_rng, prof.profile_id)
            hold_str.append(generate_trace(sp, seed=base + 70_100 + i, session_id=f"{prof.profile_id}_hs_{i}", claimed_profile=prof.profile_id, condition="stranger"))

        hold_mim: list[OperatorTrace] = []
        for i in range(n_hold_mim):
            hold_mim.append(generate_trace(prof, seed=base + 80_000 + i, session_id=f"{prof.profile_id}_hm_{i}", claimed_profile=prof.profile_id, condition="param_mimic"))

        out[prof.profile_id] = {
            "enroll": enroll,
            "cal_pos": cal_pos,
            "cal_imp": cal_imp,
            "hold_gen": hold_gen,
            "hold_fat": hold_fat,
            "hold_str": hold_str,
            "hold_mim": hold_mim,
        }
    return out


if __name__ == "__main__":
    ds = build_split_datasets(seed=42323)
    for pid, splits in ds.items():
        print(pid, {k: len(v) for k, v in splits.items()})
