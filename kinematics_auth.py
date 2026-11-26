#!/usr/bin/env python3
"""Multi-Channel Event Kinematics & Rhythm Anomaly Classifier (DISPATCH-06).

Standalone Python 3.12 zero-dependency CLI & module verifying operator continuity from:
  - Generic normalized 2D trajectory ticks (ms, x in [0,1], y in [0,1], trigger_kind)
  - Discrete symbol-emission bursts labeled by channel (DIRECT_KEY, VIRTUAL_GRID,
    BATCH_CHUNK) including BACKSPACE / REPLACE edits.

Strict zero-leakage pipeline:
  1. Fit regularized Mahalanobis model (pure-Python Cholesky) strictly on enrollment +
     calibration splits.
  2. Freeze exact decision threshold tau strictly prior to holdout generation/evaluation
     and commit its SHA-256 digest.
  3. Evaluate disjoint holdout sessions across genuine, fatigued-self, random stranger,
     parameter-matched mimic, and full N x N cross-profile swap matrix.
"""
from __future__ import annotations

import hashlib
import json
import math
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any, Sequence

from synth_traces import (
    PROFILES,
    OperatorTrace,
    SymbolBurst,
    TrajectoryTick,
    build_split_datasets,
)

FEATURE_NAMES: tuple[str, ...] = (
    "curv_signed_mean",
    "path_efficiency",
    "peak_vel_t_ratio",
    "end_vel_ratio",
    "log_jerk_dimless",
    "submove_lat_rms",
    "dwell_stroke_ratio",
    "frac_direct_key",
    "frac_virtual_grid",
    "frac_batch_chunk",
    "mean_syms_per_burst",
    "frac_backspace",
    "frac_replace",
    "post_edit_hesitation",
    "iki_skewness",
    "iki_ar1",
    "dwell_iki_ratio",
    "dwell_rel_cv",
    "log_stroke_to_iki_ratio",
)


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _std(xs: Sequence[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def _skewness(xs: Sequence[float]) -> float:
    n = len(xs)
    if n < 3:
        return 0.0
    m = _mean(xs)
    s = math.sqrt(sum((x - m) ** 2 for x in xs) / n)
    if s < 1e-9:
        return 0.0
    return sum(((x - m) / s) ** 3 for x in xs) / n


def _ar1(xs: Sequence[float]) -> float:
    if len(xs) < 4:
        return 0.0
    x0 = xs[:-1]
    x1 = xs[1:]
    m0, m1 = _mean(x0), _mean(x1)
    num = sum((a - m0) * (b - m1) for a, b in zip(x0, x1))
    d0 = math.sqrt(sum((a - m0) ** 2 for a in x0))
    d1 = math.sqrt(sum((b - m1) ** 2 for b in x1))
    if d0 < 1e-9 or d1 < 1e-9:
        return 0.0
    return max(-1.0, min(1.0, num / (d0 * d1)))


def extract_features(trace: OperatorTrace) -> list[float]:
    """Extract 18-dimensional fatigue-invariant kinematic & multi-channel rhythm vector."""
    # --- 1. Segment 2D trajectory strokes (DOWN -> MOVE/DRAG* -> DWELL -> UP) ---
    strokes: list[list[TrajectoryTick]] = []
    cur: list[TrajectoryTick] = []
    for tk in trace.ticks:
        if tk.trigger_kind == "DOWN":
            if len(cur) >= 4:
                strokes.append(cur)
            cur = [tk]
        else:
            if cur:
                cur.append(tk)
            if tk.trigger_kind == "UP" and len(cur) >= 4:
                strokes.append(cur)
                cur = []
    if len(cur) >= 4:
        strokes.append(cur)

    curv_vals: list[float] = []
    eff_vals: list[float] = []
    peak_t_vals: list[float] = []
    end_vel_vals: list[float] = []
    jerk_vals: list[float] = []
    sub_rms_vals: list[float] = []
    dwell_ratios: list[float] = []
    stroke_durs: list[float] = []

    for st in strokes:
        # Separate active motion ticks from terminal DWELL/UP ticks
        motion: list[TrajectoryTick] = []
        dwell_ms = 0.0
        for i, tk in enumerate(st):
            if tk.trigger_kind == "DWELL":
                if i > 0:
                    dwell_ms += max(0.0, tk.ms - st[i - 1].ms)
            elif tk.trigger_kind == "UP":
                continue
            else:
                motion.append(tk)
        if len(motion) < 5:
            continue

        p0, p1 = motion[0], motion[-1]
        dx, dy = p1.x - p0.x, p1.y - p0.y
        chord = math.hypot(dx, dy)
        stroke_dur = max(1.0, p1.ms - p0.ms)
        if chord < 1e-4:
            continue
        nx, ny = -dy / chord, dx / chord

        # Signed lateral deviations, arc length, speeds, accelerations, jerks
        arc = 0.0
        speeds: list[tuple[float, float]] = []  # (norm_t, speed)
        lats: list[tuple[float, float]] = []    # (norm_t, signed_lat / chord)
        for i in range(1, len(motion)):
            a, b = motion[i - 1], motion[i]
            dt = max(0.5, b.ms - a.ms)
            seg = math.hypot(b.x - a.x, b.y - a.y)
            arc += seg
            u = (b.ms - p0.ms) / stroke_dur
            speeds.append((u, seg / dt))
            lat = ((b.x - p0.x) * nx + (b.y - p0.y) * ny) / chord
            lats.append((u, lat))

        eff_vals.append(min(1.0, chord / max(chord, arc)))

        # Fit primary arch c * sin(pi * u) and measure signed bow & 3rd-harmonic residual
        sin_vals = [math.sin(math.pi * u) for u, _ in lats]
        denom = sum(s * s for s in sin_vals)
        c_arch = sum(s * lv for s, (_, lv) in zip(sin_vals, lats)) / max(1e-9, denom)
        curv_vals.append(c_arch)
        resid_sq = [(lv - c_arch * s) ** 2 for s, (_, lv) in zip(sin_vals, lats)]
        sub_rms_vals.append(math.sqrt(_mean(resid_sq)))

        # Velocity profile shape: peak velocity timing & terminal Fitts deceleration
        v_max = max(v for _, v in speeds)
        if v_max > 1e-9:
            # Weighted peak time around top 15% velocities for low-noise peak estimate
            top = [(u, v) for u, v in speeds if v >= 0.85 * v_max]
            t_pk = sum(u * v for u, v in top) / sum(v for _, v in top)
            peak_t_vals.append(t_pk)
            tail = [v for u, v in speeds if u >= 0.78]
            end_vel_vals.append((_mean(tail) / v_max) if tail else 0.0)

        # Dimensionless log-jerk (invariant to overall duration and path scale)
        vel_only = [v for _, v in speeds]
        dt_avg = stroke_dur / max(1, len(vel_only))
        accs = [(vel_only[i] - vel_only[i - 1]) / dt_avg for i in range(1, len(vel_only))]
        jerks = [abs(accs[i] - accs[i - 1]) / dt_avg for i in range(1, len(accs))]
        if jerks:
            dimless_j = (_mean(jerks) * (stroke_dur ** 3)) / max(1e-5, chord)
            jerk_vals.append(math.log1p(max(0.0, dimless_j)))

        dwell_ratios.append(dwell_ms / max(1.0, dwell_ms + stroke_dur))
        stroke_durs.append(stroke_dur)

    # --- 2. Multi-channel symbol burst & edit rhythm features ---
    bursts = trace.bursts
    nb = max(1, len(bursts))
    f_direct = sum(1 for b in bursts if b.channel == "DIRECT_KEY") / nb
    f_vgrid = sum(1 for b in bursts if b.channel == "VIRTUAL_GRID") / nb
    f_chunk = sum(1 for b in bursts if b.channel == "BATCH_CHUNK") / nb
    mean_syms = _mean([float(b.symbol_count) for b in bursts])
    f_bs = sum(1 for b in bursts if b.edit_kind == "BACKSPACE") / nb
    f_rep = sum(1 for b in bursts if b.edit_kind == "REPLACE") / nb

    ikis: list[float] = []
    emit_ikis: list[float] = []
    post_edit_ikis: list[float] = []
    dwell_ratios_b: list[float] = []
    for i in range(1, len(bursts)):
        iki = max(5.0, bursts[i].ms - bursts[i - 1].ms)
        ikis.append(iki)
        if bursts[i - 1].edit_kind == "EMIT":
            emit_ikis.append(iki)
        else:
            post_edit_ikis.append(iki)
        dwell_ratios_b.append(bursts[i].dwell_ms / iki)

    med_emit_iki = statistics.median(emit_ikis) if emit_ikis else (statistics.median(ikis) if ikis else 100.0)
    post_edit_hes = (
        (_mean(post_edit_ikis) / max(1.0, med_emit_iki))
        if post_edit_ikis else 1.0
    )
    iki_skew = _skewness(ikis)
    log_ikis = [math.log(x) for x in ikis]
    iki_ar1 = _ar1(log_ikis)
    dwell_iki_ratio = statistics.median(dwell_ratios_b) if dwell_ratios_b else 0.4
    dwell_vals = [b.dwell_ms for b in bursts]
    dwell_rel_cv = _std(dwell_vals) / max(1.0, _mean(dwell_vals))

    return [
        _mean(curv_vals),
        _mean(eff_vals),
        _mean(peak_t_vals),
        _mean(end_vel_vals),
        _mean(jerk_vals),
        _mean(sub_rms_vals),
        statistics.median(dwell_ratios) if dwell_ratios else 0.0,
        f_direct,
        f_vgrid,
        f_chunk,
        mean_syms,
        f_bs,
        f_rep,
        post_edit_hes,
        iki_skew,
        iki_ar1,
        dwell_iki_ratio,
        dwell_rel_cv,
        math.log(max(1.0, statistics.median(stroke_durs) if stroke_durs else 300.0) / max(1.0, med_emit_iki)),
    ]


def _cholesky(mat: list[list[float]]) -> list[list[float]]:
    n = len(mat)
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = sum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                val = mat[i][i] - s
                L[i][j] = math.sqrt(max(1e-9, val))
            else:
                L[i][j] = (mat[i][j] - s) / max(1e-9, L[j][j])
    return L


def _solve_lower(L: list[list[float]], b: list[float]) -> list[float]:
    n = len(b)
    y = [0.0] * n
    for i in range(n):
        s = sum(L[i][k] * y[k] for k in range(i))
        y[i] = (b[i] - s) / max(1e-9, L[i][i])
    return y


@dataclass(frozen=True)
class FrozenOperatorModel:
    profile_id: str
    feature_names: tuple[str, ...]
    mean: tuple[float, ...]
    scale: tuple[float, ...]
    chol_lower: tuple[tuple[float, ...], ...]
    threshold_tau: float
    cal_pos_max_dist: float
    cal_imp_min_dist: float
    cal_margin: float
    frozen_sha256: str

    def distance(self, trace: OperatorTrace) -> float:
        x = extract_features(trace)
        z = [(xi - mi) / si for xi, mi, si in zip(x, self.mean, self.scale)]
        L = [list(row) for row in self.chol_lower]
        y = _solve_lower(L, z)
        return math.sqrt(sum(v * v for v in y) / len(y))

    def verify(self, trace: OperatorTrace) -> tuple[bool, float]:
        d = self.distance(trace)
        return (d <= self.threshold_tau), round(d, 6)


def fit_and_freeze_model(
    profile_id: str,
    enroll_traces: Sequence[OperatorTrace],
    cal_pos_traces: Sequence[OperatorTrace],
    cal_imp_traces: Sequence[OperatorTrace],
    *,
    shrinkage: float = 0.55,
) -> FrozenOperatorModel:
    """Fit regularized Mahalanobis model and freeze threshold strictly prior to holdout."""
    # True 3-way split: fit mean & shrinkage covariance strictly on enroll_traces ONLY,
    # keeping cal_pos_traces strictly out-of-sample for unbiased threshold calibration!
    fit_vecs = [extract_features(tr) for tr in enroll_traces]
    n = len(fit_vecs)
    d = len(FEATURE_NAMES)
    assert n >= 4, "Need >=4 fitting traces"

    mu = [sum(v[j] for v in fit_vecs) / n for j in range(d)]
    # Standardize each feature dimension with safe floor to avoid zero-variance collapse
    scale = [
        max(0.015, math.sqrt(sum((v[j] - mu[j]) ** 2 for v in fit_vecs) / max(1, n - 1)))
        for j in range(d)
    ]
    Z = [[(v[j] - mu[j]) / scale[j] for j in range(d)] for v in fit_vecs]

    # Shrinkage covariance: (1 - lambda) * SampleCov + lambda * I
    cov = [[0.0] * d for _ in range(d)]
    for i in range(d):
        for j in range(i, d):
            c = sum(z[i] * z[j] for z in Z) / max(1, n - 1)
            val = (1.0 - shrinkage) * c + (shrinkage if i == j else 0.0)
            if i == j:
                val = max(0.05, val)
            cov[i][j] = val
            cov[j][i] = val
    L = _cholesky(cov)

    def raw_dist(tr: OperatorTrace) -> float:
        x = extract_features(tr)
        z = [(xi - mi) / si for xi, mi, si in zip(x, mu, scale)]
        y = _solve_lower(L, z)
        return math.sqrt(sum(v * v for v in y) / d)

    pos_dists = sorted(raw_dist(tr) for tr in cal_pos_traces)
    imp_dists = sorted(raw_dist(tr) for tr in cal_imp_traces)
    pos_max = pos_dists[-1]
    imp_min = imp_dists[0]

    # Calibrate fixed decision threshold tau strictly on calibration data:
    # geometric/weighted split favoring low FAR while leaving headroom above worst fatigued-self.
    pos_mean = _mean(pos_dists)
    pos_std = max(0.12, _std(pos_dists))
    # Principled one-class Mahalanobis upper envelope anchored strictly on calibration data
    chi_cap = max(pos_max + 0.48, pos_mean + 4.5 * pos_std)
    if imp_min > pos_max:
        tau = min(pos_max + 0.32 * (imp_min - pos_max), chi_cap)
    else:
        # Minimize calibration FAR + FRR across candidate midpoints
        cands = sorted(set(pos_dists + imp_dists))
        best_tau = 0.5 * (pos_max + imp_min)
        best_err = 999.0
        for c in cands:
            frr = sum(1 for x in pos_dists if x > c) / len(pos_dists)
            far = sum(1 for x in imp_dists if x <= c) / len(imp_dists)
            err = far + frr
            if err < best_err:
                best_err = err
                best_tau = c
        tau = best_tau

    tau = round(tau, 6)
    commit_payload = json.dumps({
        "profile_id": profile_id,
        "mean": [round(x, 8) for x in mu],
        "scale": [round(x, 8) for x in scale],
        "threshold_tau": tau,
    }, sort_keys=True).encode("utf-8")
    digest = hashlib.sha256(commit_payload).hexdigest()

    return FrozenOperatorModel(
        profile_id=profile_id,
        feature_names=FEATURE_NAMES,
        mean=tuple(mu),
        scale=tuple(scale),
        chol_lower=tuple(tuple(row) for row in L),
        threshold_tau=tau,
        cal_pos_max_dist=round(pos_max, 6),
        cal_imp_min_dist=round(imp_min, 6),
        cal_margin=round(imp_min - pos_max, 6),
        frozen_sha256=digest,
    )


def evaluate_seed(seed: int = 42323) -> dict[str, Any]:
    """Run full zero-leakage enrollment -> threshold freeze -> holdout & swap matrix audit."""
    t0 = time.perf_counter()
    ds = build_split_datasets(seed=seed)
    profile_ids = list(PROFILES.keys())

    # Step 1: Fit and freeze all models strictly BEFORE touching any holdout split
    models: dict[str, FrozenOperatorModel] = {}
    for pid in profile_ids:
        models[pid] = fit_and_freeze_model(
            pid,
            ds[pid]["enroll"],
            ds[pid]["cal_pos"],
            ds[pid]["cal_imp"],
        )

    # Step 2: Evaluate strictly held-out sessions per profile
    per_profile: dict[str, Any] = {}
    tot_gen_n = tot_gen_rej = 0
    tot_fat_n = tot_fat_rej = 0
    tot_str_n = tot_str_acc = 0
    tot_mim_n = tot_mim_acc = 0
    min_holdout_margin = 999.0

    for pid in profile_ids:
        m = models[pid]
        gen_res = [m.verify(tr) for tr in ds[pid]["hold_gen"]]
        fat_res = [m.verify(tr) for tr in ds[pid]["hold_fat"]]
        str_res = [m.verify(tr) for tr in ds[pid]["hold_str"]]
        mim_res = [m.verify(tr) for tr in ds[pid]["hold_mim"]]

        g_rej = sum(1 for ok, _ in gen_res if not ok)
        f_rej = sum(1 for ok, _ in fat_res if not ok)
        s_acc = sum(1 for ok, _ in str_res if ok)
        m_acc = sum(1 for ok, _ in mim_res if ok)

        pos_max = max(d for _, d in gen_res + fat_res)
        neg_min = min(d for _, d in str_res + mim_res)
        margin = round(neg_min - pos_max, 6)
        min_holdout_margin = min(min_holdout_margin, margin)

        tot_gen_n += len(gen_res); tot_gen_rej += g_rej
        tot_fat_n += len(fat_res); tot_fat_rej += f_rej
        tot_str_n += len(str_res); tot_str_acc += s_acc
        tot_mim_n += len(mim_res); tot_mim_acc += m_acc

        per_profile[pid] = {
            "regime": PROFILES[pid].regime_label,
            "threshold_tau": m.threshold_tau,
            "frozen_sha256": m.frozen_sha256[:16],
            "genuine_frr": f"{g_rej}/{len(gen_res)}",
            "fatigued_frr": f"{f_rej}/{len(fat_res)}",
            "stranger_far": f"{s_acc}/{len(str_res)}",
            "param_mimic_far": f"{m_acc}/{len(mim_res)}",
            "holdout_pos_max_d": round(pos_max, 4),
            "holdout_neg_min_d": round(neg_min, 4),
            "holdout_margin": margin,
        }

    # Step 3: Full N x N cross-profile swap matrix (model[claimed] evaluated on hold_gen + hold_fat of true_src)
    swap_matrix: dict[str, dict[str, dict[str, Any]]] = {}
    diag_ok = diag_tot = off_acc = off_tot = 0
    for claimed in profile_ids:
        swap_matrix[claimed] = {}
        m = models[claimed]
        for true_src in profile_ids:
            traces = ds[true_src]["hold_gen"] + ds[true_src]["hold_fat"]
            acc = sum(1 for tr in traces if m.verify(tr)[0])
            tot = len(traces)
            swap_matrix[claimed][true_src] = {
                "accepted": acc,
                "total": tot,
                "accept_rate": round(acc / tot, 4),
            }
            if claimed == true_src:
                diag_ok += acc
                diag_tot += tot
            else:
                off_acc += acc
                off_tot += tot

    tot_pos = tot_gen_n + tot_fat_n
    tot_pos_rej = tot_gen_rej + tot_fat_rej
    tot_neg = tot_str_n + tot_mim_n
    tot_neg_acc = tot_str_acc + tot_mim_acc
    wall_s = round(time.perf_counter() - t0, 4)

    return {
        "seed": seed,
        "wall_s": wall_s,
        "profiles": len(profile_ids),
        "total_holdout_sessions": tot_pos + tot_neg,
        "overall": {
            "frr": f"{tot_pos_rej}/{tot_pos}",
            "frr_rate": round(tot_pos_rej / tot_pos, 6),
            "genuine_frr": f"{tot_gen_rej}/{tot_gen_n}",
            "fatigued_self_frr": f"{tot_fat_rej}/{tot_fat_n}",
            "far": f"{tot_neg_acc}/{tot_neg}",
            "far_rate": round(tot_neg_acc / tot_neg, 6),
            "stranger_far": f"{tot_str_acc}/{tot_str_n}",
            "param_mimic_far": f"{tot_mim_acc}/{tot_mim_n}",
            "swap_diagonal_accept": f"{diag_ok}/{diag_tot}",
            "swap_offdiag_far": f"{off_acc}/{off_tot}",
            "min_holdout_margin": round(min_holdout_margin, 6),
        },
        "per_profile": per_profile,
        "swap_matrix": swap_matrix,
    }


def format_report_table(res: dict[str, Any]) -> str:
    lines = [
        f"=== DISPATCH-06 MULTI-CHANNEL KINEMATICS AUTH REPORT (seed={res['seed']}, wall={res['wall_s']}s) ===",
        f"Overall Holdout ({res['total_holdout_sessions']} sessions, thresholds frozen pre-holdout):",
        f"  FRR = {res['overall']['frr']} ({res['overall']['frr_rate']*100:.2f}%) "
        f"[genuine={res['overall']['genuine_frr']}, fatigued_self={res['overall']['fatigued_self_frr']}]",
        f"  FAR = {res['overall']['far']} ({res['overall']['far_rate']*100:.2f}%) "
        f"[stranger={res['overall']['stranger_far']}, param_mimic={res['overall']['param_mimic_far']}]",
        f"  Cross-Profile Swap Matrix: diagonal={res['overall']['swap_diagonal_accept']}, "
        f"off-diagonal FAR={res['overall']['swap_offdiag_far']}, min_margin={res['overall']['min_holdout_margin']:+.4f}",
        "",
        f"{'Profile':<16} {'tau':>7} {'GenFRR':>8} {'FatFRR':>8} {'StrFAR':>8} {'MimFAR':>8} {'Margin':>8} {'FrozenSHA':>16}",
        "-" * 87,
    ]
    for pid, r in res["per_profile"].items():
        lines.append(
            f"{pid:<16} {r['threshold_tau']:>7.3f} {r['genuine_frr']:>8} {r['fatigued_frr']:>8} "
            f"{r['stranger_far']:>8} {r['param_mimic_far']:>8} {r['holdout_margin']:>+8.3f} {r['frozen_sha256']:>16}"
        )
    lines.append("")
    lines.append("Cross-Profile Swap Matrix (rows=claimed model, cols=true source [gen+fatigued]):")
    pids = list(res["swap_matrix"].keys())
    lines.append(f"{'Claimed \\\\ True':<16}" + "".join(f"{p:>16}" for p in pids))
    for c in pids:
        row_s = f"{c:<16}"
        for t in pids:
            cell = res["swap_matrix"][c][t]
            row_s += f"{cell['accepted']}/{cell['total']:<13}".rjust(16)
        lines.append(row_s)
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    seed = 42323
    if "--seed" in args:
        seed = int(args[args.index("--seed") + 1])

    if "--sweep" in args:
        seeds = [42323 + i * 97 for i in range(10)]
        out = []
        all_perfect = True
        for s in seeds:
            r = evaluate_seed(s)
            ov = r["overall"]
            out.append({
                "seed": s,
                "frr": ov["frr"],
                "far": ov["far"],
                "swap_offdiag_far": ov["swap_offdiag_far"],
                "min_holdout_margin": ov["min_holdout_margin"],
            })
            if ov["frr_rate"] > 0.0 or ov["far_rate"] > 0.0:
                all_perfect = False
        print(json.dumps({"sweep_seeds": len(seeds), "all_zero_far_frr": all_perfect, "results": out}, indent=2))
        return 0 if all_perfect else 1

    res = evaluate_seed(seed)
    if "--json" in args:
        print(json.dumps(res, indent=2))
    else:
        print(format_report_table(res))
    return 0 if (res["overall"]["frr_rate"] <= 0.05 and res["overall"]["far_rate"] <= 0.05) else 1


if __name__ == "__main__":
    raise SystemExit(main())
