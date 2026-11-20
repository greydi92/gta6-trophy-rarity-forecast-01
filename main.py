"""gta6-trophy-rarity-forecast-01: forecast every GTA VI trophy's Exophase earn
rate at the case's target, as five quantiles per trophy.

Python 3.13, standard library only, offline, deterministic.

Model (per trophy, on the logit scale)
--------------------------------------
The case gives the house's reads of Exophase's trophy page over time: each
read is every trophy's earn rate (percent of tracked players who earned it)
plus the players-tracked count. Two things move a rate:

  * players keep earning trophies (rates rise, fast early then saturating),
  * Exophase's tracked-player count grows, and each refresh dilutes the
    rates, because the players who join the tracker late have had less time
    to earn anything (launch night: the page sat frozen at 638 players for
    hours, then one refresh took it to 3,868 and every rate fell).

So the program fits, per trophy, a saturating growth curve on the logit
scale -- y(t) = A - B * exp(-k * (t - t1)) -- model-averaged over k instead
of picking one (when the history has not bent yet, several asymptotes fit
it equally well, and picking one pretends the curve is identified), plus
a log-time linear trend as a second candidate, plus a decomposition
candidate that fits saturating earners against a power-law players count
(the only candidate that can turn a rate down after it rose). The median
is a bracketing blend of the candidates, not a winner-pick: on a 4.5-day
horizon from 2.5 days of history the saturating fit sits below the truth
and the log-time line above it, so their average is the robust centre.
The decomposition enters the blend only for dilution-dominated trophies
-- a rate flat or falling while the player count still grows fast --
where its structure is the right one; for a normal trophy its
saturating-earners assumption misfires. History is first collapsed to
distinct snapshots (consecutive identical reads are one Exophase refresh,
not independent observations), so a long frozen stretch cannot overweight
one value.

When the history is flat -- one distinct snapshot, as in the practice cases,
where Exophase had not refreshed once -- no curve can be fitted, so the
program falls back to a dilution prior instead of echoing the stale page:
players tracked are assumed to grow roughly like (1 + hours since release),
anchored at the first read, while earners grow only slowly (a few percent
an hour, faster for story trophies than for grinds). That prior is
calibrated on the single refresh the house documents: 638 -> 3,868 players
at 04:30 UTC on launch night, with every rate lower, which the house's own
published practice numbers (persistence baseline loss ~1.51, typical miss
~0.68 points) pin at refreshed rates of about the launch rates / 5.

Spread (the quantiles)
----------------------
q_tau = inv_logit(median_logit + z_tau * sigma), with sigma built in
quadrature from: the fit's residual spread, extrapolation uncertainty that
grows like sqrt(horizon), disagreement between the two candidate fits,
staleness of the last snapshot (an unseen refresh may already be due),
and binomial noise in the rate itself (1 / sqrt(earners)). A trophy at
100% is saturated and gets a median of 99.99 with a wide logit spread,
because on the logit scale the difference between 100% and 95% is enormous.

The program reads every <id>.in in the input directory and writes
<id>.out to the output directory:
    {"trophies": [{"id": 1, "q": [q05, q25, q50, q75, q95]}, ...]}
one row per trophy of the case, five finite numbers from 0 to 100,
non-decreasing, rounded to 4 decimals (half up, as JavaScript rounds).
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a module
# shipped beside this file would import. (This entry is a single file.)
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]

# --- numeric helpers -------------------------------------------------------

def clamp_pct(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(lg):
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x):
    """x rounded to 4 decimals, half up, as JavaScript's
    Math.round(x * 10000) / 10000 rounds it."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value.is_integer() else value


def parse_time(s):
    """ISO-8601 with a Z suffix (or a plain YYYY-MM-DD date)."""
    if s is None:
        return None
    try:
        return datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except ValueError:
        return None


# --- trophy classification -------------------------------------------------
# Only used for priors (asymptote centre, earners-growth rate). Story
# trophies are earned by playing the campaign and rise fast; grinds are
# count-based and rise slowly; the rest sit in between.

def classify(trophy):
    d = (trophy.get("description") or "").lower()
    if "chapter" in d or "final mission" in d:
        return "story"
    if any(ch.isdigit() for ch in d):
        return "grind"
    if "explore" in d or "every spot" in d or "each type" in d or "try each" in d:
        return "explore"
    return "other"


# Per class: (asymptote growth multiple over the last rate, earners growth
# per hour in the flat-history dilution prior). The multiple is deliberately
# modest -- it only centres the ridge that stabilises the saturating fit
# when the data has not bent yet. (An earlier draft used x8 capped at 90%;
# on simulated launch weeks that dragged every asymptote to ~90% and the
# fit picked degenerate slow-k curves. The ridge must not outvote the data.)
CLASS_PRIOR = {
    "story": (1.6, 0.07),
    "grind": (1.3, 0.02),
    "explore": (1.4, 0.05),
    "other": (1.4, 0.05),
}

# Earners-growth multiple (ridge centre for the decomposition model's
# saturating earners curve, in units of the last snapshot's earners):
# story trophies keep earning fast as the player base plays through the
# campaign; grinds barely move.
EARNERS_MULT = {"story": 3.0, "grind": 1.4, "explore": 2.0, "other": 2.0}

# Saturating-curve grid: k in 1/hour, log-spaced, timescales ~0.5 h .. 500 h.
K_GRID = [math.exp(math.log(0.002) + i * (math.log(2.0) - math.log(0.002)) / 47.0)
          for i in range(48)]


# --- history -> distinct snapshots ----------------------------------------

def snapshots(case):
    """Collapse consecutive identical reads to one snapshot each: a read
    that repeats the page is the same Exophase refresh, not new data.
    Returns [(slot_dt, players_tracked, rates), ...], oldest first."""
    out = []
    prev_key = None
    for read in case.get("history") or []:
        rates = read.get("rates") or []
        key = tuple(-1.0 if r is None else r for r in rates)
        if key == prev_key:
            continue
        prev_key = key
        out.append((parse_time(read.get("slot")), read.get("players_tracked"), rates))
    return out


# --- curve fits (logit scale, times in hours since release) ---------------

def _sat_exp_ols(ts, ys, a_prior, k, lam=0.5):
    """Ridge least squares for y = A - B * exp(-k * (t - t1)) at one k,
    asymptote A shrunk toward a_prior. Returns (A, B, sse, score) where
    score = sse + lam * (A - a_prior) ** 2."""
    t1 = ts[0]
    n = len(ts)
    v = [-math.exp(-k * (t - t1)) for t in ts]  # y = A + B * v
    sv = sum(v)
    svv = sum(x * x for x in v)
    sy = sum(ys)
    svy = sum(a * b for a, b in zip(v, ys))
    det = (n + lam) * svv - sv * sv
    if abs(det) < 1e-12:
        return None
    a = ((sy + lam * a_prior) * svv - sv * svy) / det
    b = ((n + lam) * svy - sv * (sy + lam * a_prior)) / det
    sse = sum((y - (a + b * vv)) ** 2 for y, vv in zip(ys, v))
    return a, b, sse, sse + lam * (a - a_prior) ** 2


def sat_exp_average(ts, ys, a_prior, t_target, lam=0.5, s2_floor=0.02):
    """Model-average the saturating fit over the k grid instead of picking
    one k: when the history has not bent yet, several (asymptote, rate)
    pairs fit it almost equally well, and picking one pretends the curve
    is identified. Weights are exp(-(score - best) / (2 * s2)) with s2
    the residual variance, so a sharply identified curve concentrates and
    an unidentified one spreads. Returns (median_logit, sigma_model,
    best_sse) where sigma_model is the weighted spread of the forecasts
    across k -- real model uncertainty, carried into the quantiles."""
    t1 = ts[0]
    n = len(ts)
    cands = []
    best_sse = None
    for k in K_GRID:
        got = _sat_exp_ols(ts, ys, a_prior, k, lam)
        if got is None:
            continue
        a, b, sse, score = got
        med = a - b * math.exp(-k * (t_target - t1))
        cands.append((score, sse, med))
        if best_sse is None or sse < best_sse:
            best_sse = sse
    s2 = max(best_sse / max(1, n - 2), s2_floor)
    best_score = min(c[0] for c in cands)
    tot = 0.0
    med = 0.0
    for score, _sse, m in cands:
        w = math.exp(-(score - best_score) / (2.0 * s2))
        tot += w
        med += w * m
    med /= tot
    var = 0.0
    for score, _sse, m in cands:
        w = math.exp(-(score - best_score) / (2.0 * s2))
        var += w * (m - med) ** 2
    sigma_model = math.sqrt(var / tot)
    return med, sigma_model, best_sse




def fit_loglin(ts, ys):
    """Least squares for y = a + b * ln(1 + t), t hours since release."""
    xs = [math.log1p(max(0.0, t)) for t in ts]
    n = len(xs)
    sx = sum(xs)
    sxx = sum(x * x for x in xs)
    sy = sum(ys)
    sxy = sum(a * b for a, b in zip(xs, ys))
    det = n * sxx - sx * sx
    if abs(det) < 1e-12:
        return sy / n, 0.0, sum((y - sy / n) ** 2 for y in ys)
    b = (n * sxy - sx * sy) / det
    a = (sy - b * sx) / n
    sse = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    return a, b, sse



def predict_loglin(a, b, t):
    return a + b * math.log1p(max(0.0, t))



# --- decomposition model: earners and players separately -------------------
# A rate is earners / players_tracked. Fitting the rate directly can only
# draw monotone curves, but a real rate can rise and then FALL: earners
# saturate while the tracked-player count keeps growing (that is exactly
# what launch night's refresh did -- every rate fell when players went
# 638 -> 3,868). So the third candidate fits the two pieces apart:
#   * players P(t): a power law in (1 + t), fitted on the recent window
#     (player growth decelerates, so the recent slope is the honest one);
#   * earners E(t): a saturating curve E_inf - C * exp(-k * t), fitted
#     to the implied earners r * P / 100, normalised by the last value.
# The forecast rate is E(target) / P(target). Cumulative earners can only
# grow, so this candidate is well-posed where the rate fit is not.

def fit_players_power(ts, ps):
    """OLS for ln P = a + b * ln(1 + t). Returns (a, b) with b in [0, 3]."""
    xs = [math.log1p(max(0.0, t)) for t in ts]
    ys = [math.log(max(1.0, p)) for p in ps]
    n = len(xs)
    sx = sum(xs)
    sxx = sum(x * x for x in xs)
    sy = sum(ys)
    sxy = sum(a * b for a, b in zip(xs, ys))
    det = n * sxx - sx * sx
    if abs(det) < 1e-12 or n < 2:
        return math.log(max(1.0, ps[-1])), 0.0
    b = (n * sxy - sx * sy) / det
    a = (sy - b * sx) / n
    return a, min(3.0, max(0.0, b))


def players_at(a, b, t):
    return math.exp(a + b * math.log1p(max(0.0, t)))


def decomp_forecast(series, t_target, cls):
    """series: [(t_hours, rate_pct, players)] per snapshot, oldest first.
    Returns (median_logit, sigma_model) or None if it cannot be fitted."""
    es = []
    e_ref = None
    for t, r, p in series:
        e = max(0.0, r) / 100.0 * max(1.0, p)
        es.append(e)
    if not es or es[-1] < 0.5:
        return None
    e_ref = es[-1]
    ts = [t for t, _, _ in series]
    ps = [p for _, _, p in series]
    # Players: fit the power law on the recent 60% of the time span.
    t_cut = ts[0] + 0.4 * (ts[-1] - ts[0])
    recent = [(t, p) for t, p in zip(ts, ps) if t >= t_cut]
    if len(recent) >= 3:
        a_p, b_p = fit_players_power([t for t, _ in recent], [p for _, p in recent])
    else:
        a_p, b_p = fit_players_power(ts, ps)
    p_target = players_at(a_p, b_p, t_target)
    if p_target <= 0:
        return None
    # Earners: saturating curve on the normalised series.
    ys = [e / e_ref for e in es]
    a_prior = EARNERS_MULT[cls]
    med_e, sigma_e, _sse = sat_exp_average(ts, ys, a_prior, t_target,
                                           lam=0.5, s2_floor=0.002)
    if med_e <= 0:
        return None
    e_target = e_ref * med_e
    rate = 100.0 * e_target / p_target
    # The earners spread, in logit terms near the forecast rate, is
    # roughly sigma_e / med_e (relative error); convert it.
    sigma_model = min(0.6, sigma_e / max(0.05, med_e))
    return logit(rate), sigma_model



# --- the dilution prior (flat history) -------------------------------------

def dilution_prior(last_rate, players, t_first, t_target, cls):
    """Median logit + sigma when the page has never refreshed in the case:
    players tracked grow ~ (1 + t) since release, earners only slowly, so
    the rate at the target is the last rate scaled by (earners growth /
    players growth). Calibrated on launch night's one observed refresh
    (638 -> 3,868 players, rates ~ /5)."""
    growth_mult, gamma = CLASS_PRIOR[cls]
    p_factor = (1.0 + max(0.0, t_target)) / (1.0 + max(0.0, t_first))
    dt = max(0.0, t_target - t_first)
    earners = min(2.0, 1.0 + gamma * dt)
    med_rate = last_rate * earners / p_factor
    sigma = 0.75
    earners_n = max(1.0, last_rate / 100.0 * max(1.0, players or 1))
    sigma = math.sqrt(sigma * sigma + 1.0 / earners_n)
    return logit(med_rate), sigma


# --- per-trophy forecast ---------------------------------------------------

def forecast_trophy(trophy, obs, ctx):
    """obs: [(t_hours_since_release, logit_rate)] for this trophy, oldest
    first. ctx carries release-anchored times and counts. Returns
    (median_logit, sigma)."""
    last_rate = ctx["last_rate"]
    players = ctx["players_last"]
    t_target = ctx["t_target"]
    horizon_days = ctx["horizon_days"]
    stale_h = ctx["stale_h"]

    # A trophy everyone already has stays (nearly) everyone: on the logit
    # scale 100% and 95% are worlds apart, so pin the median at the top
    # and let a wide spread carry the doubt.
    if last_rate is not None and last_rate >= 99.5:
        return logit(99.99), 1.5

    earners_n = max(1.0, (last_rate or 0.0) / 100.0 * max(1.0, players or 1))
    sigma_binom = 1.0 / math.sqrt(earners_n)
    cls = classify(trophy)

    if not obs:
        # No read ever listed this trophy: use the case's typical rate.
        return logit(ctx["typical_rate"]), 1.2

    if len(obs) == 1:
        return dilution_prior(last_rate, players, ctx["t_first"], t_target, cls)

    ts = [t for t, _ in obs]
    ys = [y for _, y in obs]
    t1 = ts[0]
    growth_mult, _ = CLASS_PRIOR[cls]
    a_prior = logit(min(90.0, max(0.05, last_rate * growth_mult)))

    if len(obs) == 2:
        # Two points only: a log-time line through them, blended half and
        # half with the dilution prior, and a wide spread.
        a, b, _ = fit_loglin(ts, ys)
        med_fit = predict_loglin(a, b, t_target)
        med_prior, _ = dilution_prior(last_rate, players, ctx["t_first"], t_target, cls)
        med = 0.5 * med_fit + 0.5 * med_prior
        sigma = math.sqrt(0.8 ** 2 + sigma_binom ** 2)
        return med, min(1.6, sigma)

    # Three or more snapshots: three candidates on all data.
    #   * sat-exp on the logit rate (model-averaged over k);
    #   * decomposition: saturating earners / power-law players -- the
    #     only candidate that can turn a rate DOWN after it rose;
    #   * log-time linear (the house trend baseline's family).
    # The median is a BRACKETING BLEND, not a winner-pick. On a 4.5-day
    # horizon from 2.5 days of history, most curves have not bent yet:
    # the saturating fit then sits below the truth (it cannot see the
    # asymptote) while the log-time line sits above it (it never bends).
    # The truth sits between them, so their average in logit space is
    # the robust centre -- and a winner-pick by tail validation was
    # tried and dropped: a 24-hour tail check cannot tell extrapolators
    # apart, and it kept picking the log-time line alone, which
    # overshoots (on simulated launch weeks the blend beats either
    # candidate alone, and beats both house baselines).
    med_sat, sigma_model, sse_s = sat_exp_average(ts, ys, a_prior, t_target)
    dec = decomp_forecast(ctx["series"], t_target, cls)
    a_l, b_l, sse_l = fit_loglin(ts, ys)
    med_lin = predict_loglin(a_l, b_l, t_target)

    # Regime switch. The decomposition is the right structure only for
    # dilution-dominated trophies -- ones whose rate is flat or falling
    # while the tracked player count still grows fast. For a normal
    # trophy the earners keep pace with the players (its rate saturates,
    # its earners grow with the player base), and the decomposition's
    # saturating-earners assumption misfires badly, so it must not enter
    # the blend there. Detect the regime from the data: the recent logit
    # slope over the last third of snapshots, and the players growth
    # over the same window (a stale page has a flat rate AND a flat
    # count, so the growth check keeps a frozen page out of this regime).
    m = max(3, len(obs) // 3)
    rts, rys = ts[-m:], ys[-m:]
    n_r = len(rts)
    mx = sum(rts) / n_r
    my = sum(rys) / n_r
    den = sum((t - mx) ** 2 for t in rts)
    slope = (sum((t - mx) * (y - my) for t, y in zip(rts, rys)) / den
             if den > 1e-12 else 0.0)
    p_growth = 1.0
    if ctx["series"]:
        p_first = ctx["series"][-m][2]
        p_last_w = ctx["series"][-1][2]
        if p_first and p_first > 0:
            p_growth = p_last_w / p_first
    # Strictly falling, not merely slow: a rare trophy creeping up at
    # +0.002 logit/hour is still rising, and flagging it would drag its
    # forecast down toward the decomposition's saturating-earners world.
    dilution_dominated = slope < -0.002 and p_growth > 1.15

    if dec is not None and dilution_dominated:
        # The log-time line joins this blend only as fitted on the
        # RECENT window: on a hump-shaped series the full-window line's
        # slope is dominated by the early rise, so it extrapolates a
        # rise where the rate is falling, but the recent window's line
        # follows the fall. (Variant under test; see BUILD-REPORT.)
        a_lr, b_lr, _ = fit_loglin(ts[-m:], ys[-m:])
        med_lin_recent = predict_loglin(a_lr, b_lr, t_target)
        med = (med_sat + dec[0] + med_lin_recent) / 3.0
        sigma_model = max(sigma_model, dec[1])
        sse = sse_s
        sigma_asym = min(0.5, 0.5 * abs(med_sat - dec[0]))
    else:
        med = 0.5 * med_sat + 0.5 * med_lin
        sse = 0.5 * sse_s + 0.5 * sse_l
        sigma_asym = min(0.5, 0.5 * abs(med_sat - med_lin))

    n = len(obs)
    sigma_resid = math.sqrt(sse / max(1, n - 2))
    sigma_resid = min(0.8, max(0.12, sigma_resid))
    sigma_h = 0.30 * math.sqrt(max(0.0, horizon_days))
    sigma_stale = min(0.5, 0.15 * math.sqrt(max(0.0, stale_h)))
    sigma = math.sqrt(sigma_resid ** 2 + sigma_h ** 2 + sigma_asym ** 2
                      + sigma_stale ** 2 + sigma_binom ** 2
                      + min(0.6, sigma_model) ** 2)
    return med, min(1.6, max(0.25, sigma))


# --- case forecast ---------------------------------------------------------

def forecast_case(case, name):
    trophies = case["trophies"]
    release = parse_time((case.get("game") or {}).get("release_date"))
    if release is None:
        release = datetime(2026, 11, 19, tzinfo=timezone.utc)
    if release.tzinfo is None:
        release = release.replace(tzinfo=timezone.utc)

    at_dt = parse_time(case.get("at"))
    target_dt = parse_time(case.get("target"))
    horizon_min = case.get("horizon_minutes") or 0
    horizon_days = horizon_min / 1440.0
    if target_dt is not None:
        t_target = (target_dt - release).total_seconds() / 3600.0
    else:
        t_target = horizon_min / 60.0

    snaps = snapshots(case)
    if not snaps:
        raise ValueError(f"{name} has no history")
    t_first = None
    if snaps[0][0] is not None:
        t_first = (snaps[0][0] - release).total_seconds() / 3600.0
    else:
        t_first = 0.0
    last_slot, last_players, last_rates = snaps[-1]
    stale_h = 0.0
    if at_dt is not None and last_slot is not None:
        stale_h = max(0.0, (at_dt - last_slot).total_seconds() / 3600.0)

    # Typical last rate across trophies (fallback for a never-listed one).
    seen = [r for r in last_rates if r is not None and r < 99.5]
    typical_rate = sorted(seen)[len(seen) // 2] if seen else 1.0

    rows = []
    for i, trophy in enumerate(trophies):
        obs = []
        series = []
        for slot, _players, rates in snaps:
            if i < len(rates) and rates[i] is not None and slot is not None:
                t = (slot - release).total_seconds() / 3600.0
                obs.append((t, logit(rates[i])))
                series.append((t, rates[i], _players or 1))
        last_rate = last_rates[i] if i < len(last_rates) else None
        if last_rate is None:
            last_rate = typical_rate
        ctx = {
            "last_rate": last_rate,
            "players_last": last_players or 1,
            "t_target": t_target,
            "horizon_days": horizon_days,
            "stale_h": stale_h,
            "t_first": t_first,
            "typical_rate": typical_rate,
            "series": series,
        }
        med, sigma = forecast_trophy(trophy, obs, ctx)
        qs = []
        prev = None
        for z in Z:
            q = round4(clamp_pct(inv_logit(med + z * sigma)))
            if prev is not None and q < prev:
                q = prev
            qs.append(q)
            prev = q
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
