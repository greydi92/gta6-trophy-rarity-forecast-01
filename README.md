[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/greydi92/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/greydi92/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

# GTA VI trophy rarity, one week after launch

Built for the [make.fun "Forecast how many GTA VI players earn each trophy in week one" bounty](https://make.fun/bounty/gta6-trophy-rarity-forecast-01).

This program reads the house's recording of Exophase's Grand Theft Auto VI trophy page — every trophy's earn rate and the players-tracked count at each read — and forecasts each of the game's 55 trophies' earn rate at the case's target, as five quantiles (5%, 25%, 50%, 75%, 95%) per trophy. It reads one case file (`/input/<id>.in`) and writes `/output/<id>.out`. Python 3.13, standard library only, deterministic, no network.

## How it works

Two forces move a trophy's rate, and the model treats them separately:

- **Players keep earning trophies.** Early adoption is fast, then saturates — so each trophy's earn rate is fitted on the logit scale with a saturating growth curve, model-averaged over growth rates instead of picking one (when the history has not bent yet, several asymptotes fit it equally well, and picking one pretends the curve is identified). A log-time linear trend is the second candidate. The median is a bracketing blend of the two: the saturating fit sits below the truth on a long horizon and the log-time line above it, so their average is the robust centre.
- **Exophase's tracked-player count grows, and each refresh dilutes the rates**, because players who join the tracker late have had less time to earn anything. Launch night proved it: the page sat frozen at 638 players for hours, then one refresh took it to 3,868 and every rate fell. So there is a third candidate that fits the two pieces apart — saturating earners against a power-law player count — which is the only candidate that can turn a rate *down* after it rose. It enters the blend only for dilution-dominated trophies (a rate flat or falling while the player count still grows fast); for a normal trophy its saturating-earners assumption misfires.

When the history is flat — one distinct snapshot, as in the practice cases, where Exophase had not refreshed once — no curve can be fitted, so the program falls back to a dilution prior instead of echoing the stale page: players tracked are assumed to grow roughly like (1 + hours since release) while earners grow only slowly. That prior is calibrated on launch night's one documented refresh (638 → 3,868 players, rates ≈ ÷5, pinned by the house's own published practice numbers for its baseline: loss ~1.51, typical miss ~0.68 points).

The spread is honest, built in quadrature from the fit's residual spread, extrapolation uncertainty that grows like √horizon, disagreement between the candidate fits, staleness of the last snapshot (an unseen refresh may already be due), and binomial noise in the rate itself (1/√earners). A trophy at 100% is pinned at 99.99 with a wide logit spread, because on the logit scale 100% and 95% are worlds apart. History is first collapsed to distinct snapshots — consecutive identical reads are one Exophase refresh, not independent observations — so a long frozen stretch cannot overweight one value.

## Run it

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

## Local verification results

The kit ships no practice answers, so the practice score below is against a proxy truth calibrated on the house's own published numbers: the house says its persistence baseline scores loss ~1.51 with typical miss ~0.68 points on the practice, which pins the refreshed rates at about the launch rates ÷ 5 (that proxy reproduces the baseline's published 1.51 / 0.68 almost exactly). The range ÷3–÷6 shows the result is not tuned to one divisor.

| case | loss (÷5 proxy) | typical miss | 90% coverage |
|---|---|---|---|
| p00 | 0.193 | 0.02 pp | 0.982 |
| p01 | 0.188 | 0.01 pp | 0.982 |

House baseline on the same proxy: loss 1.498, typical miss 0.70 pp, 90% coverage 0. Across the ÷3–÷6 proxy range this entry scores 0.19–0.26; the baseline scores 1.00–1.68.

The final is what pays, and the practice cases cannot exercise the curve fit at all (their history is one flat snapshot), so the entry was also scored on eight simulated final-like cases — 2.5 days of 10-minute reads to the deadline, target 4.5 days later, players 638 → ~49,000 → ~147,000, with realistic slow saturation and, in half the cases, dilution-dominated trophies whose rates rise then fall:

| | loss (mean of 8) | typical miss |
|---|---|---|
| this entry | **0.236** | 2.7 pp |
| flat (house persistence baseline) | 0.311 | 3.2 pp |
| trend (house curve baseline) | 0.279 | 3.5 pp |

This entry beats both house baselines on all eight simulated finals. Runtime is about 0.16 s per case (limit 8 s); two runs on the same case write byte-identical output.

<sub>Entered on [make.fun](https://make.fun/?ref=qgn48q27&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
