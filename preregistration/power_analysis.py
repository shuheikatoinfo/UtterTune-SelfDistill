"""Power analysis behind the stimulus allocation (PREREGISTRATION.md §3, §5.1).

Run:  scripts/eval/.venv_metrics/bin/python preregistration/power_analysis.py

The allocation is derived here, not chosen to fill a budget. Two findings drive it:

  * H1 (superiority, McNemar exact, Holm α=0.025) is already saturated at 50 items on ONE
    speaker for any effect this method plausibly produces. A second speaker adds nothing.
  * H3 (non-inferiority, Δ=0.15) is the binding constraint. At a realistic 15-20% inter-rater
    discordance, 50 items on one speaker fails to establish non-inferiority 23-33% of the time
    EVEN WHEN THE TRUE DIFFERENCE IS ZERO. Doubling to 100 paired observations (2 speakers)
    lifts that to 90-96%.

So the speakers go where the power is needed: the locality cells, not the headline cells.
"""
import numpy as np
from math import comb

rng = np.random.default_rng(0)


def mcnemar_p(b, c):
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(comb(n, k) for k in range(0, min(b, c) + 1)) / 2 ** n)


def power_h1(p0, p1, n, alpha=0.025, reps=4000):
    """Paired: item difficulty is shared, so arms are monotonically coupled (realistic)."""
    hits = 0
    for _ in range(reps):
        u = rng.random(n)
        a, b = (u < p0), (u < p1)
        hits += mcnemar_p(int(np.sum(a & ~b)), int(np.sum(~a & b))) < alpha
    return hits / reps


def power_h3(n, q_disc, true_d=0.0, margin=0.15, reps=3000, B=600):
    """Non-inferiority on the paired difference. q_disc = fraction of items the two arms
    disagree on; true_d = the real difference (0 = no degradation)."""
    ok = 0
    for _ in range(reps):
        d = np.zeros(n)
        nb = int(round(n * (q_disc + abs(true_d)) / 2))
        nc = max(0, int(round(n * (q_disc - abs(true_d)) / 2)))
        nb = max(0, min(n, nb)); nc = min(nc, n - nb)
        d[:nb] = -1.0 if true_d <= 0 else 1.0
        d[nb:nb + nc] = 1.0 if true_d <= 0 else -1.0
        rng.shuffle(d)
        bs = np.array([np.mean(rng.choice(d, n, replace=True)) for _ in range(B)])
        ok += np.percentile(bs, 2.5) > -margin
    return ok / reps


if __name__ == "__main__":
    print("H1 superiority — power at Holm α=0.025")
    for lab, (p0, p1) in {"0.30->0.90": (.30, .90), "0.50->0.85": (.50, .85),
                          "0.60->0.80": (.60, .80), "0.70->0.85": (.70, .85)}.items():
        print(f"  {lab}  n=50: {power_h1(p0,p1,50):.2f}   n=100: {power_h1(p0,p1,100):.2f}")
    print("\nH3 non-inferiority (Δ=0.15) — P(conclude NI | true difference = 0)")
    for q in (0.10, 0.15, 0.20, 0.30):
        print(f"  discordance {q:.0%}  n=50: {power_h3(50,q):.2f}   n=100: {power_h3(100,q):.2f}")
    print("\nH3 — P(wrongly conclude NI | true difference = -0.10)")
    for q in (0.20, 0.30):
        print(f"  discordance {q:.0%}  n=50: {power_h3(50,q,-0.10):.2f}   n=100: {power_h3(100,q,-0.10):.2f}")


# ---------------------------------------------------------------------------
# H4 added 2026-09-01. H4 (data-free vs the released UtterTune adapter on CosyVoice2) sits in
# the same family and at the same n as H3, and the original derivation covered only H3 --- but
# H4 is the paper's headline claim, so it needs its own number.
#
#   P(conclude non-inferiority | the two arms are TRULY identical), Delta = 0.15
#
#     discordance |  n=50   n=75   n=100  n=150  n=200
#            10%  |  0.91   0.98   0.99   1.00   1.00
#            15%  |  0.78   0.92   0.96   1.00   1.00
#            20%  |  0.68   0.84   0.90   0.98   1.00
#            25%  |  0.58   0.76   0.83   0.96   0.98
#            30%  |  0.51   0.69   0.77   0.92   0.97
#
#   Same, with Delta widened to 0.20:
#            20%  |  0.86   0.96   0.99   1.00   1.00
#            25%  |  0.78   0.92   0.97   1.00   1.00
#
# At n=50 and the discordance an accent judgment realistically produces (20-25% when both
# arms score near 0.8), the test concludes non-inferiority only 58-68% of the time even when
# the arms are genuinely equal. Delta=0.15 is at the resolution limit of n=50: the 95% CI
# half-width on the risk difference is ~0.12 at 20% discordance, so the margin and the noise
# are the same size. This is a property of the design, known before data, not a result.

# ---------------------------------------------------------------------------
# CORRECTION 2026-09-01 to power_h1 above.
#
# power_h1() couples the two arms as  a = (u < p0), b = (u < p1)  from ONE uniform draw.
# That is perfect comonotonicity: every discordant pair falls the same way, b = 0 always.
# It is the best case, not a realistic one, and it is why the section below originally
# reported "power 1.00 at n=50". Real paired data has discordance in BOTH directions --- a
# word the base happens to get right and the edited system gets wrong.
#
# McNemar depends only on the two discordant counts, so the honest parameter is the TOTAL
# discordance q. q = |d| is the perfectly-coupled floor; anything above it is symmetric noise.
#
#   Power at n=50, Holm strictest alpha (3 tests = .0167 / 5 tests = .0100):
#
#     accent effect |  q=d+0%   q=d+10%  q=d+20%  q=d+30%      (3 tests / 5 tests)
#              18%  | .82/.70   .44/.34  .29/.23  .22/.17
#              25%  | .98/.96   .71/.63  .55/.46  .45/.36
#              35%  | 1.0/1.0   .93/.90  .83/.77  .75/.67
#              45%  | 1.0/1.0   .99/.99  .96/.94  .92/.88
#
# Reading of this table, fixed in advance:
#   * The design is adequate iff the accent effect is >= ~0.35. That is what the paper's own
#     premise predicts: an unedited base has no channel through which to place an accent
#     nucleus, so it matches the prescribed type only by chance.
#   * If the observed effect is nearer 0.18-0.25, no n we can afford resolves it, and the
#     study reports estimates with CIs rather than claiming a null. Stated here so that a
#     small effect is reported as unresolved, not as evidence of absence.
#   * Growing Family A from 3 to 5 tests (H6/H7, the second-speaker replications) costs about
#     0.06 of power at the expected effect size. That is affordable; the replication is not
#     traded against the main result.
