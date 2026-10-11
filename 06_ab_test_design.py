import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

from config import *

# how big should an A/B test of fix 2 be?
# I simulate many experiments on real sessions: split them into A and B at random, take the measured
# saving away from the cost of group B and check how often the test finds the difference
ALPHA = 0.05
POWER = 0.8
SIMS = 400
SIZES = [1000, 2000, 5000, 10000, 20000, 40000]
z = norm.ppf(1 - ALPHA / 2) + norm.ppf(POWER)

sessions = pd.read_parquet("data/sessions.parquet")
calls = pd.read_parquet("data/calls", columns=["session_id", "result"])
calls["cancel"] = calls["result"] == "UserCancel"
cancel = calls.groupby("session_id")["cancel"].any()
sessions["user_cancel"] = sessions["session_id"].map(cancel).astype(int)

y = sessions["cost"].values
x = sessions["before_trigger_cost"].values
effect = sessions["saving_net"].values
trig = sessions["triggered"].values
print(f"sessions: {len(y)}, the rule fires in {trig.mean():.1%} of them, they make {y[trig].sum() / y.sum():.0%} of the cost")
print(f"effect: -{effect.sum() / y.sum():.1%} on all sessions, -{effect[trig].sum() / y[trig].sum():.1%} on sessions where the rule fires")
print(f"corr(cost, cost before the rule fires): all {np.corrcoef(y, x)[0, 1]:.2f}, "
      f"rule fires {np.corrcoef(y[trig], x[trig])[0, 1]:.2f}")


def p_value(a, b, xa=None, xb=None):
    if xa is not None:
        # CUPED: remove the part of the cost that is explained by the covariate
        x_all = np.concatenate([xa, xb])
        y_all = np.concatenate([a, b])
        theta = np.cov(y_all, x_all)[0, 1] / x_all.var(ddof=1)
        a = a - theta * (xa - x_all.mean())
        b = b - theta * (xb - x_all.mean())
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return 2 * norm.sf(abs((b.mean() - a.mean()) / se))


def one_experiment(n, with_effect, rng):
    idx = rng.permutation(len(y))[:2 * n]
    a, b = idx[:n], idx[n:]
    yb = y[b] - effect[b] if with_effect else y[b]
    # in a real test the control group also logs "the rule would fire here", so both groups can be filtered
    fa, fb = trig[a], trig[b]
    return [
        p_value(y[a], yb),
        p_value(y[a], yb, x[a], x[b]),
        p_value(y[a][fa], yb[fb]),
        p_value(y[a][fa], yb[fb], x[a][fa], x[b][fb]),
    ]


methods = ["All sessions", "All sessions + CUPED", "Only sessions where the rule fires",
           "Only sessions where the rule fires + CUPED"]
rng = np.random.default_rng(42)

# A/A test first: without any effect every method should find a "difference" only in 5% of cases
aa = np.array([one_experiment(20000, False, rng) for i in range(1000)])
print("\nA/A test, false positive rate:")
for m, v in zip(methods, (aa < ALPHA).mean(axis=0)):
    print(f"  {m}: {v:.1%}")

power = []
for n in SIZES:
    res = np.array([one_experiment(n, True, rng) for i in range(SIMS)])
    power.append((res < ALPHA).mean(axis=0))
    print(f"n={n}: power " + ", ".join(f"{m} {v:.2f}" for m, v in zip(methods, power[-1])))
power = np.array(power)

# the same with a formula, n per group = 2 * z^2 * variance / effect^2
rho = np.corrcoef(y, x)[0, 1]
rho_t = np.corrcoef(y[trig], x[trig])[0, 1]
n_plain = 2 * z ** 2 * y.var() / effect.mean() ** 2
n_t = 2 * z ** 2 * y[trig].var() / effect[trig].mean() ** 2 / trig.mean()
needed = [n_plain, n_plain * (1 - rho ** 2), n_t, n_t * (1 - rho_t ** 2)]

# guardrails: the rule must not make the agent worse. the rule can only change sessions where it fires,
# so I look at them. smallest changes I would want to see: +2% turns, +5% (relative) sessions with a user cancel
t = sessions[sessions["triggered"]]
turns_mde = 0.02 * t["turns"].mean()
rho_turns = np.corrcoef(t["turns"], t["turns_before_trigger"])[0, 1]
cancel_rate = t["user_cancel"].mean()
cancel_mde = 0.05 * cancel_rate
n_turns = 2 * z ** 2 * t["turns"].var() * (1 - rho_turns ** 2) / turns_mde ** 2 / trig.mean()
n_cancel = 2 * z ** 2 * cancel_rate * (1 - cancel_rate) / cancel_mde ** 2 / trig.mean()
print(f"\nguardrails on sessions where the rule fires: turns {t['turns'].mean():.1f} "
      f"(corr with turns before the rule fires {rho_turns:.2f}), user cancel {cancel_rate:.1%}")

table = pd.DataFrame({
    "metric": ["cost per session"] * 4 + ["turns per session (guardrail)", "sessions with a user cancel (guardrail)"],
    "method": methods + ["only sessions where the rule fires + CUPED", "only sessions where the rule fires"],
    "change_to_detect": [f"-{effect.sum() / y.sum():.1%}"] * 4 + ["+2%", "+5% relative"],
    "sessions_per_group": np.round(needed + [n_turns, n_cancel], -2).astype(int),
})
table.to_csv("results/ab_test_sizes.csv", index=False)
print(table.to_string(index=False))

colors = [GRAY, DARK, "#90cdf4", BLUE]
fig, ax = plt.subplots(figsize=(9, 4.2))
for i, m in enumerate(methods):
    ax.plot(SIZES, power[:, i] * 100, color=colors[i], marker="o", linewidth=2,
            label=f"{m}: {needed[i] / 1000:.1f}k")
ax.axhline(POWER * 100, color=RED, linestyle="--", linewidth=1)
ax.text(SIZES[0], POWER * 100 + 2, "80% power", color=RED)
ax.set_xscale("log")
ax.set_xticks(SIZES)
ax.set_xticklabels([f"{s // 1000}k" for s in SIZES])
ax.set_ylim(0, 105)
ax.set_xlabel("Sessions per group (all sessions that enter the test)")
ax.set_ylabel("Chance to detect the saving, %")
ax.set_title("Filtering and CUPED cut the test size in half")
ax.legend(title="Sessions per group for 80% power", frameon=False, loc="lower right", fontsize=8.5, title_fontsize=9)
plt.tight_layout()
plt.savefig("images/ab_test_power.png")
plt.close()

# what decides the size of the test
labels = ["Saving: cost per session, -7.8%", "Guardrail: sessions with a user cancel, +5% rel.", "Guardrail: turns per session, +2%"]
vals = [needed[3], n_cancel, n_turns]
fig, ax = plt.subplots(figsize=(9, 2.8))
ax.barh(labels[::-1], np.array(vals[::-1]) / 1000, color=[RED, RED, BLUE], height=0.55)
for i, v in enumerate(vals[::-1]):
    ax.text(v / 1000 + 2, i, f"{v / 1000:.0f}k", va="center")
ax.set_xlim(0, max(vals) / 1000 * 1.15)
ax.set_xlabel("Sessions per group for 80% power (best method for each metric)")
ax.set_title("Guardrails, not the saving, set the test size")
plt.tight_layout()
plt.savefig("images/ab_test_size.png")
plt.close()
