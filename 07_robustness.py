import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import *

# how much can I trust the main numbers?
# 1. 95% confidence intervals with a bootstrap over sessions
# 2. other price lists
# 3. every day separately
TEST_DAYS = ["2026-06-05", "2026-06-06", "2026-06-07"]
BOOT = 1000

s = pd.read_parquet("data/sessions_cache.parquet")
s = s.merge(pd.read_parquet("data/sessions.parquet")[["session_id", "removed_uncached", "removed_cached"]], on="session_id")
s = s.merge(pd.read_parquet("data/keep_warm_test_sessions.parquet"), on="session_id", how="left")
s[["everyone", "model", "oracle"]] = s[["everyone", "model", "oracle"]].fillna(0)
reread = pd.read_csv("results/reread.csv")["reread"][0]
s["test"] = s["date"].isin(TEST_DAYS)


def money(df, cached_share=PRICE_CACHED / PRICE_INPUT, write_fee=1.0):
    # prices relative to the input price. every result is a % of the bill, so the price level does not matter.
    # write_fee: some providers charge extra for the tokens that go into the cache
    miss = write_fee
    hit = cached_share
    out = PRICE_OUTPUT / PRICE_INPUT
    m = pd.DataFrame({"date": df["date"], "test": df["test"]})
    m["bill"] = df["uncached"] * miss + df["cached"] * hit + df["completion"] * out
    m["idle"] = df["idle_extra_tokens"] * (miss - hit)
    m["switch"] = df["switch_extra_tokens"] * (miss - hit)
    m["prune"] = (df["removed_uncached"] * miss + df["removed_cached"] * hit) * (1 - reread)
    return m


def shares(m):
    t = m[m["test"]]
    return {
        "idle": m["idle"].sum() / m["bill"].sum() * 100,
        "switch": m["switch"].sum() / m["bill"].sum() * 100,
        "prune_week": m["prune"].sum() / m["bill"].sum() * 100,
        "prune_test": t["prune"].sum() / t["bill"].sum() * 100 if len(t) else np.nan,
    }


# 1. confidence intervals, reference prices
m = money(s)
m["bill_usd"] = m["bill"] * PRICE_INPUT / 1e6
for c in ["everyone", "model", "oracle"]:
    m["kw_" + c] = s[c]

cols = ["bill", "idle", "switch", "prune", "bill_usd", "kw_everyone", "kw_model", "kw_oracle"]
values = m[cols].values
test = m["test"].values


def all_numbers(v, t):
    total = v.sum(axis=0)
    total_t = v[t].sum(axis=0)
    return [
        total[1] / total[0] * 100,
        total[2] / total[0] * 100,
        total_t[5] / total_t[4] * 100,
        total_t[6] / total_t[4] * 100,
        total_t[7] / total_t[4] * 100,
        total_t[3] / total_t[0] * 100,
    ]


names = ["User idle > 5 min", "Model switches", "Keep warm, everyone", "Keep warm, model",
         "Keep warm, perfect foresight", "Drop old tool outputs\non cold restart"]
point = all_numbers(values, test)
rng = np.random.default_rng(42)
boot = []
for i in range(BOOT):
    idx = rng.integers(0, len(values), len(values))
    boot.append(all_numbers(values[idx], test[idx]))
boot = np.array(boot)
ci = pd.DataFrame({
    "number": [n.replace("\n", " ") for n in names],
    "period": ["whole week"] * 2 + ["test days"] * 4,
    "pct_of_bill": point,
    "ci_low": np.percentile(boot, 2.5, axis=0),
    "ci_high": np.percentile(boot, 97.5, axis=0),
})
ci.to_csv("results/robustness_ci.csv", index=False)
print(ci.round(2).to_string(index=False))

fig, ax = plt.subplots(figsize=(9, 4))
y = np.array([7, 6, 3.5, 2.5, 1.5, 0.5])
colors = [RED, GRAY, GRAY, GRAY, GRAY, BLUE]
for yi, p, lo, hi, c in zip(y, ci["pct_of_bill"], ci["ci_low"], ci["ci_high"], colors):
    ax.plot([lo, hi], [yi, yi], color=c, linewidth=6, solid_capstyle="butt", alpha=0.4)
    ax.scatter(p, yi, color=c, s=40, zorder=3)
    ax.text(max(hi, 0) + 0.2, yi, f"{p:+.1f}%  [{lo:+.1f}, {hi:+.1f}]", va="center", fontsize=9, color=DARK)
ax.axvline(0, color=DARK, linewidth=0.8)
ax.set_yticks(y)
ax.set_yticklabels(names)
ax.text(-1.4, 7.75, "What it costs (whole week)", fontsize=9, color="#4a5568", fontweight="bold")
ax.text(-1.4, 4.25, "What the fixes save (test days Jun 5-7)", fontsize=9, color="#4a5568", fontweight="bold")
ax.set_xlim(-1.5, 11)
ax.set_ylim(0, 8.2)
ax.set_xlabel("% of the total bill, 95% confidence interval (bootstrap over sessions)")
ax.set_title("Random noise does not change any conclusion")
plt.tight_layout()
plt.savefig("images/robustness_ci.png")
plt.close()

# 2. other price lists
scenarios = [
    ("Cache = 10% of input\n(reference)", 0.1, 1.0),
    ("Cache = 25%", 0.25, 1.0),
    ("Cache = 50%", 0.5, 1.0),
    ("Cache = 10%,\n+25% to write it", 0.1, 1.25),
]
rows = []
for name, cached_share, fee in scenarios:
    r = shares(money(s, cached_share, fee))
    r["prices"] = name.replace("\n", " ")
    rows.append(r)
prices = pd.DataFrame(rows)
prices.to_csv("results/robustness_prices.csv", index=False)
print(prices.round(2).to_string(index=False))

# 3. every day
days = []
for d, part in s.groupby("date"):
    r = shares(money(part))
    r["date"] = d
    days.append(r)
days = pd.DataFrame(days)
days = days.drop(columns="prune_test")
days.to_csv("results/robustness_days.csv", index=False)
print(days.round(2).to_string(index=False))
days["day"] = pd.to_datetime(days["date"]).dt.strftime("%a\n%b %d")

fig, axes = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={"width_ratios": [1.1, 1]})
ax = axes[0]
x = np.arange(len(prices))
w = 0.26
ax.bar(x - w, prices["idle"], w, color=RED, label="Cost of user idle > 5 min")
ax.bar(x, prices["switch"], w, color=GRAY, label="Cost of model switches")
ax.bar(x + w, prices["prune_test"], w, color=BLUE, label="Saving of fix 2")
for i in range(len(prices)):
    for dx, col in [(-w, "idle"), (0, "switch"), (w, "prune_test")]:
        ax.text(i + dx, prices[col][i] + 0.15, f"{prices[col][i]:.1f}", ha="center", fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels([n for n, _, _ in scenarios], fontsize=8.5)
ax.set_ylabel("% of the total bill")
ax.set_ylim(0, max(prices[["idle", "prune_test"]].max()) * 1.5)
ax.set_title("Other price lists")
ax.legend(frameon=False, fontsize=8.5, loc="upper left")

ax = axes[1]
ax.plot(days["day"], days["idle"], color=RED, marker="o", linewidth=2)
ax.plot(days["day"], days["switch"], color=GRAY, marker="o", linewidth=2)
ax.plot(days["day"], days["prune_week"], color=BLUE, marker="o", linewidth=2)
last = len(days) - 1
ax.text(last + 0.15, days["prune_week"].iloc[-1], "fix 2", color=BLUE, va="center", fontsize=8.5)
ax.text(last + 0.15, days["idle"].iloc[-1], "idle > 5 min", color=RED, va="center", fontsize=8.5)
ax.text(last + 0.15, days["switch"].iloc[-1], "switches", color="#718096", va="center", fontsize=8.5)
ax.set_xlim(-0.4, last + 1.4)
ax.set_ylim(0, max(days[["idle", "prune_week"]].max()) * 1.5)
ax.tick_params(axis="x", labelsize=8.5)
ax.set_ylabel("% of the bill of that day")
ax.set_title("Every day separately")
plt.tight_layout()
plt.savefig("images/robustness_prices_days.png")
plt.close()
