import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import *

# when the user comes back after the cache expired, the whole context is read at full price anyway.
# so this is the one moment when the context can be changed for free.
# idea: drop tool outputs from older turns (keep only the last turn) before sending the request
TEST_DAYS = ["2026-06-05", "2026-06-06", "2026-06-07"]

calls = load_calls()
calls = add_prev_call(calls)

models = pd.read_csv("results/model_cache_retention.csv")
short_ttl = set(models.loc[~models["keeps_cache"], "model"])

calls["cold"] = (calls["kind"] == "new_turn") & calls["model"].isin(short_ttl) & (calls["idle"] > CACHE_TTL)
cold = calls[calls["cold"]]
print("cold restarts:", len(cold))
print(f"cache hit on them: {cold['cached'].sum() / cold['prompt'].sum():.1%}")
print(f"tool outputs in their context: {cold['tool_tokens'].sum() / cold['prompt'].sum():.1%}")

# tool outputs that were already in the context before the previous turn started
first_in_turn = calls[["session_id", "turn", "tool_tokens"]].groupby(["session_id", "turn"], as_index=False).first()
first_in_turn["old_tool_tokens"] = first_in_turn.groupby("session_id")["tool_tokens"].shift()
calls = calls.merge(first_in_turn[["session_id", "turn", "old_tool_tokens"]], on=["session_id", "turn"], how="left")

# after a cold restart the removed tokens stay out of the context for the rest of the session
calls["removed"] = np.where(calls["cold"], calls["old_tool_tokens"].fillna(0), np.nan)
calls["removed"] = calls.groupby("session_id")["removed"].ffill().fillna(0)
calls["removed"] = np.minimum(calls["removed"], calls["tool_tokens"])

# a removed token would have cost the same as an average input token of that call
input_price = (calls["uncached"] * PRICE_INPUT + calls["cached"] * PRICE_CACHED) / calls["prompt"].replace(0, np.nan) / 1e6
calls["saving"] = calls["removed"] * input_price.fillna(0)
# the same split into tokens that would be paid in full and from cache, for other prices in 07_robustness.py
calls["removed_uncached"] = (calls["removed"] * calls["uncached"] / calls["prompt"].replace(0, np.nan)).fillna(0)
calls["removed_cached"] = calls["removed"] - calls["removed_uncached"]

total = calls["cost"].sum()
print(f"\nwhole week: saving ${calls['saving'].sum():,.0f} = {calls['saving'].sum() / total:.1%} of ${total:,.0f}")
print(f"only on the cold restart calls: {calls.loc[calls['cold'], 'saving'].sum() / total:.1%}")
print(f"tokens removed: {calls['removed'].sum() / calls['prompt'].sum():.1%} of all input tokens")

# how much of the dropped content would the agent read again?
# copilot already cleans old tool outputs by itself sometimes: tool tokens in the context fall to almost zero,
# user and agent messages stay. I use these cases as a natural experiment. I compare how much new tool output
# the agent adds in the 10 calls after such a cleanup vs. the 10 calls before, and the same for similar
# moments without a cleanup (difference in differences)
K = 10
g = calls.groupby("session_id")
calls["pos"] = g.cumcount()
calls["prev_tool"] = g["tool_tokens"].shift()
calls["cleanup"] = (calls["kind"].isin(["same_turn", "new_turn"]) & (calls["prev_tool"] > 20000)
                    & (calls["tool_tokens"] < 0.5 * calls["prev_tool"]))
calls["new_tool"] = (calls["tool_tokens"] - calls["prev_tool"]).clip(lower=0).fillna(0)
calls.loc[calls["cleanup"], "new_tool"] = 0
calls["cum_new"] = calls.groupby("session_id")["new_tool"].cumsum()
calls["cum_cleanup"] = calls.groupby("session_id")["cleanup"].cumsum()

g = calls.groupby("session_id")
before = g["cum_new"].shift(K + 1).where(calls["pos"] != K, 0)
calls["pre"] = g["cum_new"].shift(1) - before
calls["post"] = g["cum_new"].shift(-K) - calls["cum_new"]
calls["cleanups_around"] = g["cum_cleanup"].shift(-K) - g["cum_cleanup"].shift(K + 1).fillna(0)
calls["pos_bin"] = pd.cut(calls["pos"], [0, 10, 20, 40, 80, 160, 1e9], labels=False, right=False)
calls["tool_bin"] = pd.cut(calls["prev_tool"], [0, 20e3, 40e3, 60e3, 80e3, 100e3, 130e3, 1e9], labels=False, right=False)

# only cleanups at the start of a turn after a break > 5 min, the same moment when my rule works
ok = calls["pre"].notna() & calls["post"].notna() & (calls["kind"] == "new_turn") & (calls["idle"] > CACHE_TTL)
events = calls[ok & calls["cleanup"] & (calls["cleanups_around"] == 1)].copy()
control = calls[ok & (calls["cleanups_around"] == 0)]
ctrl_means = control.groupby(["pos_bin", "tool_bin"])[["pre", "post"]].mean()
events = events.join(ctrl_means, on=["pos_bin", "tool_bin"], rsuffix="_ctrl").dropna(subset=["pre_ctrl"])
events["removed_tool"] = events["prev_tool"] - events["tool_tokens"]
events["extra"] = (events["post"] - events["pre"]) - (events["post_ctrl"] - events["pre_ctrl"])
reread = events["extra"].sum() / events["removed_tool"].sum()

rng = np.random.default_rng(42)
boot = []
for i in range(500):
    idx = rng.integers(0, len(events), len(events))
    boot.append(events["extra"].values[idx].sum() / events["removed_tool"].values[idx].sum())
lo, hi = np.percentile(boot, [2.5, 97.5])
print(f"\ncleanups by copilot at turn start: {len(events)} (in {events['session_id'].nunique()} sessions)")
print(f"tool tokens removed per cleanup: {events['removed_tool'].mean():,.0f}")
print(f"read again in the next {K} calls: {reread:.1%} of removed (95% CI {lo:.1%} to {hi:.1%})")
pd.DataFrame({"reread": [reread], "low": [lo], "high": [hi]}).to_csv("results/reread.csv", index=False)

offsets = [o for o in range(-K, K + 1) if o != 0]
new_tool = calls["new_tool"].values
ev_profile = np.array([new_tool[events.index + o].mean() for o in offsets])
weights = events.groupby(["pos_bin", "tool_bin"]).size()
ctrl_profile = np.zeros(len(offsets))
for (pb, tb), n in weights.items():
    rows = control.index[(control["pos_bin"] == pb) & (control["tool_bin"] == tb)]
    ctrl_profile += n * np.array([new_tool[rows + o].mean() for o in offsets])
ctrl_profile = ctrl_profile / weights.sum()

# sessions with a cleanup read more tools in general, so I move the control line to the same level
# before the message. the gap after the message is the extra reading
gap = ev_profile[np.array(offsets) < 0].mean() - ctrl_profile[np.array(offsets) < 0].mean()
expected = ctrl_profile + gap

fig, ax = plt.subplots(figsize=(9, 4))
x = np.array(offsets)
post = x > 0
ax.plot(x[~post], ev_profile[~post] / 1000, color=RED, marker="o", linewidth=2, label="Copilot dropped old tool outputs")
ax.plot(x[post], ev_profile[post] / 1000, color=RED, marker="o", linewidth=2)
ax.plot(x[~post], expected[~post] / 1000, color=GRAY, marker="o", linewidth=2, linestyle="--", label="Expected without a cleanup")
ax.plot(x[post], expected[post] / 1000, color=GRAY, marker="o", linewidth=2, linestyle="--")
ax.fill_between(x[post], expected[post] / 1000, ev_profile[post] / 1000, color=RED, alpha=0.1)
ax.axvline(0, color=DARK, linestyle="--", linewidth=1)
ax.text(-0.3, ax.get_ylim()[1] * 0.97, "new user\nmessage", color=DARK, va="top", ha="right")
ax.set_xticks(range(-K, K + 1, 2))
ax.set_xlabel("Model calls before / after the user's new message")
ax.set_ylabel("New tool output per call, k tokens")
ax.set_title(f"After a cleanup the agent reads back about {reread:.0%} of what was removed")
ax.legend(frameon=False, loc="upper right")
plt.tight_layout()
plt.savefig("images/reread_after_cleanup.png")
plt.close()

test = calls["date"].isin(TEST_DAYS)
test_bill = calls.loc[test, "cost"].sum()
gross = calls.loc[test, "saving"].sum()
prune = gross * (1 - reread) / test_bill * 100
prune_lo = gross * (1 - hi) / test_bill * 100
prune_hi = gross * (1 - lo) / test_bill * 100
print(f"\ntest days, measured re-read: {prune:.1f}% of the bill (95% CI {prune_lo:.1f}% to {prune_hi:.1f}%)")
scenarios = {}
for r in [0, 0.25, 0.5]:
    scenarios[r] = gross * (1 - r) / test_bill * 100
    print(f"test days, if {r:.0%} is read again: {scenarios[r]:.1f}% of the bill")

# one row per session, used in 06_ab_test_design.py
# cost before the first cold restart can't be changed by the rule, so it is a fair covariate for the test
calls["after_trigger"] = calls.groupby("session_id")["cold"].cummax()
calls["before_trigger_cost"] = calls["cost"].where(~calls["after_trigger"], 0)
turns_before = calls[~calls["after_trigger"]].groupby("session_id")["turn"].nunique()
sessions = calls.groupby("session_id").agg(
    date=("date", "first"),
    cost=("cost", "sum"),
    before_trigger_cost=("before_trigger_cost", "sum"),
    saving=("saving", "sum"),
    removed_uncached=("removed_uncached", "sum"),
    removed_cached=("removed_cached", "sum"),
    triggered=("cold", "any"),
    turns=("turn", "nunique"),
).reset_index()
sessions["turns_before_trigger"] = sessions["session_id"].map(turns_before).fillna(0)
sessions["saving_net"] = sessions["saving"] * (1 - reread)
sessions.to_parquet("data/sessions.parquet", index=False)

# compare with keeping the cache warm (from 04_keep_warm.py), same test days
kw = pd.read_csv("results/keep_warm_policies.csv")
kw = kw[kw["chosen"]].iloc[0]

out = pd.DataFrame({
    "fix": ["keep warm, everyone", "keep warm, model", "keep warm, perfect foresight",
            "drop old tool outputs, measured re-read", "drop old tool outputs, 0% re-read",
            "drop old tool outputs, 25% re-read", "drop old tool outputs, 50% re-read"],
    "saving_pct_of_bill": [kw["ping_everyone_pct"], kw["model_pct"], kw["oracle_pct"], prune,
                           scenarios[0], scenarios[0.25], scenarios[0.5]],
})
out.to_csv("results/fixes_compared.csv", index=False)
print(out)

labels = ["Keep the cache warm\nfor everyone",
          "Keep it warm when the model\nexpects the user back",
          "Keep it warm with perfect\nforesight (upper bound)",
          "Drop old tool outputs when\nthe cache is already cold"]
vals = [kw["ping_everyone_pct"], kw["model_pct"], kw["oracle_pct"], prune]

fig, ax = plt.subplots(figsize=(9, 3.8))
y = np.arange(len(vals))[::-1]
ax.barh(y, vals, color=[GRAY, GRAY, "#cbd5e0", BLUE], height=0.6)
ax.errorbar(prune, y[-1], xerr=[[prune - prune_lo], [prune_hi - prune]], color=DARK, capsize=4, linewidth=1)
ax.axvline(0, color=DARK, linewidth=0.8)
for yi, v in zip(y[:3], vals[:3]):
    ax.text(max(v, 0) + 0.25, yi, f"{v:+.1f}%", va="center", color=DARK)
ax.text(prune_hi + 0.2, y[-1], f"{prune:+.1f}%", va="center", color=BLUE, fontweight="bold")
ax.set_yticks(y)
ax.set_yticklabels(labels)
ax.set_xlim(min(vals) - 0.5, prune_hi + 2)
ax.set_xlabel("Net saving, % of the total bill (test days Jun 5-7)")
ax.set_title("Cleaning old context beats keeping the cache warm")
plt.tight_layout()
plt.savefig("images/fixes_compared.png")
plt.close()
