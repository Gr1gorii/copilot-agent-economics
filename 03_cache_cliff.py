import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import *

calls = load_calls()
calls = add_prev_call(calls)
total_cost = calls["cost"].sum()
print(calls["kind"].value_counts())


def hit_rate(df):
    return df["cached"].sum() / df["prompt"].sum()


# which models lose the cache after 5 minutes of user idle time
nt = calls[calls["kind"] == "new_turn"]
rows = []
for m, d in nt.groupby("model"):
    warm = d[d["idle"].between(10, 240)]
    late = d[d["idle"].between(360, 1800)]
    if len(late) < 200:
        continue
    rows.append([m, len(d), hit_rate(warm), hit_rate(late)])
models = pd.DataFrame(rows, columns=["model", "new_turns", "hit_warm", "hit_after_5min"])
models["keeps_cache"] = models["hit_after_5min"] / models["hit_warm"] > 0.5
print(models.sort_values("new_turns", ascending=False).to_string(index=False))
short_ttl = set(models.loc[~models["keeps_cache"], "model"])
long_ttl = set(models.loc[models["keeps_cache"], "model"])
models.to_csv("results/model_cache_retention.csv", index=False)

calls["group"] = "other"
calls.loc[calls["model"].isin(short_ttl), "group"] = "short"
calls.loc[calls["model"].isin(long_ttl), "group"] = "long"
print("share of cost by group:", (calls.groupby("group")["cost"].sum() / total_cost).round(3).to_dict())

# hit rate by idle minutes
nt = calls[calls["kind"] == "new_turn"].copy()
edges = list(range(0, 21))
nt["minute"] = pd.cut(nt["idle"] / 60, edges, labels=edges[:-1])
curve = {}
for grp in ["short", "long"]:
    d = nt[nt["group"] == grp]
    s = d.groupby("minute", observed=True)[["cached", "prompt"]].sum()
    curve[grp] = s["cached"] / s["prompt"] * 100
    print(grp, curve[grp].round(1).to_dict())

fig, ax = plt.subplots(figsize=(9, 4.2))
x = np.array(edges[:-1]) + 0.5
ax.axvspan(5, 20, color=RED, alpha=0.06)
ax.plot(x, curve["short"].values, color=RED, marker="o", linewidth=2, label=f"{len(short_ttl)} models with a 5 min cache")
ax.plot(x, curve["long"].values, color=BLUE, marker="o", linewidth=2, label=f"{len(long_ttl)} models with a longer cache")
ax.axvline(5, color=DARK, linestyle="--", linewidth=1)
ax.text(5.2, 95, "5 minutes", color=DARK)
ax.set_xlim(0, 20)
ax.set_ylim(0, 100)
ax.set_xticks(range(0, 21, 2))
ax.set_xlabel("Minutes between the agent's last answer and the user's next message")
ax.set_ylabel("Input tokens served from cache, %")
ax.set_title("The user waits 5 minutes and the agent has to pay for the whole context again")
ax.legend(frameon=False, loc="center right")
plt.tight_layout()
plt.savefig("images/cache_cliff.png")
plt.close()

# extra money vs a warm cache
saving_per_token = (PRICE_INPUT - PRICE_CACHED) / 1e6

warm_rate = models.set_index("model")["hit_warm"]
late = calls[(calls["kind"] == "new_turn") & (calls["idle"] > CACHE_TTL) & (calls["group"] == "short")]
late_extra = (late["model"].map(warm_rate) * late["prompt"] - late["cached"]).clip(lower=0) * saving_per_token

# a switch also drops the cache. compare with normal calls that came after the same idle time
ref_same = hit_rate(calls[(calls["kind"] == "same_turn") & (calls["idle"] <= CACHE_TTL)])
ref_new = hit_rate(calls[(calls["kind"] == "new_turn") & (calls["idle"] <= CACHE_TTL)])
sw = calls[(calls["kind"] == "switch") & (calls["idle"] <= CACHE_TTL)]
sw_ref = np.where(sw["turn"] == sw["prev_turn"], ref_same, ref_new)
switch_extra = (sw_ref * sw["prompt"] - sw["cached"]).clip(lower=0) * saving_per_token

n_sessions_switch = calls.loc[calls["kind"] == "switch", "session_id"].nunique()
print(f"sessions with a model switch: {n_sessions_switch / calls['session_id'].nunique():.1%}")
print(f"hit rate after switch: {hit_rate(calls[calls['kind'] == 'switch']):.1%}")
print(f"idle > 5 min on short cache models: {len(late)} turns, extra ${late_extra.sum():,.0f} "
      f"= {late_extra.sum() / total_cost:.1%} of total cost")
print(f"model switches: {len(sw)} calls, extra ${switch_extra.sum():,.0f} = {switch_extra.sum() / total_cost:.1%} of total cost")

summary = pd.DataFrame({
    "cause": ["User idle > 5 min", "Model switch"],
    "events": [len(late), len(sw)],
    "extra_cost": [late_extra.sum(), switch_extra.sum()],
})
summary["share_of_bill"] = summary["extra_cost"] / total_cost
summary.to_csv("results/cache_loss_causes.csv", index=False)

fig, ax = plt.subplots(figsize=(8, 2.6))
vals = summary["share_of_bill"].values * 100
ax.barh(summary["cause"][::-1], vals[::-1], color=[GRAY, RED])
for i, (v, e) in enumerate(zip(vals[::-1], summary["extra_cost"].values[::-1])):
    ax.text(v + 0.1, i, f"{v:.1f}% of the bill (${e:,.0f})", va="center")
ax.set_xlim(0, max(vals) * 1.6)
ax.set_xlabel("Extra cost compared to a warm cache, % of the total bill")
ax.set_title("A coffee break costs more than all model switches together")
plt.tight_layout()
plt.savefig("images/idle_vs_switch.png")
plt.close()

# extra tokens per session, used in 07_robustness.py to recount with other prices
calls["idle_extra_tokens"] = 0.0
calls.loc[late.index, "idle_extra_tokens"] = (late["model"].map(warm_rate) * late["prompt"] - late["cached"]).clip(lower=0)
calls["switch_extra_tokens"] = 0.0
calls.loc[sw.index, "switch_extra_tokens"] = (sw_ref * sw["prompt"] - sw["cached"]).clip(lower=0)
per_session = calls.groupby("session_id").agg(
    date=("date", "first"),
    uncached=("uncached", "sum"),
    cached=("cached", "sum"),
    completion=("completion", "sum"),
    idle_extra_tokens=("idle_extra_tokens", "sum"),
    switch_extra_tokens=("switch_extra_tokens", "sum"),
).reset_index()
per_session.to_parquet("data/sessions_cache.parquet", index=False)
