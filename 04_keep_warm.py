import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from config import *

# idea: when the user goes quiet, send a tiny request every 4 minutes so the cache
# does not expire. each ping costs a cached read of the whole context.
# it only makes sense if the user comes back soon, so I try to predict that.
PING_EVERY = 240
TRAIN_DAYS = ["2026-06-01", "2026-06-02", "2026-06-03", "2026-06-04"]

calls = load_calls()
calls["end"] = calls["ts"] + pd.to_timedelta(calls["duration_ms"], unit="ms")
total_cost_by_day = calls.groupby("date")["cost"].sum()

models = pd.read_csv("results/model_cache_retention.csv")
short_ttl = set(models.loc[~models["keeps_cache"], "model"])
warm_rate = models.set_index("model")["hit_warm"]

turns = calls.groupby(["session_id", "turn"]).agg(
    date=("date", "first"),
    start=("ts", "first"),
    end=("end", "last"),
    model=("model", "last"),
    first_model=("model", "first"),
    n_calls=("call", "size"),
    last_prompt=("prompt", "last"),
    last_completion=("completion", "last"),
    first_prompt=("prompt", "first"),
    first_cached=("cached", "first"),
    turn_cost=("cost", "sum"),
    last_result=("result", "last"),
).reset_index()

tools = pd.read_parquet("data/tools", columns=["session_id", "turn", "ts", "name", "status"])
tools = tools.sort_values("ts")
tools["error"] = tools["status"] == 2
tool_stats = tools.groupby(["session_id", "turn"]).agg(
    n_tools=("name", "size"),
    n_tool_errors=("error", "sum"),
    last_tool=("name", "last"),
).reset_index()
turns = turns.merge(tool_stats, on=["session_id", "turn"], how="left")
turns["n_tools"] = turns["n_tools"].fillna(0)
turns["n_tool_errors"] = turns["n_tool_errors"].fillna(0)
turns["last_tool"] = turns["last_tool"].fillna("none")

turns = turns.sort_values(["session_id", "turn"]).reset_index(drop=True)
g = turns.groupby("session_id")
turns["next_start"] = g["start"].shift(-1)
turns["next_model"] = g["first_model"].shift(-1)
turns["next_prompt"] = g["first_prompt"].shift(-1)
turns["next_cached"] = g["first_cached"].shift(-1)
turns["idle"] = (turns["next_start"] - turns["end"]).dt.total_seconds()  # NaN = user never came back
turns["context"] = turns["last_prompt"] + turns["last_completion"]

# session history up to this turn
turns["turn_no"] = g.cumcount()
turns["prev_idle"] = g["idle"].shift()
turns["prev_long_breaks"] = (turns["prev_idle"] > CACHE_TTL).astype(int).groupby(turns["session_id"]).cumsum()
turns["mean_prev_idle"] = turns["prev_idle"].fillna(0).groupby(turns["session_id"]).cumsum() / turns["turn_no"].replace(0, np.nan)
turns["session_cost_so_far"] = g["turn_cost"].cumsum()
turns["turn_minutes"] = (turns["end"] - turns["start"]).dt.total_seconds() / 60
turns["hour"] = turns["end"].dt.hour

# only turns on short cache models where the user did not come back within 4 minutes
risk = turns[turns["model"].isin(short_ttl) & ~(turns["idle"] <= PING_EVERY)].copy()
print("turns at risk:", len(risk), "| never came back:", risk["idle"].isna().mean().round(3))

saving_per_token = (PRICE_INPUT - PRICE_CACHED) / 1e6
same_model = risk["next_model"] == risk["model"]
risk["save"] = 0.0
risk.loc[same_model, "save"] = ((risk["model"].map(warm_rate) * risk["next_prompt"] - risk["next_cached"])
                                .clip(lower=0) * saving_per_token)[same_model]
risk["ping_cost"] = risk["context"] * PRICE_CACHED / 1e6


def turn_net(df, keep_warm, k):
    back = df["idle"].notna()
    covered = keep_warm & back & (df["idle"] <= PING_EVERY * (k + 1))
    pings = np.where(back, np.minimum(k, np.floor(df["idle"].fillna(0) / PING_EVERY)), k)
    return np.where(covered, df["save"], 0) - keep_warm * pings * df["ping_cost"]


def net_saving(df, keep_warm, k):
    return turn_net(df, keep_warm, k).sum()


features = ["model", "last_tool", "last_result", "turn_no", "n_calls", "n_tools", "n_tool_errors",
            "turn_minutes", "context", "last_completion", "prev_idle", "mean_prev_idle",
            "prev_long_breaks", "session_cost_so_far", "hour"]
X = risk[features].copy()
for c in ["model", "last_tool", "last_result"]:
    top = X[c].value_counts().index[:30]
    X[c] = X[c].where(X[c].isin(top), "other").astype("category")

is_train = risk["date"].isin(TRAIN_DAYS)
train, test = risk[is_train], risk[~is_train]
test_bill = total_cost_by_day[~total_cost_by_day.index.isin(TRAIN_DAYS)].sum()
print("train turns:", len(train), "test turns:", len(test))

results = []
best = None
for k in [1, 2, 3, 4, 6, 8, 12, 15]:
    y = (risk["idle"] <= PING_EVERY * (k + 1)) & (risk["save"] > 0)
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, categorical_features="from_dtype",
                                         random_state=42)
    clf.fit(X[is_train], y[is_train])
    p_train = clf.predict_proba(X[is_train])[:, 1]
    p_test = clf.predict_proba(X[~is_train])[:, 1]
    auc = roc_auc_score(y[~is_train], p_test)

    # threshold is picked on train days only
    best_t, best_v = 1.0, 0
    for t in np.arange(0.05, 0.96, 0.025):
        v = net_saving(train, p_train > t, k)
        if v > best_v:
            best_t, best_v = t, v

    everyone = net_saving(test, np.ones(len(test), dtype=bool), k)
    model = net_saving(test, p_test > best_t, k)
    oracle = net_saving(test, y[~is_train].values, k)
    results.append([k, (k + 1) * PING_EVERY / 60, auc, best_t, everyone, model, oracle])
    print(f"k={k} horizon={(k + 1) * PING_EVERY / 60:.0f}min auc={auc:.3f} t={best_t:.2f} "
          f"everyone={everyone / test_bill:.2%} model={model / test_bill:.2%} oracle={oracle / test_bill:.2%}")
    if best is None or best_v > best[1]:
        best = (k, best_v, clf, best_t)

res = pd.DataFrame(results, columns=["pings", "horizon_min", "auc", "threshold", "ping_everyone", "model", "oracle"])
for c in ["ping_everyone", "model", "oracle"]:
    res[c + "_pct"] = res[c] / test_bill * 100
res["chosen"] = res["pings"] == best[0]
res.to_csv("results/keep_warm_policies.csv", index=False)

k, _, clf, t = best
row = res[res["pings"] == k].iloc[0]
print(f"\nbest on train: {k} pings ({row['horizon_min']:.0f} min), threshold {t:.2f}")
print(f"test days bill: ${test_bill:,.0f}")
print(f"ping everyone: ${row['ping_everyone']:,.0f} ({row['ping_everyone_pct']:.2f}%)")
print(f"model:         ${row['model']:,.0f} ({row['model_pct']:.2f}%)")
print(f"oracle:        ${row['oracle']:,.0f} ({row['oracle_pct']:.2f}%)")

y_test = ((test["idle"] <= PING_EVERY * (k + 1)) & (test["save"] > 0)).values
p_test = clf.predict_proba(X[~is_train])[:, 1]
picked = p_test > t
print(f"pinged {picked.mean():.1%} of quiet turns, {y_test[picked].mean():.1%} of them came back in time "
      f"(base rate {y_test.mean():.1%})")

# result of every policy per session on the test days, used in 07_robustness.py
per_session = test[["session_id"]].copy()
per_session["everyone"] = turn_net(test, np.ones(len(test), dtype=bool), k)
per_session["model"] = turn_net(test, picked, k)
per_session["oracle"] = turn_net(test, y_test, k)
per_session.groupby("session_id").sum().reset_index().to_parquet("data/keep_warm_test_sessions.parquet", index=False)

# savings vs how long we keep the cache warm
fig, ax = plt.subplots(figsize=(9, 4))
ax.axhline(0, color=DARK, linewidth=0.8)
ax.plot(res["horizon_min"], res["ping_everyone_pct"], color=GRAY, marker="o", linewidth=2, label="Keep warm for everyone")
ax.plot(res["horizon_min"], res["model_pct"], color=BLUE, marker="o", linewidth=2, label="Keep warm only if the model says the user will return")
ax.plot(res["horizon_min"], res["oracle_pct"], color=DARK, linestyle="--", linewidth=1, label="Perfect foresight (upper bound)")
ax.set_xlabel("How long the cache is kept warm, minutes")
ax.set_ylabel("Net saving, % of total bill")
ax.set_title("Keeping the cache warm does not pay off, even with a model")
ax.legend(frameon=False, loc="lower left")
plt.tight_layout()
plt.savefig("images/keep_warm_horizon.png")
plt.close()
