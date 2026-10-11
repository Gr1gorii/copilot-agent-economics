import pandas as pd
import matplotlib.pyplot as plt

# $ per 1M tokens. models in the dataset are anonymized, so I use one reference
# price list (typical frontier model in 2026: cached input = 10% of input price)
PRICE_INPUT = 3.00
PRICE_CACHED = 0.30
PRICE_OUTPUT = 15.00

CACHE_TTL = 300  # seconds

BLUE = "#2b6cb0"
RED = "#c53030"
GRAY = "#a0aec0"
DARK = "#2d3748"

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titleweight": "bold",
    "axes.titlesize": 12,
    "axes.titlelocation": "left",
})


def load_calls():
    calls = pd.read_parquet("data/calls")
    # "tool-model" rows have no token data
    calls = calls[calls["prompt"].notna() & (calls["model"] != "tool-model")].copy()
    calls = calls.sort_values(["session_id", "call"]).reset_index(drop=True)
    calls["cached"] = calls["cached"].fillna(0)
    calls["uncached"] = calls["prompt"] - calls["cached"]
    calls["cost"] = (calls["uncached"] * PRICE_INPUT + calls["cached"] * PRICE_CACHED
                     + calls["completion"] * PRICE_OUTPUT) / 1e6
    return calls


def add_prev_call(calls):
    g = calls.groupby("session_id")
    calls["prev_model"] = g["model"].shift()
    calls["prev_turn"] = g["turn"].shift()
    prev_end = g["ts"].shift() + pd.to_timedelta(g["duration_ms"].shift(), unit="ms")
    calls["idle"] = (calls["ts"] - prev_end).dt.total_seconds()

    calls["kind"] = "same_turn"
    calls.loc[calls["turn"] != calls["prev_turn"], "kind"] = "new_turn"
    calls.loc[calls["model"] != calls["prev_model"], "kind"] = "switch"
    calls.loc[calls["prev_model"].isna(), "kind"] = "first"
    return calls
