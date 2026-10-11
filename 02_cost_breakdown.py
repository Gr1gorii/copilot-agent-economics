import numpy as np
import matplotlib.pyplot as plt

from config import *

calls = load_calls()
print("sessions:", calls["session_id"].nunique())
print("llm calls:", len(calls))

# 1. tokens vs money
tokens = {
    "Input, cache miss": calls["uncached"].sum(),
    "Input, cache hit": calls["cached"].sum(),
    "Output": calls["completion"].sum(),
}
money = {
    "Input, cache miss": calls["uncached"].sum() * PRICE_INPUT / 1e6,
    "Input, cache hit": calls["cached"].sum() * PRICE_CACHED / 1e6,
    "Output": calls["completion"].sum() * PRICE_OUTPUT / 1e6,
}
total_tokens = sum(tokens.values())
total_cost = sum(money.values())
print(f"total tokens: {total_tokens / 1e9:.1f}B, total cost: ${total_cost:,.0f}")
for k in tokens:
    print(f"{k}: {tokens[k] / total_tokens:.1%} of tokens, {money[k] / total_cost:.1%} of cost")

fig, ax = plt.subplots(figsize=(9, 2.8))
colors = [RED, BLUE, DARK]
for i, (name, d, tot) in enumerate([("Tokens", tokens, total_tokens), ("Cost", money, total_cost)]):
    left = 0
    for (k, v), c in zip(d.items(), colors):
        share = v / tot * 100
        ax.barh(i, share, left=left, color=c, label=k if i == 0 else None, height=0.6)
        if share > 4:
            ax.text(left + share / 2, i, f"{share:.0f}%", ha="center", va="center", color="white", fontweight="bold")
        left += share
ax.text(100.5, 0, "<1%", va="center", color=DARK)
ax.set_yticks([0, 1])
ax.set_yticklabels(["Share of tokens", "Share of cost"])
ax.invert_yaxis()
ax.set_xlim(0, 100)
ax.set_xticks([])
ax.spines["bottom"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.02), frameon=False)
ax.set_title("14% of tokens miss the cache, and they make up more than half of the bill")
plt.tight_layout()
plt.savefig("images/tokens_vs_cost.png")
plt.close()

# 2. what the prompt is made of
roles = calls[["tool_tokens", "user_tokens", "assistant_tokens", "system_tokens"]].sum()
roles = roles / roles.sum() * 100
labels = ["Tool outputs (files, terminal, search)", "User messages and IDE context", "Agent's earlier messages",
          "System prompt and instructions"]
print(dict(zip(labels, roles.round(1))))

fig, ax = plt.subplots(figsize=(8, 2.8))
ax.barh(labels[::-1], roles.values[::-1], color=[GRAY, GRAY, GRAY, BLUE])
for i, v in enumerate(roles.values[::-1]):
    ax.text(v + 0.8, i, f"{v:.0f}%", va="center")
ax.set_xlim(0, 60)
ax.set_xlabel("% of all input tokens")
ax.set_title("Half of the context is output of the agent's own tools")
plt.tight_layout()
plt.savefig("images/context_composition.png")
plt.close()

# 3. cost per session
sess = calls.groupby("session_id")["cost"].sum().sort_values(ascending=False)
print(sess.describe(percentiles=[0.5, 0.9, 0.99]))
cum = sess.cumsum() / sess.sum() * 100
share_x = np.arange(1, len(sess) + 1) / len(sess) * 100
for q in [1, 5, 10, 20]:
    print(f"top {q}% of sessions -> {cum.iloc[int(len(sess) * q / 100) - 1]:.1f}% of cost")

fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
ax = axes[0]
bins = np.logspace(np.log10(max(sess.min(), 1e-4)), np.log10(sess.max()), 60)
ax.hist(sess, bins=bins, color=BLUE)
ax.set_xscale("log")
ax.set_xlim(1e-3, 1e3)
ax.set_xticks([0.001, 0.01, 0.1, 1, 10, 100, 1000])
ax.set_xticklabels(["$0.001", "$0.01", "$0.1", "$1", "$10", "$100", "$1000"])
ax.axvline(sess.median(), color=DARK, linestyle="--", linewidth=1)
ax.text(sess.median() * 1.2, ax.get_ylim()[1] * 0.9, f"median ${sess.median():.2f}", color=DARK)
ax.set_xlabel("Cost of one session (log scale)")
ax.set_ylabel("Sessions")
ax.set_title("Cost per session")

ax = axes[1]
ax.plot(share_x, cum.values, color=BLUE, linewidth=2)
ax.plot([0, 100], [0, 100], color=GRAY, linestyle="--", linewidth=1)
for q in [1, 10]:
    y = cum.iloc[int(len(sess) * q / 100) - 1]
    ax.scatter(q, y, color=RED, zorder=3)
    ax.text(q + 6, y - 7, f"top {q}% = {y:.0f}% of cost", color=RED)
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.set_xlabel("% of sessions (most expensive first)")
ax.set_ylabel("% of total cost")
ax.set_title("A few sessions take most of the money")
plt.tight_layout()
plt.savefig("images/session_cost.png")
plt.close()
