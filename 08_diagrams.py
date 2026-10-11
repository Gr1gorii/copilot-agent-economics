import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Patch

from config import *

# schemes for the README, one version in english and one in italian
LIGHT_RED = "#fed7d7"
LIGHT_BLUE = "#ebf4ff"
LIGHT_GRAY = "#f7fafc"

T = {
    "en": {
        "a_title": "How a session spends money (schematic)",
        "a_cache": "Read from cache (10% of the price)",
        "a_full": "Not in cache (full price)",
        "a_turn": "Turn",
        "a_first": "First call:\nnothing cached yet",
        "a_loop": "Each call re-reads the whole context.\nOnly the new part is paid in full",
        "a_short": "User away 3 min\ncache still alive",
        "a_long": "User away 8 min\ncache expired",
        "a_again": "Whole context\npaid again",
        "a_x": "Model calls in one session",
        "a_y": "Context size",

        "b_title": "How the project works",
        "b_raw": ("GitHub release", "7 archives, 3.2 GB\nJSON traces"),
        "b_01": ("01_build_tables.py", "one row per LLM call\nand per tool call"),
        "b_data": ("Parquet tables", "data/calls  9.3M rows\ndata/tools  8.7M rows"),
        "b_02": ("02_cost_breakdown.py", "Where does the\nmoney go?", "53% of the bill\n= cache misses"),
        "b_03": ("03_cache_cliff.py", "What does a user\nbreak cost?", "break > 5 min\n= 5.9% of the bill"),
        "b_04": ("04_keep_warm.py", "Fix 1: keep the\ncache warm", "ML model: +0.3%\nnot worth it"),
        "b_05": ("05_prune_when_cold.py", "Fix 2: clean the\ncontext on cold\nrestart", "+7.7% of the bill"),
        "b_06": ("06_ab_test_design.py", "How big must\nthe A/B test be?", "9k sessions per group,\nguardrails 155k"),
        "b_07": "07_robustness.py    confidence intervals, other prices, every day",
        "b_links": "Arrows between scripts: 03 passes the list of models with a 5 min cache, 04 passes fix 1 results, "
                   "05 passes a table of sessions",
        "b_out": "images/  charts          results/  csv tables and printed numbers",
        "b_config": "config.py: prices, chart style and data loading, used by every script",

        "c_title": "Fix 1: keep the cache warm with pings",
        "c_rows": ["No pings", "Pings, user\ncomes back", "Pings, user\nnever comes back"],
        "c_answer": "agent\nanswers",
        "c_alive": "cache alive",
        "c_ping": "ping",
        "c_back_cold": "user back:\nfull price",
        "c_back_warm": "user back:\ncache hit, saved",
        "c_wasted": "pings paid\nfor nothing",
        "c_x": "Minutes after the agent's answer",
        "c_note": "One ping costs about 1/6 of what a return saves. After 8 min of silence only 12% of users "
                  "come back in the next 4 min,\nso the model has to guess who comes back. It guesses badly (ROC AUC 0.63).",

        "d_title": "Fix 2: clean the context when the cache is already cold",
        "d_rows": ["Without the rule", "With the rule"],
        "d_parts": ["System", "User", "Agent", "Tool outputs,\nolder turns", "Tool outputs,\nlast turn"],
        "d_dropped": "dropped",
        "d_note": "When: the user was away > 5 min on a model with a 5 min cache. The cache is already gone, so the request "
                  "is paid in full anyway\nand nothing cached is lost. The dropped part stays out until the end of the "
                  "session, so every next call is cheaper too.",
    },
    "it": {
        "a_title": "Come una sessione spende soldi (schema)",
        "a_cache": "Letto dalla cache (10% del prezzo)",
        "a_full": "Non in cache (prezzo pieno)",
        "a_turn": "Turno",
        "a_first": "Prima chiamata:\nniente in cache",
        "a_loop": "Ogni chiamata rilegge tutto il contesto.\nSolo la parte nuova si paga per intero",
        "a_short": "Utente via 3 min\ncache ancora viva",
        "a_long": "Utente via 8 min\ncache scaduta",
        "a_again": "Tutto il contesto\npagato di nuovo",
        "a_x": "Chiamate al modello in una sessione",
        "a_y": "Dimensione del contesto",

        "b_title": "Come funziona il progetto",
        "b_raw": ("Release GitHub", "7 archivi, 3,2 GB\ntracce JSON"),
        "b_01": ("01_build_tables.py", "una riga per chiamata LLM\ne per chiamata a un tool"),
        "b_data": ("Tabelle parquet", "data/calls  9,3 mln righe\ndata/tools  8,7 mln righe"),
        "b_02": ("02_cost_breakdown.py", "Dove vanno i soldi?", "53% della spesa\n= cache miss"),
        "b_03": ("03_cache_cliff.py", "Quanto costa\nuna pausa?", "pausa > 5 min\n= 5,9% della spesa"),
        "b_04": ("04_keep_warm.py", "Soluzione 1: tenere\nla cache calda", "modello ML: +0,3%\nnon conviene"),
        "b_05": ("05_prune_when_cold.py", "Soluzione 2: pulire\nil contesto\na freddo", "+7,7% della spesa"),
        "b_06": ("06_ab_test_design.py", "Quanto deve essere\ngrande il test A/B?", "9 mila per gruppo,\nguardrail 155 mila"),
        "b_07": "07_robustness.py    intervalli di confidenza, altri prezzi, ogni giorno",
        "b_links": "Frecce tra gli script: 03 passa la lista dei modelli con cache di 5 min, 04 passa i risultati della "
                   "soluzione 1, 05 passa una tabella delle sessioni",
        "b_out": "images/  grafici          results/  tabelle csv e numeri stampati",
        "b_config": "config.py: prezzi, stile dei grafici e caricamento dei dati, usato da tutti gli script",

        "c_title": "Soluzione 1: tenere la cache calda con i ping",
        "c_rows": ["Senza ping", "Ping, l'utente\ntorna", "Ping, l'utente\nnon torna"],
        "c_answer": "risposta\nagente",
        "c_alive": "cache viva",
        "c_ping": "ping",
        "c_back_cold": "l'utente torna:\nprezzo pieno",
        "c_back_warm": "l'utente torna:\ncache hit, risparmio",
        "c_wasted": "ping pagati\nper niente",
        "c_x": "Minuti dopo la risposta dell'agente",
        "c_note": "Un ping costa circa 1/6 di quello che si risparmia se l'utente torna. Dopo 8 min di silenzio solo il 12% "
                  "degli utenti torna nei 4 min successivi,\nquindi il modello deve indovinare chi torna. Indovina male (ROC AUC 0,63).",

        "d_title": "Soluzione 2: pulire il contesto quando la cache è già fredda",
        "d_rows": ["Senza la regola", "Con la regola"],
        "d_parts": ["System", "Utente", "Agente", "Output dei tool,\nturni vecchi", "Output dei tool,\nultimo turno"],
        "d_dropped": "eliminato",
        "d_note": "Quando: l'utente è stato via > 5 min su un modello con cache di 5 min. La cache non c'è più, quindi la "
                  "richiesta si paga comunque per intero\ne non si perde niente. La parte eliminata resta fuori fino alla "
                  "fine della sessione, quindi costano meno anche tutte le chiamate successive.",
    },
}


def box(ax, x, y, w, h, title, body="", result="", color=LIGHT_GRAY, edge=GRAY):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.12",
                                facecolor=color, edgecolor=edge, linewidth=1.2))
    cx = x + w / 2
    if not body and not result:
        ax.text(cx, y + h / 2, title, ha="center", va="center", fontsize=9.5, fontweight="bold", color=DARK)
        return
    ax.text(cx, y + h - 0.22, title, ha="center", va="top", fontsize=9.5, fontweight="bold", color=DARK)
    ax.text(cx, y + h - 0.55, body, ha="center", va="top", fontsize=9, color=DARK, linespacing=1.3)
    if result:
        ax.text(cx, y + 0.2, result, ha="center", va="bottom", fontsize=9, color=BLUE, fontweight="bold",
                linespacing=1.3)


def arrow(ax, x1, y1, x2, y2, text="", color=DARK):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.2, shrinkA=0, shrinkB=0))
    if text:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.12, text, ha="center", va="bottom", fontsize=8, color="#4a5568")


def session_scheme(lang):
    t = T[lang]
    # schematic numbers, not real data
    sizes = [20, 24, 29, 33, 37, 41, 45, 49, 53, 57]
    new = [20, 4, 5, 4, 9, 4, 4, 49, 4, 4]
    xs = [0, 1, 2, 3, 5.2, 6.2, 7.2, 10.2, 11.2, 12.2]

    fig, ax = plt.subplots(figsize=(10, 4.4))
    ax.axvspan(3.6, 4.6, color=LIGHT_BLUE)
    ax.axvspan(7.8, 9.6, color=LIGHT_RED)
    ax.text(4.1, 70, t["a_short"], ha="center", va="top", fontsize=8.5, color=BLUE)
    ax.text(8.7, 70, t["a_long"], ha="center", va="top", fontsize=8.5, color=RED)

    for x, s, n in zip(xs, sizes, new):
        ax.bar(x, s - n, color=BLUE, width=0.6)
        ax.bar(x, n, bottom=s - n, color=RED, width=0.6)

    ax.annotate(t["a_first"], xy=(0, 21), xytext=(-0.4, 40), ha="left", fontsize=8.5, color=DARK,
                arrowprops=dict(arrowstyle="-", color=GRAY))
    ax.text(2, 52, t["a_loop"], ha="center", va="bottom", fontsize=8.5, color=DARK)
    ax.annotate(t["a_again"], xy=(10.2, 50), xytext=(11.2, 66), ha="center", fontsize=8.5, color=RED,
                fontweight="bold", arrowprops=dict(arrowstyle="-", color=RED))

    ax.set_xlim(-0.8, 13)
    ax.set_ylim(0, 76)
    ax.set_xticks([1.5, 6.2, 11.2])
    ax.set_xticklabels([f"{t['a_turn']} {i}" for i in [1, 2, 3]])
    ax.tick_params(axis="x", length=0, pad=6)
    ax.set_yticks([])
    ax.set_xlabel(t["a_x"], labelpad=8)
    ax.set_ylabel(t["a_y"])
    ax.set_title(t["a_title"])
    ax.legend(handles=[Patch(color=BLUE, label=t["a_cache"]), Patch(color=RED, label=t["a_full"])],
              frameon=False, loc="upper left", fontsize=8.5)
    plt.tight_layout()
    plt.savefig(f"images/scheme_session_{lang}.png")
    plt.close()


def pipeline_scheme(lang):
    t = T[lang]
    fig, ax = plt.subplots(figsize=(11, 6.9))
    ax.set_xlim(0, 11)
    ax.set_ylim(-0.15, 7.4)
    ax.axis("off")
    ax.set_title(t["b_title"])

    box(ax, 0.2, 5.8, 2.8, 1.3, t["b_raw"][0], t["b_raw"][1])
    box(ax, 4.1, 5.8, 2.8, 1.3, t["b_01"][0], t["b_01"][1], color=LIGHT_BLUE, edge=BLUE)
    box(ax, 8.0, 5.8, 2.8, 1.3, t["b_data"][0], t["b_data"][1])
    arrow(ax, 3.0, 6.45, 4.1, 6.45)
    arrow(ax, 6.9, 6.45, 8.0, 6.45)

    w = 1.98
    xs = [0.2 + i * (w + 0.175) for i in range(5)]
    for x, key in zip(xs, ["b_02", "b_03", "b_04", "b_05", "b_06"]):
        title, body, result = t[key]
        box(ax, x, 2.55, w, 2.2, title, body, result, color=LIGHT_BLUE, edge=BLUE)

    # tables go to every analysis script
    ax.plot([xs[4] + w / 2, xs[4] + w / 2], [5.8, 5.25], color=DARK, lw=1.2)
    ax.plot([xs[0] + w / 2, xs[4] + w / 2], [5.25, 5.25], color=DARK, lw=1.2)
    for x in xs:
        arrow(ax, x + w / 2, 5.25, x + w / 2, 4.75)
    for i in [1, 2, 3]:
        arrow(ax, xs[i] + w, 3.65, xs[i + 1], 3.65)

    # 07 checks the numbers of 03, 04 and 05
    box(ax, xs[1], 1.45, xs[3] + w - xs[1], 0.55, t["b_07"], color=LIGHT_BLUE, edge=BLUE)
    for x in xs[1:4]:
        arrow(ax, x + w / 2, 2.55, x + w / 2, 2.0)
    arrow(ax, xs[2] + w / 2, 1.45, xs[2] + w / 2, 1.1)
    for x in [xs[0], xs[4]]:
        arrow(ax, x + w / 2, 2.55, x + w / 2, 1.1)

    box(ax, 0.2, 0.5, 10.6, 0.6, t["b_out"])
    ax.text(5.5, 0.12, t["b_links"] + "\n" + t["b_config"], ha="center", va="center", fontsize=8.5,
            color="#4a5568", linespacing=1.5)
    plt.tight_layout()
    plt.savefig(f"images/scheme_pipeline_{lang}.png")
    plt.close()


def keep_warm_scheme(lang):
    t = T[lang]
    fig, ax = plt.subplots(figsize=(10, 4.4))
    rows = [2.4, 1.4, 0.4]
    ax.set_xlim(-2.6, 16.5)
    ax.set_ylim(-0.1, 3.2)
    for y, name in zip(rows, t["c_rows"]):
        ax.text(-0.4, y, name, ha="right", va="center", fontsize=9, color=DARK)
        ax.plot([0, 16], [y, y], color="#e2e8f0", lw=1, zorder=0)
        ax.scatter(0, y, color=DARK, s=30, zorder=3)

    ax.text(0, 2.85, t["c_answer"], ha="center", va="bottom", fontsize=8, color=DARK)

    # row 1: no pings, cache dies at 5 min, user back at 10
    ax.barh(rows[0], 5, left=0, height=0.28, color=BLUE, alpha=0.35)
    ax.text(2.5, rows[0], t["c_alive"], ha="center", va="center", fontsize=8, color=BLUE)
    ax.scatter(10, rows[0], color=RED, s=60, zorder=3)
    ax.text(10.4, rows[0], t["c_back_cold"], ha="left", va="center", fontsize=8.5, color=RED)

    # row 2: pings at 4 and 8, cache alive until 12, user back at 10
    ax.barh(rows[1], 10, left=0, height=0.28, color=BLUE, alpha=0.35)
    for p in [4, 8]:
        ax.scatter(p, rows[1], marker="v", color=DARK, s=45, zorder=3)
        ax.text(p, rows[1] + 0.2, t["c_ping"], ha="center", va="bottom", fontsize=8, color=DARK)
    ax.scatter(10, rows[1], color=BLUE, s=60, zorder=3)
    ax.text(10.4, rows[1], t["c_back_warm"], ha="left", va="center", fontsize=8.5, color=BLUE)

    # row 3: pings but the user never comes back
    ax.barh(rows[2], 12, left=0, height=0.28, color=GRAY, alpha=0.35)
    for p in [4, 8]:
        ax.scatter(p, rows[2], marker="v", color=DARK, s=45, zorder=3)
        ax.text(p, rows[2] + 0.2, t["c_ping"], ha="center", va="bottom", fontsize=8, color=DARK)
    ax.text(12.4, rows[2], t["c_wasted"], ha="left", va="center", fontsize=8.5, color=RED)

    ax.set_yticks([])
    ax.set_xticks(range(0, 17, 2))
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_bounds(0, 16)
    ax.set_xlabel(t["c_x"])
    ax.xaxis.set_label_coords(0.57, -0.12)
    ax.set_title(t["c_title"])
    fig.text(0.5, 0.02, t["c_note"], ha="center", va="bottom", fontsize=8.5, color="#4a5568")
    plt.tight_layout(rect=(0, 0.1, 1, 1))
    plt.savefig(f"images/scheme_keep_warm_{lang}.png")
    plt.close()


def prune_scheme(lang):
    t = T[lang]
    sizes = [6, 9, 9, 15, 6]
    colors = ["#cbd5e0", "#cbd5e0", "#cbd5e0", RED, BLUE]

    fig, ax = plt.subplots(figsize=(10, 3.4))
    for row, y in enumerate([1, 0]):
        left = 0
        for i, (s, c) in enumerate(zip(sizes, colors)):
            if row == 1 and i == 3:
                continue
            ax.barh(y, s, left=left, height=0.55, color=c, edgecolor="white", linewidth=2)
            ax.text(left + s / 2, y, t["d_parts"][i], ha="center", va="center", fontsize=8.5,
                    color="white" if c in [RED, BLUE] else DARK)
            left += s
        ax.text(-0.5, y, t["d_rows"][row], ha="right", va="center", fontsize=9.5, color=DARK, fontweight="bold")

    x_drop = sum(sizes[:3]) + sizes[3] / 2
    ax.annotate(t["d_dropped"], xy=(x_drop, 0.68), xytext=(x_drop, 0.38), ha="center", va="center",
                fontsize=9, color=RED, fontweight="bold", arrowprops=dict(arrowstyle="<|-", color=RED))

    ax.set_xlim(-9, 46)
    ax.set_ylim(-0.5, 1.5)
    ax.axis("off")
    ax.set_title(t["d_title"], loc="left")
    fig.text(0.5, 0.03, t["d_note"], ha="center", va="bottom", fontsize=8.5, color="#4a5568")
    plt.tight_layout(rect=(0, 0.15, 1, 1))
    plt.savefig(f"images/scheme_prune_{lang}.png")
    plt.close()


for lang in ["en", "it"]:
    session_scheme(lang)
    pipeline_scheme(lang)
    keep_warm_scheme(lang)
    prune_scheme(lang)
