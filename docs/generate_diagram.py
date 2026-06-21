"""
Generates docs/architecture.png — run once, then commit the PNG.
Usage:  python docs/generate_diagram.py
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

# ── Canvas ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(24, 16))
ax.set_xlim(0, 24)
ax.set_ylim(0, 16)
ax.axis("off")
fig.patch.set_facecolor("#0d1117")
ax.set_facecolor("#0d1117")

# ── Colour palette ────────────────────────────────────────────────────────────
C_BG       = "#0d1117"
C_PANEL    = "#161b22"
C_BORDER   = "#30363d"
C_USER     = "#1f6feb"
C_COORD    = "#388bfd"
C_NODE1    = "#d29922"
C_NODE2    = "#3fb950"
C_NODE3    = "#d29922"
C_NODE4    = "#58a6ff"
C_NODE5    = "#a371f7"
C_RAG      = "#3fb950"
C_CALC     = "#bc8cff"
C_SEARCH   = "#ff7b72"
C_MULTI    = "#f78166"
C_DUCK     = "#58a6ff"
C_GROQ     = "#a371f7"
C_AGENTUI  = "#1f6feb"
C_TEXT     = "#e6edf3"
C_SUBTEXT  = "#8b949e"
C_ARROW    = "#484f58"
C_TRACKER  = "#ffa657"
C_GRA      = "#a371f7"


def box(ax, x, y, w, h, label, sublabel="", color=C_PANEL, border=C_BORDER,
        fontsize=9.5, subfontsize=7.5, radius=0.22, text_color=C_TEXT):
    rect = FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        linewidth=1.5, edgecolor=border, facecolor=color, zorder=3
    )
    ax.add_patch(rect)
    dy = 0.14 if sublabel else 0
    ax.text(x, y + dy, label, ha="center", va="center",
            fontsize=fontsize, fontweight="bold", color=text_color, zorder=4)
    if sublabel:
        ax.text(x, y - 0.3, sublabel, ha="center", va="center",
                fontsize=subfontsize, color=C_SUBTEXT, zorder=4)


def arrow(ax, x1, y1, x2, y2, color=C_ARROW, lw=1.5, label=""):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                connectionstyle="arc3,rad=0.0"),
                zorder=2)
    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mx + 0.1, my, label, fontsize=7, color=C_SUBTEXT,
                ha="left", va="center", zorder=5)


def slab(ax, x0, y0, w, h, border="#2d333b"):
    ax.add_patch(FancyBboxPatch(
        (x0, y0), w, h,
        boxstyle="round,pad=0,rounding_size=0.2",
        linewidth=1, edgecolor=border, facecolor=C_PANEL, zorder=1
    ))


def section_label(ax, x, y, text):
    ax.text(x, y, text, fontsize=7.5, color=C_SUBTEXT,
            ha="left", va="center", style="italic", zorder=5)


# ── Title ─────────────────────────────────────────────────────────────────────
ax.text(12, 15.55, "AI Research Assistant — Project Delphi", ha="center",
        va="center", fontsize=17, fontweight="bold", color=C_TEXT, zorder=5)
ax.text(12, 15.1,
        "Five-Node Atomic Pipeline  |  Five Agents  |  Ollama phi3.5 (local)  |  "
        "Groq qwen3-32b  |  DuckDB  |  Agno AgentOS",
        ha="center", va="center", fontsize=9, color=C_SUBTEXT, zorder=5)

# ── Row 1: User + AgentOS UI ──────────────────────────────────────────────────
box(ax, 2.5, 14.1, 3.0, 0.8, "User Query",
    "natural language", color="#1c2128", border=C_USER, fontsize=10)
box(ax, 21.5, 14.1, 3.0, 0.8, "Agno AgentOS UI",
    "os.agno.com -> :7777", color="#1c2128", border=C_AGENTUI, fontsize=10)

arrow(ax, 4.0, 14.1, 6.5, 14.1, color=C_USER, lw=2, label="query")
arrow(ax, 17.5, 14.1, 20.0, 14.1, color=C_AGENTUI, lw=2, label="final answer")

# ── Coordinator shell ─────────────────────────────────────────────────────────
slab(ax, 5.8, 13.4, 11.4, 1.1)
section_label(ax, 5.95, 14.35, "COORDINATOR SHELL  (Agno Agent  |  Groq gpt-oss-20b  |  JSON tool calls)")
box(ax, 11.5, 13.9, 7.5, 0.75, "route_query() tool",
    "calls run_coordinator() — returns RAW TOOL RESULT + SYNTHESIZED ANSWER",
    color="#1c2128", border=C_COORD, fontsize=9.5)
arrow(ax, 7.5, 14.1, 8.0, 13.9, color=C_BORDER, lw=1)
arrow(ax, 8.0, 13.9, 8.5, 13.9, color=C_COORD, lw=1.5)

# ── Five-Node Pipeline row ────────────────────────────────────────────────────
slab(ax, 0.4, 9.1, 23.2, 3.5)
section_label(ax, 0.55, 12.45, "FIVE-NODE PIPELINE")

NODES = [
    (2.5,  10.9, "Node 1", "Intent Classifier", "Ollama phi3.5\nkeep_alive=5m", C_NODE1),
    (6.8,  10.9, "Node 2", "Similarity Safety Net", "Python  |  DuckDB\ntop_k=1 check", C_NODE2),
    (11.1, 10.9, "Node 3", "Query Rewriter", "Ollama phi3.5\nalready hot", C_NODE3),
    (15.4, 10.9, "Node 4", "Tool Execution", "Python  |  zero LLM\ndirect dispatch", C_NODE4),
    (19.9, 10.9, "Node 5", "General Reasoning Agent", "Groq qwen3-32b\n6-step synthesis", C_NODE5),
]

for x, y, num, title, sub, color in NODES:
    ax.text(x, y + 1.05, num, ha="center", va="center",
            fontsize=7.5, color=color, fontweight="bold", zorder=5)
    box(ax, x, y, 3.6, 1.8, title, sub, color="#1c2128", border=color,
        fontsize=9.5, subfontsize=8.0)

# Classification labels inside Node 1
for i, (lbl, col) in enumerate([("RAG", C_RAG), ("CALC", C_CALC),
                                  ("SEARCH", C_SEARCH), ("MULTI", C_MULTI)]):
    ax.text(1.15 + i * 0.69, 10.25, lbl, ha="center", va="center",
            fontsize=6.5, fontweight="bold", color=col, zorder=5)

# Node 2 override annotation
ax.text(6.8, 9.85, "similarity > 0.25\n-> override to RAG",
        ha="center", va="center", fontsize=7, color=C_NODE2,
        style="italic", zorder=5)

# Arrows between nodes
arrow(ax, 4.3, 10.9,  5.0, 10.9,  color=C_NODE1, lw=1.8)
arrow(ax, 8.6, 10.9,  9.3, 10.9,  color=C_NODE2, lw=1.8)
arrow(ax, 12.9, 10.9, 13.6, 10.9, color=C_NODE3, lw=1.8)
arrow(ax, 17.2, 10.9, 17.9, 10.9, color=C_NODE4, lw=1.8)

# ── Agent layer ───────────────────────────────────────────────────────────────
slab(ax, 0.4, 5.5, 23.2, 3.3)
section_label(ax, 0.55, 8.65, "AGENT LAYER  (Agno Agents — Node 4 dispatches to these)")

AGENTS = [
    (4.5,  7.2, "RAG Agent", "Groq qwen3-32b\ndocument_lookup tool\nself-eval + retry", C_RAG),
    (11.1, 7.2, "Tracker Agent", "Groq qwen3-32b\ndoc_lookup + calc + search\nMULTI path only", C_TRACKER),
    (19.9, 7.2, "General Reasoning Agent", "Groq qwen3-32b\nno tools — pure reasoning\n6-step structured output", C_GRA),
]

for x, y, title, sub, color in AGENTS:
    box(ax, x, y, 4.2, 1.9, title, sub, color="#1c2128", border=color,
        fontsize=9.5, subfontsize=8.0)

# Arrows from Node 4 to agents
arrow(ax, 13.8, 10.0,  4.5, 8.15, color=C_RAG,     lw=1.5, label="RAG path")
arrow(ax, 15.4, 10.0, 11.1, 8.15, color=C_TRACKER, lw=1.5, label="MULTI path")
# Node 5 connects to General Reasoning Agent upward (it IS that agent)
arrow(ax, 19.9, 10.0, 19.9, 8.15, color=C_GRA,     lw=1.8, label="Node 5")

# Direct Python paths (not through agent) — CALC and SEARCH labels on Node 4
ax.text(15.4, 9.6, "CALC -> safe_calculate()", ha="center", va="center",
        fontsize=7, color=C_CALC, zorder=5)
ax.text(15.4, 9.35, "SEARCH -> web_search()", ha="center", va="center",
        fontsize=7, color=C_SEARCH, zorder=5)

# Return arrows from agents back up to Node 5 area
arrow(ax, 4.5, 6.25,  17.5, 10.2,  color="#2a4a2a", lw=1.1, label="raw chunks")
arrow(ax, 11.1, 6.25, 18.0, 10.2,  color="#4a3a20", lw=1.1, label="multi-tool result")

# ── Tool functions layer ──────────────────────────────────────────────────────
slab(ax, 0.4, 1.9, 23.2, 3.3)
section_label(ax, 0.55, 5.05, "TOOL FUNCTIONS  (Pure Python — zero LLM)")

TOOLS = [
    (3.5,  3.6, "document_lookup()", "DuckDB HNSW vector search\nreturns top-K chunks + scores", C_RAG),
    (9.5,  3.6, "safe_calculate()", "asteval  |  AST-safe\nno eval() — math only", C_CALC),
    (15.5, 3.6, "web_search()", "Tavily API  |  httpx\nSSL bypass verify=False", C_SEARCH),
    (21.0, 3.6, "DuckDB + vss", "embeddings.db\nHNSW index + SQL fallback", C_DUCK),
]

for x, y, title, sub, color in TOOLS:
    bg = (
        "#0d1f0d" if color == C_RAG else
        "#1a0f3a" if color == C_CALC else
        "#2d0f0c" if color == C_SEARCH else
        "#0d1c2e"
    )
    box(ax, x, y, 3.8, 1.5, title, sub, color=bg, border=color,
        fontsize=9, subfontsize=7.5)

# Arrows from agents down to tool functions
arrow(ax, 4.5, 6.25,  3.5,  4.35, color=C_RAG,    lw=1.5)
arrow(ax, 11.1, 6.25, 9.5,  4.35, color=C_CALC,   lw=1.2)
arrow(ax, 11.1, 6.25, 15.5, 4.35, color=C_SEARCH, lw=1.2)
arrow(ax, 3.5,  2.85, 21.0, 4.35, color=C_DUCK,   lw=1.2, label="reads/writes")

# BAAI embedding note
ax.text(12.0, 2.35,
        "BAAI/bge-small-en-v1.5  |  33M params  |  384-dim  |  fully local  |  "
        "Ingestion: documents/ -> chunker -> embed -> DuckDB upsert",
        ha="center", va="center", fontsize=7.5, color=C_SUBTEXT, zorder=5)

# ── Legend ────────────────────────────────────────────────────────────────────
items = [
    (C_NODE1,   "Node 1: Classify (Ollama)"),
    (C_NODE2,   "Node 2: Safety Net (Python)"),
    (C_NODE3,   "Node 3: Rewrite (Ollama)"),
    (C_NODE4,   "Node 4: Execute (Python)"),
    (C_GRA,     "Node 5: General Reasoning Agent"),
    (C_RAG,     "RAG path"),
    (C_CALC,    "Calculator path"),
    (C_SEARCH,  "Search path"),
    (C_TRACKER, "MULTI / Tracker Agent"),
]
for i, (color, label) in enumerate(items):
    x = 0.5 + i * 2.6
    ax.add_patch(mpatches.Rectangle((x, 0.35), 0.32, 0.22, color=color, zorder=5))
    ax.text(x + 0.42, 0.46, label, fontsize=6.8, color=C_SUBTEXT, va="center", zorder=5)

ax.text(12, 0.1,
        "Project Delphi  |  Phase 1 Baseline  |  Agno 2.6.18  |  "
        "Five-Node Atomic Pipeline  |  Five Agents",
        ha="center", va="center", fontsize=7.5, color="#484f58", zorder=5)

# ── Save ──────────────────────────────────────────────────────────────────────
out = os.path.join(os.path.dirname(__file__), "architecture.png")
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=C_BG)
plt.close(fig)
print(f"Saved: {out}")
