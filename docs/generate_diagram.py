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
fig, ax = plt.subplots(figsize=(22, 14))
ax.set_xlim(0, 22)
ax.set_ylim(0, 14)
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
C_DUCK     = "#58a6ff"
C_GROQ     = "#a371f7"
C_AGENTUI  = "#1f6feb"
C_TEXT     = "#e6edf3"
C_SUBTEXT  = "#8b949e"
C_ARROW    = "#484f58"


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
ax.text(11, 13.5, "AI Research Assistant — Project Delphi", ha="center",
        va="center", fontsize=17, fontweight="bold", color=C_TEXT, zorder=5)
ax.text(11, 13.05, "Five-Node Atomic Pipeline  ·  Agno AgentOS  ·  Ollama phi3.5 (local)  ·  Groq qwen3-32b  ·  DuckDB",
        ha="center", va="center", fontsize=9, color=C_SUBTEXT, zorder=5)

# ── Row 1: User + AgentOS UI ──────────────────────────────────────────────────
box(ax, 2.5, 12.0, 3.0, 0.8, "User Query",
    "natural language", color="#1c2128", border=C_USER, fontsize=10)
box(ax, 19.5, 12.0, 3.0, 0.8, "Agno AgentOS UI",
    "os.agno.com → :7777", color="#1c2128", border=C_AGENTUI, fontsize=10)

arrow(ax, 4.0, 12.0, 6.3, 12.0, color=C_USER, lw=2, label="query")
arrow(ax, 15.7, 12.0, 18.0, 12.0, color=C_AGENTUI, lw=2, label="final answer")

# ── Coordinator shell ─────────────────────────────────────────────────────────
slab(ax, 5.6, 11.3, 10.8, 1.1)
section_label(ax, 5.75, 12.25, "COORDINATOR SHELL  (Agno Agent · Groq gpt-oss-20b · JSON tool calls)")
box(ax, 11.0, 11.75, 7.0, 0.75, "route_query() tool",
    "calls run_coordinator(), returns RAW + SYNTHESIZED sections",
    color="#1c2128", border=C_COORD, fontsize=9.5)
arrow(ax, 7.0, 12.0, 7.0, 12.15, color=C_BORDER, lw=1)
arrow(ax, 7.0, 11.75, 7.5, 11.75, color=C_COORD, lw=1.5)

# ── Five nodes — left-to-right flow ───────────────────────────────────────────
slab(ax, 0.4, 7.9, 21.2, 2.9)
section_label(ax, 0.55, 10.65, "FIVE-NODE PIPELINE")

# Node positions (y=9.5 for label row, boxes at y=9.2)
NODES = [
    (2.3,  9.2, "Node 1", "Intent Classify", "Ollama phi3.5\nkeep_alive=5m", C_NODE1),
    (6.3,  9.2, "Node 2", "Similarity Safety Net", "Python · DuckDB\ntop_k=1 check", C_NODE2),
    (10.3, 9.2, "Node 3", "Query Rewrite", "Ollama phi3.5\nalready hot", C_NODE3),
    (14.3, 9.2, "Node 4", "Tool Execution", "Python · zero LLM\ndirect dispatch", C_NODE4),
    (18.5, 9.2, "Node 5", "Synthesis", "Groq qwen3-32b\nhttpx POST", C_NODE5),
]

for x, y, num, title, sub, color in NODES:
    ax.text(x, y + 0.97, num, ha="center", va="center",
            fontsize=7.5, color=color, fontweight="bold", zorder=5)
    box(ax, x, y, 3.3, 1.7, title, sub, color="#1c2128", border=color,
        fontsize=9.5, subfontsize=8.0)

# Classification labels inside Node 1
for i, (lbl, col) in enumerate([("RAG", C_RAG), ("CALC", C_CALC), ("SEARCH", C_SEARCH)]):
    ax.text(1.25 + i * 0.72, 8.6, lbl, ha="center", va="center",
            fontsize=7, fontweight="bold", color=col, zorder=5)

# Arrows between nodes
arrow(ax, 3.95, 9.2,  4.65, 9.2,  color=C_NODE1, lw=1.8)
arrow(ax, 7.95, 9.2,  8.65, 9.2,  color=C_NODE2, lw=1.8)
arrow(ax, 11.95, 9.2, 12.65, 9.2, color=C_NODE3, lw=1.8)
arrow(ax, 15.95, 9.2, 16.65, 9.2, color=C_NODE4, lw=1.8)

# Node 2 override annotation
ax.text(6.3, 8.2, "similarity > 0.25\n→ override to RAG",
        ha="center", va="center", fontsize=7, color=C_NODE2,
        style="italic", zorder=5)

# ── Tool layer ────────────────────────────────────────────────────────────────
slab(ax, 0.4, 4.7, 21.2, 2.9)
section_label(ax, 0.55, 7.45, "TOOL LAYER  (Node 4 dispatches directly — zero LLM tool calling)")

# Three tool paths
TOOLS = [
    (4.5,  6.2, "search_chunks()", "DuckDB vector search\nHNSW / cosine fallback", C_RAG),
    (11.0, 6.2, "safe_calculate()", "asteval · AST-safe\nno eval()", C_CALC),
    (17.5, 6.2, "web_search()", "Tavily API · httpx\nSSL bypass verify=False", C_SEARCH),
]
for x, y, title, sub, color in TOOLS:
    box(ax, x, y, 3.8, 1.5, title, sub, color="#0d1f0d" if color == C_RAG else
        ("#1a0f3a" if color == C_CALC else "#2d0f0c"),
        border=color, fontsize=9.5, subfontsize=8.0)

# Arrows from Node 4 down to three tools
arrow(ax, 13.0, 8.3,  4.5,  7.0,  color=C_RAG,    lw=1.5, label="RAG path")
arrow(ax, 14.3, 8.3, 11.0,  7.0,  color=C_CALC,   lw=1.5, label="CALC path")
arrow(ax, 15.6, 8.3, 17.5,  7.0,  color=C_SEARCH, lw=1.5, label="SEARCH path")

# Result arrows back up to Node 4 (dashed-style via color hint)
arrow(ax, 4.5,  5.45, 13.0, 8.0,  color="#2a4a2a", lw=1.2, label="raw chunks")
arrow(ax, 11.0, 5.45, 14.0, 8.0,  color="#2a2040", lw=1.2, label="expression/result")
arrow(ax, 17.5, 5.45, 15.4, 8.0,  color="#4a2020", lw=1.2, label="web results")

# ── Storage layer ─────────────────────────────────────────────────────────────
slab(ax, 0.4, 1.9, 10.8, 2.5)
section_label(ax, 0.55, 4.25, "STORAGE & EMBEDDING LAYER")

box(ax, 3.2, 3.35, 3.8, 1.1, "DuckDB + vss",
    "embeddings.db · HNSW index\nSQL cosine fallback",
    color="#0d1c2e", border=C_DUCK, fontsize=9.5, subfontsize=7.5)
box(ax, 8.0, 3.35, 3.8, 1.1, "BAAI/bge-small-en-v1.5",
    "384-dim · 33M params\nfully local · CPU inference",
    color="#0a1a0a", border=C_RAG, fontsize=9, subfontsize=7.5)

arrow(ax, 4.5, 4.7,  3.2, 3.9,  color=C_DUCK, lw=1.5)
arrow(ax, 8.0, 3.9,  4.8, 3.5,  color=C_RAG,  lw=1.2, label="embed query")

# Ingestion pipeline note
ax.text(5.6, 2.35, "Ingestion: documents/ → chunker → BAAI embed → DuckDB upsert",
        ha="center", va="center", fontsize=7.5, color=C_SUBTEXT, zorder=5)

# ── Groq API box ──────────────────────────────────────────────────────────────
slab(ax, 11.6, 1.9, 10.0, 2.5, border="#3a2a5a")
section_label(ax, 11.75, 4.25, "GROQ API  (Node 5 synthesis only)")

box(ax, 14.5, 3.35, 4.0, 1.1, "qwen/qwen3-32b",
    "synthesis only · no tool calling\ndirect httpx POST",
    color="#1a0d2e", border=C_GROQ, fontsize=9.5, subfontsize=7.5)
box(ax, 19.5, 3.35, 3.2, 1.1, "gpt-oss-20b",
    "coordinator shell only\nJSON tool calls reliable",
    color="#1c2128", border=C_COORD, fontsize=9, subfontsize=7.5)

arrow(ax, 18.5, 8.3,  14.5, 3.9,  color=C_GROQ, lw=1.5, label="Node 5 →")
arrow(ax, 14.5, 2.8,  18.5, 8.0,  color="#3a2a5a", lw=1.2, label="synthesis")

# ── Legend ────────────────────────────────────────────────────────────────────
items = [
    (C_NODE1, "Node 1: Classify (Ollama)"),
    (C_NODE2, "Node 2: Safety Net (Python)"),
    (C_NODE3, "Node 3: Rewrite (Ollama hot)"),
    (C_NODE4, "Node 4: Execute (Python)"),
    (C_NODE5, "Node 5: Synthesize (Groq)"),
    (C_RAG,   "RAG path"),
    (C_CALC,  "Calculator path"),
    (C_SEARCH,"Search path"),
]
for i, (color, label) in enumerate(items):
    x = 0.5 + i * 2.73
    ax.add_patch(mpatches.Rectangle((x, 0.35), 0.32, 0.22, color=color, zorder=5))
    ax.text(x + 0.42, 0.46, label, fontsize=7, color=C_SUBTEXT, va="center", zorder=5)

ax.text(11, 0.1, "Project Delphi · Phase 1 Baseline · Agno 2.6.18 · Five-Node Atomic Pipeline",
        ha="center", va="center", fontsize=7.5, color="#484f58", zorder=5)

# ── Save ──────────────────────────────────────────────────────────────────────
out = os.path.join(os.path.dirname(__file__), "architecture.png")
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=C_BG)
plt.close(fig)
print(f"Saved: {out}")
