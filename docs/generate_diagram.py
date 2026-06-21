"""
Generates docs/architecture.png — run once, then commit the PNG.
Usage:  python docs/generate_diagram.py
"""

import os
import matplotlib
matplotlib.use("Agg")          # headless — no display needed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ── Canvas ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(20, 13))
ax.set_xlim(0, 20)
ax.set_ylim(0, 13)
ax.axis("off")
fig.patch.set_facecolor("#0d1117")
ax.set_facecolor("#0d1117")

# ── Colour palette ────────────────────────────────────────────────────────────
C_BG       = "#0d1117"
C_PANEL    = "#161b22"
C_BORDER   = "#30363d"
C_USER     = "#1f6feb"
C_COORD    = "#388bfd"
C_OLLAMA   = "#d29922"
C_RAG      = "#3fb950"
C_CALC     = "#bc8cff"
C_SEARCH   = "#ff7b72"
C_TOOL     = "#21262d"
C_DUCK     = "#58a6ff"
C_LLM      = "#f0883e"
C_GROQ     = "#a371f7"
C_AGENTUI  = "#1f6feb"
C_TEXT     = "#e6edf3"
C_SUBTEXT  = "#8b949e"
C_ARROW    = "#484f58"
C_ARROW_HI = "#58a6ff"

def box(ax, x, y, w, h, label, sublabel="", color=C_PANEL, border=C_BORDER,
        fontsize=10, subfontsize=8, radius=0.25, text_color=C_TEXT):
    rect = FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        linewidth=1.5, edgecolor=border, facecolor=color, zorder=3
    )
    ax.add_patch(rect)
    dy = 0.13 if sublabel else 0
    ax.text(x, y + dy, label, ha="center", va="center",
            fontsize=fontsize, fontweight="bold", color=text_color, zorder=4)
    if sublabel:
        ax.text(x, y - 0.28, sublabel, ha="center", va="center",
                fontsize=subfontsize, color=C_SUBTEXT, zorder=4)

def arrow(ax, x1, y1, x2, y2, color=C_ARROW, lw=1.5, label="", style="->"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                                connectionstyle="arc3,rad=0.0"),
                zorder=2)
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx + 0.08, my, label, fontsize=7, color=C_SUBTEXT,
                ha="left", va="center", zorder=5)

def section_label(ax, x, y, text):
    ax.text(x, y, text, fontsize=7.5, color=C_SUBTEXT, ha="left",
            va="center", style="italic", zorder=5)

# ── Title ─────────────────────────────────────────────────────────────────────
ax.text(10, 12.4, "AI Research Assistant — Project Delphi", ha="center",
        va="center", fontsize=16, fontweight="bold", color=C_TEXT, zorder=5)
ax.text(10, 12.0, "Phase 1 Architecture · Agno AgentOS · OpenRouter / Groq / Ollama / DuckDB",
        ha="center", va="center", fontsize=9, color=C_SUBTEXT, zorder=5)

# ── Row 1 — User + AgentOS UI ─────────────────────────────────────────────────
box(ax, 3.0, 10.8, 3.2, 0.85, "User", "natural language query", color="#1c2128", border=C_USER, fontsize=11)
box(ax, 17.0, 10.8, 3.2, 0.85, "Agno AgentOS UI", "https://app.agno.com → :7777",
    color="#1c2128", border=C_AGENTUI, fontsize=10)

arrow(ax, 4.6, 10.8, 6.8, 10.8, color=C_USER, lw=2, label="query")
arrow(ax, 13.2, 10.8, 15.4, 10.8, color=C_AGENTUI, lw=2, label="response")

# ── Row 2 — REST API / AgentOS ────────────────────────────────────────────────
box(ax, 10.0, 10.8, 3.8, 0.85,
    "FastAPI Backend",
    "POST /query  ·  GET /health  ·  GET /chunks",
    color=C_PANEL, border=C_BORDER, fontsize=10)

# ── Row 3 — Coordinator ───────────────────────────────────────────────────────
# Background slab
slab1 = FancyBboxPatch((5.6, 8.55), 8.8, 1.9,
    boxstyle="round,pad=0,rounding_size=0.2",
    linewidth=1, edgecolor="#2d333b", facecolor="#161b22", zorder=1)
ax.add_patch(slab1)
section_label(ax, 5.75, 10.3, "COORDINATOR LAYER")

box(ax, 7.5, 9.45, 3.0, 1.0, "Coordinator Agent",
    "run_coordinator()", color="#1c2128", border=C_COORD, fontsize=10)
box(ax, 12.1, 9.45, 3.4, 1.0, "Ollama phi3.5-mini",
    "intent classifier · local · :11434",
    color="#2d2208", border=C_OLLAMA, fontsize=10)

arrow(ax, 10.0, 10.8, 10.0, 10.4, color=C_BORDER, lw=1.5)
arrow(ax, 9.0, 9.45, 10.4, 9.45, color=C_OLLAMA, lw=1.8, label="classify →")
arrow(ax, 10.4, 9.45, 9.0, 9.45, color=C_OLLAMA, lw=1.8)

# classification labels under Ollama box
for i, (lbl, col) in enumerate([("RAG", C_RAG), ("CALCULATOR", C_CALC), ("SEARCH", C_SEARCH)]):
    ax.text(10.8 + i*0.85, 8.75, lbl, ha="center", va="center",
            fontsize=7, fontweight="bold", color=col, zorder=5)

arrow(ax, 10.0, 9.0,  10.0, 10.35, color=C_BORDER, lw=1)

# ── Row 4 — Three agents ──────────────────────────────────────────────────────
slab2 = FancyBboxPatch((0.5, 5.55), 19.0, 2.65,
    boxstyle="round,pad=0,rounding_size=0.2",
    linewidth=1, edgecolor="#2d333b", facecolor="#161b22", zorder=1)
ax.add_patch(slab2)
section_label(ax, 0.65, 8.05, "AGENT LAYER")

# RAG Agent
box(ax, 3.5, 7.3, 3.4, 1.15, "RAG Agent",
    "document knowledge queries", color="#0d2419", border=C_RAG, fontsize=10)
# Calculator Agent
box(ax, 10.0, 7.3, 3.4, 1.15, "Calculator Agent",
    "mathematical expressions", color="#1a0f3a", border=C_CALC, fontsize=10)
# Web Search Agent
box(ax, 16.5, 7.3, 3.4, 1.15, "Web Search Agent",
    "current events / world knowledge", color="#2d0f0c", border=C_SEARCH, fontsize=10)

# Routing arrows from coordinator
arrow(ax, 7.2, 8.9,  3.8, 7.88, color=C_RAG,    lw=1.8, label="RAG")
arrow(ax, 7.5, 8.9, 10.0, 7.88, color=C_CALC,   lw=1.8, label="CALC")
arrow(ax, 7.8, 8.9, 16.0, 7.88, color=C_SEARCH, lw=1.8, label="SEARCH")

# ── Row 5 — Tools ─────────────────────────────────────────────────────────────
slab3 = FancyBboxPatch((0.5, 3.2), 19.0, 2.0,
    boxstyle="round,pad=0,rounding_size=0.2",
    linewidth=1, edgecolor="#2d333b", facecolor="#161b22", zorder=1)
ax.add_patch(slab3)
section_label(ax, 0.65, 5.05, "TOOL LAYER")

box(ax, 3.5, 4.2, 3.2, 1.0, "document_lookup()",
    "DuckDB vss · cosine similarity",
    color=C_TOOL, border=C_RAG, fontsize=9.5)
box(ax, 10.0, 4.2, 3.2, 1.0, "safe_calculate()",
    "asteval · AST interpreter · no eval()",
    color=C_TOOL, border=C_CALC, fontsize=9.5)
box(ax, 16.5, 4.2, 3.2, 1.0, "web_search()",
    "Tavily API · top-3 results + citations",
    color=C_TOOL, border=C_SEARCH, fontsize=9.5)

arrow(ax, 3.5,  6.73, 3.5,  4.7,  color=C_RAG,    lw=1.5)
arrow(ax, 10.0, 6.73, 10.0, 4.7,  color=C_CALC,   lw=1.5)
arrow(ax, 16.5, 6.73, 16.5, 4.7,  color=C_SEARCH, lw=1.5)

# ── Row 6 — Storage ───────────────────────────────────────────────────────────
box(ax, 3.5, 2.55, 3.2, 0.9, "DuckDB + vss",
    "embeddings.db · HNSW index · SQL cosine fallback",
    color="#0d1c2e", border=C_DUCK, fontsize=9)
arrow(ax, 3.5, 3.7, 3.5, 3.0, color=C_DUCK, lw=1.5)

# Embedding model note
box(ax, 3.5, 1.55, 3.8, 0.8, "BAAI/bge-small-en-v1.5",
    "384-dim · 33M params · fully local · CPU",
    color="#0a1a0a", border=C_RAG, fontsize=8.5)
arrow(ax, 3.5, 2.1, 3.5, 1.95, color=C_RAG, lw=1.2)

# ── LLM Gateway (shared by all sub-agents) ────────────────────────────────────
slab4 = FancyBboxPatch((7.0, 1.1), 12.0, 1.95,
    boxstyle="round,pad=0,rounding_size=0.2",
    linewidth=1, edgecolor="#2d333b", facecolor="#161b22", zorder=1)
ax.add_patch(slab4)
section_label(ax, 7.15, 2.92, "LLM GATEWAY  (sub-agents: OpenRouter primary → Groq fallback → direct tool)")

box(ax, 10.5, 1.95, 4.0, 0.95, "OpenRouter  [Tier 1]",
    "meta-llama/llama-3.3-70b-instruct · primary",
    color="#2d1a08", border=C_LLM, fontsize=9.5)
box(ax, 16.5, 1.95, 4.0, 0.95, "Groq  [Tier 2]",
    "llama-4-scout-17b · fallback on 429 / 5xx",
    color="#1a0d2e", border=C_GROQ, fontsize=9.5)

arrow(ax, 12.5, 1.95, 14.5, 1.95, color="#484f58", lw=1.2, label="→ on error")

# Connect agents to LLM gateway
for ax_x in [3.5, 10.0, 16.5]:
    arrow(ax, ax_x, 3.7, ax_x, 3.2, color=C_ARROW, lw=1)

arrow(ax, 10.0, 3.2, 10.5, 2.43, color=C_LLM,  lw=1.2)
arrow(ax, 16.5, 3.2, 16.5, 2.43, color=C_GROQ, lw=1.2)

# ── Legend ────────────────────────────────────────────────────────────────────
items = [
    (C_RAG,    "RAG path"),
    (C_CALC,   "Calculator path"),
    (C_SEARCH, "Search path"),
    (C_OLLAMA, "Ollama (local)"),
    (C_LLM,    "OpenRouter Tier 1"),
    (C_GROQ,   "Groq Tier 2"),
    (C_DUCK,   "DuckDB storage"),
]
for i, (color, label) in enumerate(items):
    x = 7.2 + i * 1.85
    ax.add_patch(mpatches.Rectangle((x, 0.35), 0.35, 0.25,
                 color=color, zorder=5))
    ax.text(x + 0.45, 0.48, label, fontsize=7.5, color=C_SUBTEXT,
            va="center", zorder=5)

ax.text(10.0, 0.12, "Project Delphi · Phase 1 Baseline · Agno 2.6.18",
        ha="center", va="center", fontsize=7.5, color="#484f58", zorder=5)

# ── Save ──────────────────────────────────────────────────────────────────────
out = os.path.join(os.path.dirname(__file__), "architecture.png")
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=C_BG)
plt.close(fig)
print(f"Saved: {out}")
