#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

echo ""
echo "  ███████╗ █████╗ ███████╗████████╗██╗███╗   ██╗ ██████╗"
echo "  ██╔════╝██╔══██╗██╔════╝╚══██╔══╝██║████╗  ██║██╔═══██╗"
echo "  █████╗  ███████║███████╗   ██║   ██║██╔██╗ ██║██║   ██║"
echo "  ██╔══╝  ██╔══██║╚════██║   ██║   ██║██║╚██╗██║██║   ██║"
echo "  ██║     ██║  ██║███████║   ██║   ██║██║ ╚████║╚██████╔╝"
echo "  ╚═╝     ╚═╝  ╚═╝╚══════╝   ╚═╝   ╚═╝╚═╝  ╚═══╝ ╚═════╝ "
echo "  Labs — LLM Inference Demo"
echo ""

# ── Dependencies ──────────────────────────────────────────────────────────────
echo "→ Checking Python dependencies..."
pip install -q fastapi "uvicorn[standard]" httpx pydantic

# ── Open browser (macOS) ──────────────────────────────────────────────────────
(sleep 2 && open http://localhost:8000 2>/dev/null) &

# ── Launch server ─────────────────────────────────────────────────────────────
echo "→ Starting server at http://localhost:8000"
echo "  Ctrl-C to stop."
echo ""
uvicorn src.server:app --host 0.0.0.0 --port 8000 --reload
