#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# disk_report.sh - what occupies disk, and what is safe to delete.
#
# Prefers the Python implementation (aligned table, reasons, JSON) and falls
# back to plain `du` when Python/promptmusic is not available, so this script
# works even in a half-installed checkout.
#
# See docs/10_DISK_CLEANUP.md.
# ---------------------------------------------------------------------------
set -euo pipefail

cd "$(dirname "$0")/.."

PY=".venv/bin/python"
if [ -x "$PY" ] && "$PY" -c "import promptmusic" >/dev/null 2>&1; then
  exec "$PY" -m promptmusic.cli disk-report "$@"
fi

if command -v python3 >/dev/null 2>&1 && python3 -c "import promptmusic" >/dev/null 2>&1; then
  exec python3 -m promptmusic.cli disk-report "$@"
fi

echo "promptmusic is not importable; falling back to du(1)"
echo
printf '%-44s %10s\n' "PATH" "SIZE"
printf '%-44s %10s\n' "--------------------------------------------" "----------"
for p in dataset data/downloads data/cache data/processed .hf_cache outputs .venv; do
  if [ -e "$p" ]; then
    printf '%-44s %10s\n' "$p" "$(du -sh "$p" 2>/dev/null | cut -f1)"
  fi
done
echo
echo "regenerable: dataset/ data/downloads/ data/cache/ data/processed/ .hf_cache/ outputs/ .venv/"
echo "full guidance: docs/10_DISK_CLEANUP.md"
