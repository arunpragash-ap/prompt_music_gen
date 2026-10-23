#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# init_repo.sh - initialise the LOCAL git repository (no GitHub, no push).
#
# Usage:
#   ./scripts/init_repo.sh              # init (if needed) + initial commit
#   ./scripts/init_repo.sh --branches   # also create the phase-* branches
#
# The script is idempotent: running it on an existing repository only reports
# the state and commits any uncommitted changes. It never pushes anywhere and
# never rewrites history.
# ---------------------------------------------------------------------------
set -euo pipefail

cd "$(dirname "$0")/.."

MAKE_BRANCHES=0
for arg in "$@"; do
  case "$arg" in
    --branches) MAKE_BRANCHES=1 ;;
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

if [ ! -d .git ]; then
  echo "initialising git repository (branch: main)"
  git init -b main >/dev/null
else
  echo "git repository already exists"
fi

# .gitignore is part of the repo; never overwrite a user-edited one.
if [ ! -f .gitignore ]; then
  echo "WARNING: .gitignore is missing - restore it before committing large data" >&2
fi

git add -A
if git diff --cached --quiet; then
  echo "nothing to commit"
else
  git -c user.name="${GIT_AUTHOR_NAME:-prompt_music_gen}" \
      -c user.email="${GIT_AUTHOR_EMAIL:-promptmusic@localhost}" \
      commit -m "Initial commit: prompt_music_gen (text -> MIDI -> audio)" >/dev/null
  echo "created initial commit"
fi

if [ "$MAKE_BRANCHES" -eq 1 ]; then
  # Branch-per-phase layout. Branches point at the current commit; later
  # phase work commits on top of the matching branch and merges into main.
  for b in phase-1-data phase-2-translate phase-3-lyrics \
           phase-4-reference phase-5-ragas phase-6-audit; do
    if git show-ref --verify --quiet "refs/heads/$b"; then
      echo "branch exists: $b"
    else
      git branch "$b"
      echo "created branch: $b"
    fi
  done
fi

echo
echo "repository state:"
git --no-pager log --oneline -5 2>/dev/null || true
echo
echo "branches:"
git --no-pager branch
echo
echo "remotes (should be empty - this project stays local):"
git remote -v
