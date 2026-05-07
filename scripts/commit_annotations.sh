#!/usr/bin/env bash
# Wraps `git add per-frame.json && commit && pull --rebase && push` with
# retries that handle .git/index.lock contention from sibling agents
# running on the same machine. Per-frame JSONs mean rebases never
# conflict on content; the only thing we wait out is the brief lock window.
#
# Usage:  bash scripts/commit_annotations.sh <slug> "<commit message>"
set -euo pipefail

if [ $# -ne 2 ]; then
    echo "Usage: $0 <slug> \"<commit message>\""
    echo "Example: $0 act 'Self-verb annotations: act seeds 0-4'"
    exit 1
fi

SLUG=$1
MSG=$2
BRANCH=ood-init
JSON=results/all_frames/manual_self_verb_${SLUG}.json

cd "$(dirname "$0")/.."

if [ ! -f "$JSON" ]; then
    echo "ERROR: $JSON not found"
    exit 1
fi

retry() {
    # $1 = description, rest = command. Up to 8 tries with backoff.
    local desc=$1; shift
    for attempt in 1 2 3 4 5 6 7 8; do
        if "$@"; then
            return 0
        fi
        echo "  $desc failed (attempt $attempt), backing off..."
        sleep $((attempt * 2))
    done
    echo "  ERROR: $desc failed after 8 attempts"
    return 1
}

retry "git add"          git add "$JSON"
if git diff --cached --quiet; then
    echo "(nothing to commit)"
    exit 0
fi
retry "git commit"       git commit -m "$MSG"
retry "git pull --rebase" git pull --rebase origin "$BRANCH"
retry "git push"         git push origin "$BRANCH"
echo "ok: $SLUG annotations pushed"
