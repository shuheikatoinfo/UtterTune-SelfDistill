#!/usr/bin/env bash
# Fetch the pinned public Irodori-TTS into irodori/upstream, apply upstream.patch, and install its
# environment with uv (CUDA 12.8 wheels, as the upstream lock file provides). Safe to re-run.
set -euo pipefail
UPSTREAM_URL=https://github.com/Aratako/Irodori-TTS.git
UPSTREAM_COMMIT=eaf74d6a19138f743acb5b71a445fd25a57db987
HERE=$(cd "$(dirname "$0")" && pwd)
DEST=$HERE/upstream

[ -d "$DEST/.git" ] || git clone --quiet "$UPSTREAM_URL" "$DEST"
git -C "$DEST" cat-file -e "$UPSTREAM_COMMIT^{commit}" 2>/dev/null || git -C "$DEST" fetch --quiet origin
git -C "$DEST" checkout --quiet --detach "$UPSTREAM_COMMIT"
if git -C "$DEST" apply --reverse --check "$HERE/upstream.patch" 2>/dev/null; then
  echo "upstream.patch already applied"
else
  git -C "$DEST" apply "$HERE/upstream.patch"
fi
(cd "$DEST" && uv sync --frozen --extra cu128)
echo "ready: $DEST at $(git -C "$DEST" rev-parse --short HEAD) + upstream.patch"
