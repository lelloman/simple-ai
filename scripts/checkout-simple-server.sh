#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
destination="$(dirname "$repo_root")/simple-server"
revision="$(cat "$repo_root/simple-server.rev")"

if [[ ! "$revision" =~ ^[0-9a-f]{40}$ ]]; then
    echo "simple-server.rev must contain a full Git commit hash" >&2
    exit 1
fi

if [[ -e "$destination" ]]; then
    if [[ ! -e "$destination/.git" ]] ||
        [[ "$(git -C "$destination" rev-parse HEAD)" != "$revision" ]] ||
        [[ -n "$(git -C "$destination" status --porcelain)" ]]; then
        echo "Existing $destination is not a clean checkout of $revision; leaving it unchanged." >&2
        exit 1
    fi
else
    git clone --no-checkout https://github.com/lelloman/simple-server.git "$destination"
    git -C "$destination" checkout --detach "$revision"
fi

echo "simple-server checkout verified at $revision"
