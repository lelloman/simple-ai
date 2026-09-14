#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPOSITORY_DIR=$(cd -- "$SCRIPT_DIR/.." && pwd)
ANDROID_DIR="$REPOSITORY_DIR/android"
SIGNING_PROPERTIES="$ANDROID_DIR/signing.properties"
ARTIFACT="$ANDROID_DIR/app/build/outputs/apk/release/app-release.apk"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    cat <<'HELP'
Usage: scripts/publish-android-to-lellostore.sh [publisher upload options]

Build SimpleAI's signed release APK and invoke the shared LelloStore publisher.
Requires android/signing.properties and the Android/Rust build toolchain.
Version identity comes from android/version.properties.

Examples:
  scripts/publish-android-to-lellostore.sh --dry-run --json
  scripts/publish-android-to-lellostore.sh
  scripts/publish-android-to-lellostore.sh --beta

Upload options are forwarded unchanged, including --store-url, --issuer,
--client-id, --name, --description, --dry-run, --json, --beta and --yes.
Without --yes, the publisher asks for interactive upload confirmation.

Environment: LELLOSTORE_PUBLISHER, LELLOSTORE_URL, LELLOSTORE_OIDC_ISSUER,
LELLOSTORE_CLIENT_ID. See android/README.md for setup and defaults.
HELP
    exit 0
fi

# Match the deployment defaults used by Pezzottify; all remain overridable.
export LELLOSTORE_URL="${LELLOSTORE_URL:-https://store.lelloman.com}"
export LELLOSTORE_OIDC_ISSUER="${LELLOSTORE_OIDC_ISSUER:-https://auth.lelloman.com}"
export LELLOSTORE_CLIENT_ID="${LELLOSTORE_CLIENT_ID:-22cd4a2d-a771-41e3-b76e-3f83ff8e9bbf}"

if [[ ! -f "$SIGNING_PROPERTIES" ]]; then
    echo "Missing Android release signing configuration: $SIGNING_PROPERTIES" >&2
    echo "Copy android/signing.properties.example and fill in the release keystore details." >&2
    exit 1
fi

if [[ -n "${LELLOSTORE_PUBLISHER:-}" ]]; then
    PUBLISHER="$LELLOSTORE_PUBLISHER"
elif [[ -x "${HOME}/lelloprojects/lellostore/scripts/publish-to-lellostore.py" ]]; then
    PUBLISHER="${HOME}/lelloprojects/lellostore/scripts/publish-to-lellostore.py"
elif [[ -x "$REPOSITORY_DIR/../lellostore/scripts/publish-to-lellostore.py" ]]; then
    PUBLISHER="$REPOSITORY_DIR/../lellostore/scripts/publish-to-lellostore.py"
else
    echo "Could not find the authoritative LelloStore publisher." >&2
    echo "Set LELLOSTORE_PUBLISHER to scripts/publish-to-lellostore.py in a LelloStore checkout." >&2
    exit 1
fi

if [[ ! -x "$PUBLISHER" ]]; then
    echo "LelloStore publisher is not executable: $PUBLISHER" >&2
    exit 1
fi

# Keep build/status output off stdout so --json remains machine-readable.
echo "Building signed SimpleAI release APK..." >&2
(
    cd "$ANDROID_DIR"
    ./gradlew :app:assembleRelease
) >&2

if [[ ! -s "$ARTIFACT" ]]; then
    echo "Expected signed release APK was not produced: $ARTIFACT" >&2
    exit 1
fi

ARTIFACT_SIZE=$(stat --format='%s' "$ARTIFACT")
echo "Artifact: $ARTIFACT" >&2
echo "Variant:  release" >&2
echo "Size:     $ARTIFACT_SIZE bytes" >&2

exec "$PUBLISHER" upload "$ARTIFACT" "$@"
