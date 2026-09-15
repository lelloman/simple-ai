# SimpleAI Android

SimpleAI manages shared AI models for compatible Android apps. Users download the features they need and approve clients under **Apps**. Voice capture and chat conversations belong to the connecting app; SimpleAI includes a translator that exercises its service API.

The app has four main screens: **Models** manages downloads, **Translate** translates text, **Apps** manages access approvals, and **Settings** contains network preferences, storage and support. Model actions live on individual detail screens.

## Capabilities and storage

| Capability | Processing | Artifact download |
|---|---|---|
| Voice Commands | On-device XLM-RoBERTa int8 plus client adapter | 533,595,982 bytes (533.6 MB) |
| Translation | On-device ML Kit, 59 supported languages | About 30 MB per pack; varies by language. English is built in. |
| Cloud AI | HTTPS request to the configured provider, using the user’s SimpleAI sign-in | No model download |
| Local AI | On-device Qwen3 1.7B Q4_K_M; plain-text chat, no tool calling | 1,282,439,584 bytes (1.28 GB) |

Sizes use decimal MB/GB. Voice Commands also needs a 533.6 MB working copy; interrupted downloads and model loading need additional space. Download checks reserve 67.1 MB. Model URLs/revisions/hashes are authoritative in [NluModel.kt](app/src/main/java/com/lelloman/simpleai/model/NluModel.kt) and [AvailableModels.kt](app/src/main/java/com/lelloman/simpleai/model/AvailableModels.kt).

Android 7.0/API 24 is the minimum. APKs contain ARM64 and ARMv7 components, but the bundled llama.cpp implementation requires ARM64 for Local AI. See [16 KB validation](../docs/android-16kb-validation.md) for native device acceptance still outstanding.

Opening SimpleAI inspects model inventory without eagerly loading native engines. Inference loads models on demand; native resources unload 60 seconds after work finishes. Downloads survive activity recreation through WorkManager, use unmetered networks by default, and offer pause/resume. English is separate from downloaded/removable translation packs. See [service lifetime](../docs/android-service-lifetime.md) and [client access](../docs/android-client-access.md).

## Reproducible developer setup

Install JDK 17, Android SDK command-line tools, Rust 1.96.0 and Python 3. Put SDK tools and Cargo on PATH. From a complete checkout:

```bash
cd simple-ai/android
sdkmanager 'platforms;android-36' 'build-tools;36.0.0' 'ndk;27.0.12077973'
rustup toolchain install 1.96.0 --profile minimal
rustup target add --toolchain 1.96.0 aarch64-linux-android armv7-linux-androideabi
cargo +1.96.0 install cargo-ndk --version 4.1.2 --locked
```

Set `ANDROID_HOME` to your SDK directory, or copy [local.properties.example](local.properties.example) to ignored `local.properties` and set `sdk.dir`. Gradle uses NDK 27.0.12077973. Use `RUSTUP_TOOLCHAIN=1.96.0` for the following commands when your default Rust differs:

```bash
./gradlew assembleDebug
./scripts/check
# Optional: install only onto the device you intend to test.
./gradlew installDebug
```

Gradle builds the Rust tokenizer JNI library for ARM64/ARMv7 and a host library for JVM tests. Cargo dependencies are locked in `tokenizer-native/Cargo.lock`. Initial setup needs network access; `./scripts/check --offline` uses the Gradle cache once dependencies exist. Small tokenizer fixtures were generated with Python `tokenizers==0.22.2`; see `tokenizer-native/generate_fixtures.py`.

Configure the server on the device under **Settings → Cloud AI**. Enter an HTTPS base URL and save; subsequent requests use it immediately. Clear the field to disable Cloud AI. This device-local setting survives restarts and is excluded from backup. Sign in inside SimpleAI after saving. Approved apps share this gateway without their own accounts. Changing the server clears the gateway credential; sign-out removes it locally. Saving alone does not verify connectivity.

An optional initial default can be supplied in `local.properties`:

```properties
cloud.llm.endpoint=https://your-cloud-service.example
```

Use an HTTPS base URL without credentials, query or fragment. SimpleAI appends `/v1/chat/completions`. The device setting overrides the build default, including when explicitly cleared. Missing/invalid configuration is visibly unavailable; configured availability does not certify connectivity or credentials. OAuth tokens are owned by SimpleAI and stored encrypted with Android Keystore, outside backups. They are never exposed to calling apps or stored in build configuration.

## Version and signing policy

[version.properties](version.properties) defines the app version independently of Git history or shallow clones. Increase `versionCode` monotonically for every distributed build; coordinate the next code with the release channel. Optional explicit overrides are `-Psimpleai.versionCode=228 -Psimpleai.versionName=1.0.228`. The same version file/overrides and toolchains yield the same version identity; byte-for-byte APK reproducibility is not claimed.

Debug builds use Android's debug key. Release builds require all four signing properties and an existing keystore. Copy [signing.properties.example](signing.properties.example) to ignored `signing.properties`, or pass `-Psimpleai.signingProperties=/absolute/path/to/properties`. Keystore paths are absolute or relative to `android/app`. Keep signing files/passwords out of Git.

```bash
./gradlew validateReleaseConfiguration
./gradlew assembleRelease
# Or build a signed bundle:
./gradlew bundleRelease
```

The release pre-build fails with a clear error if signing is absent; it cannot silently produce an unsigned release. These commands only build artifacts and do not publish them. Release signing/builds were not exercised with production credentials during the audit work.

## Publish to LelloStore

From the repository root, use the same build-and-publish wrapper pattern as Pezzottify and LelloStore:

```bash
# Build the signed release and validate locally, without authentication or upload:
./scripts/publish-android-to-lellostore.sh --dry-run --json
# Build and publish after the publisher asks for confirmation:
./scripts/publish-android-to-lellostore.sh
# Optional beta channel:
./scripts/publish-android-to-lellostore.sh --beta
```

Configure `android/signing.properties` first and update `android/version.properties` for each distributed version. The wrapper always builds `release` and selects exactly `android/app/build/outputs/apk/release/app-release.apk`; build/signing failures stop before publishing. It can be invoked from any working directory.

Authentication, validation and upload remain in LelloStore's authoritative `scripts/publish-to-lellostore.py`. Set `LELLOSTORE_PUBLISHER` to that executable, or keep the LelloStore checkout at `$HOME/lelloprojects/lellostore` or alongside SimpleAI. The wrapper does not copy the publisher or store credentials.

Defaults match Pezzottify: store `https://store.lelloman.com`, issuer `https://auth.lelloman.com`, public client ID `22cd4a2d-a771-41e3-b76e-3f83ff8e9bbf`. Override these with `LELLOSTORE_URL`, `LELLOSTORE_OIDC_ISSUER`, `LELLOSTORE_CLIENT_ID`, or forwarded `--store-url`, `--issuer`, `--client-id` options. All publisher upload arguments pass through unchanged. `--yes` skips its confirmation and should only be used for an already-authorized upload. `--dry-run` still builds a signed APK, but does not authenticate or upload. Build output goes to stderr so `--json` output remains usable by scripts.

## Client integration

Copy [ISimpleAI.aidl](app/src/main/aidl/com/lelloman/simpleai/ISimpleAI.aidl) into the same package in the client. Enable AIDL in its Android build. If the client queries installation/package details on Android 11+, declare package visibility:

```xml
<queries>
    <package android:name="com.lelloman.simpleai" />
</queries>
```

Bind explicitly to `com.lelloman.simpleai.service.SimpleAIService` with `Context.BIND_AUTO_CREATE`; keep the binding while requests run and unbind when finished. Handle false binds, null/dead bindings and temporary disconnections. Do not start a persistent service merely to inspect capabilities. [ServiceBinding.kt](app/src/main/java/com/lelloman/simpleai/ui/ServiceBinding.kt) demonstrates registration cleanup.

All synchronous inference calls belong on a worker thread. Current minimum and maximum protocol are both **2**. Always inspect response `status` before accessing `data`:

```kotlin
// Inside a coroutine, after obtaining api = ISimpleAI.Stub.asInterface(binder):
val result = withContext(Dispatchers.IO) { api.getServiceInfo(2) }
val envelope = Json.parseToJsonElement(result).jsonObject
if (envelope["status"]?.jsonPrimitive?.content == "success") {
    val capabilities = envelope.getValue("data").jsonObject.getValue("capabilities")
    // Render capability state before making requests.
} else {
    val error = envelope.getValue("error").jsonObject
    // Display error["message"]; handle the typed error code.
}
```

Protocol 1 returns `UNSUPPORTED_PROTOCOL`; a protocol newer than 2 returns `VERSION_TOO_OLD`. Service discovery is public. Expensive calls initially return `CLIENT_NOT_APPROVED`; direct users to SimpleAI → Apps and retry after approval. Approval is bound to package/signing identity. Per UID: one active request and up to 30 starts per minute. Four expensive requests can be active globally; excess requests return `RATE_LIMITED`.

Example successful service response (illustrative state):

```json
{
  "status": "success",
  "protocolVersion": 2,
  "data": {
    "serviceVersion": 2,
    "minProtocol": 2,
    "maxProtocol": 2,
    "supportsCancellation": true,
    "capabilities": {
      "voiceCommands": {"status": "ready", "loaded": false},
      "translation": {"status": "ready", "languages": ["it"], "builtInLanguages": ["en"]},
      "cloudAi": {"status": "error", "message": "Sign in to SimpleAI in Settings → Cloud AI", "canRetry": false},
      "localAi": {"status": "not_downloaded", "modelSize": 1282439584}
    }
  }
}
```

Other capability states are `checking`, `loading`, `downloading` (progress), and `error`. Disk-only native models use `ready, loaded: false`; first inference may need to load them.

## Request API

The [AIDL file](app/src/main/aidl/com/lelloman/simpleai/ISimpleAI.aidl) defines exact parameter order. These examples use positional arguments because generated AIDL methods are Java methods:

```kotlin
// Execute on Dispatchers.IO, using an approved bound api:
api.translate(2, "Hello", "en", "it")
api.getTranslationLanguages(2) // Includes built-in English.
api.localGenerate(2, "Write a short greeting", 128, 0.7f)
api.localChat(2, """[{"role":"user","content":"Hello"}]""", null, null)
api.cloudChat(2, """[{"role":"user","content":"Hello"}]""", null, null, null, "")
```

Local chat uses the Qwen text template with thinking disabled; tool calls/structured content are rejected. Local generation accepts 1–2,048 output tokens and finite temperature 0–2. Cloud chat explicitly requests `class:fast`, using the server’s configured fast-model routing regardless of the signed-in user’s roles. Cloud chat may include tool definitions and a cache key; HTTP 429 returns `RATE_LIMITED`, and response usage may be absent/null.

`classify(2, text, adapterId, adapterVersion, patchFd, headsFd, tokenizerFd, configFd)` applies a client adapter and returns intent/confidence/slots. Descriptors must remain open until the call returns, then the client closes them. Adapter identity is scoped to Binder UID; selection plus inference is atomic. `clearAdapter(2)` only removes that UID's active adapter. The base model remains immutable; all patches target a checked working copy.

Adapter validation limits: 1–1,024 intent/slot labels, 768 hidden features, exact little-endian heads/bias dimensions, finite weights, 1 MiB config, 32 MiB tokenizer, max sequence length 2–512, valid unique labels/BIO slots. Patches allow at most 4,096 nonoverlapping in-bounds ranges, 16 MiB per range and 128 MiB total. Invalid data is rejected before activation.

All request results use a success `data` object or an error object with `code`, `message` and optional `details`. See [ErrorCode.kt](app/src/main/java/com/lelloman/simpleai/api/ErrorCode.kt) and [request bounds/cancellation](../docs/android-request-contract.md). If discovery advertises `supportsCancellation`, another client thread may invoke `cancelCurrentRequest(2)` to signal its own current operation. Native non-suspending work may finish before cancellation cleanup; do not assume immediate native interruption.

## Backup and restore

Both legacy cloud backup and Android 12+ cloud/device transfer allow only `downloads.xml` (the small network preference). Model directories, partials, adapter working copies, ML Kit state, WorkManager databases, capability inventory and client approvals are excluded. Models must be downloaded again and clients approved again on the restored device. The include-only policy also avoids silently backing up future bulk files. Rules are in `res/xml/backup_rules.xml` and `res/xml/data_extraction_rules.xml`; see [Android's backup format](https://developer.android.com/identity/data/autobackup).

## Validation and support

```bash
./scripts/check
./gradlew testDebugUnitTest --tests 'com.lelloman.simpleai.api.RequestContractTest'
# Connected device required:
./gradlew connectedDebugAndroidTest
```

The check script builds APKs, runs unit tests/lint, tests the Rust tokenizer and checks native 16 KB alignment. Reports are in `app/build/reports/` and `app/build/test-results/`. There is no JaCoCo task or asserted Android coverage percentage. See [coverage map and CI limits](../docs/android-test-coverage.md), [accessibility checklist](../docs/android-accessibility-checks.md), and [16 KB runtime checks](../docs/android-16kb-validation.md).

About → Copy support diagnostics includes build/protocol, device ABI/API and pinned model identity; it excludes prompts, answers, tokens, client approvals and endpoints. See [metadata](../docs/android-metadata.md) and [cloud logging policy](../docs/android-cloud-diagnostics.md). This repository's license is Apache 2.0; model/library terms are linked separately in About.

### Personal gateway sign-in

The server exposes public `/.well-known/simple-ai` metadata with `issuer` and
`client_id`. Configure `[oidc].android_client_id` with a public OAuth client
supporting authorization code + PKCE and the exact redirect URI
`com.lelloman.simpleai:/oauth2redirect`. A web client may be reused only when it
supports public native authentication without an embedded secret. The Android
app uses AppAuth and sends its own access token; the server validates its signature,
issuer, audience, and expiry. The deprecated AIDL `authToken` argument is ignored.
Use an empty string in new callers. Local app approval is still required.

The gateway derives `X-SimpleAI-Source-App` from the Binder caller’s package name(s).
It is stored as `requests.source_app` and returned in request history solely for
usage attribution; it is not trusted for server authentication or authorization.
