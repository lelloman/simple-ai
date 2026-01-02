# SimpleAI Refactor Plan

## Overview

SimpleAI is a shared Android service that provides AI capabilities to other apps. It acts as a capability manager where users opt-in to features and download required models.

## Capabilities

| Capability | Description | Downloads Required |
|------------|-------------|-------------------|
| **Voice Commands** | NLU intent classification + entity extraction | XLM-RoBERTa base (~120MB) |
| **Translation** | On-device translation via ML Kit | ~30MB per language (English mandatory as pivot) |
| **Cloud AI** | Proxy to our cloud LLM endpoint | None (requires auth token from client) |
| **Local AI** | On-device LLM inference | Qwen 3 1.7B (~1.3GB) |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Client Apps                             │
│                  (SimpleEphem, others)                       │
└─────────────────────────┬───────────────────────────────────┘
                          │ AIDL (ISimpleAI)
┌─────────────────────────▼───────────────────────────────────┐
│                     SimpleAI Service                         │
│  ┌────────────────────────────────────────────────────────┐ │
│  │              Protocol Version Handler                   │ │
│  │         (v1, v2, ... backward compatibility)           │ │
│  └────────────────────────────────────────────────────────┘ │
│  ┌─────────────┬─────────────┬─────────────┬──────────────┐ │
│  │   Voice     │ Translation │  Cloud AI   │  Local AI    │ │
│  │  Commands   │  (ML Kit)   │   (Proxy)   │ (llama.cpp)  │ │
│  ├─────────────┼─────────────┼─────────────┼──────────────┤ │
│  │ OnnxNLU     │ MLKit       │ OkHttp to   │ LlamaEngine  │ │
│  │ Engine      │ Translator  │ cloud       │              │ │
│  │ + LoRA      │             │ endpoint    │              │ │
│  │ cache       │             │             │              │ │
│  └─────────────┴─────────────┴─────────────┴──────────────┘ │
│  ┌────────────────────────────────────────────────────────┐ │
│  │              Capability Manager                         │ │
│  │    (download status, progress, errors, retry)          │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

---

## AIDL Protocol Design

### Versioning Strategy

Two version numbers:

| Version | Increments | Purpose |
|---------|------------|---------|
| **SERVICE_VERSION** | Every release | "Is there an update?" |
| **MIN_PROTOCOL_VERSION** | Breaking changes only | "Can I use this SimpleAI?" |

- **SERVICE_VERSION** (int): Current release version (1, 2, 3...). Bumped on every release.
- **MIN_PROTOCOL_VERSION** (int): Minimum protocol this SimpleAI supports. Bumped only when we drop backward compatibility.
- **Client's minProtocol** (int): Minimum protocol the client needs. If SimpleAI's SERVICE_VERSION supports this protocol, we're good.

**Examples:**

```
SimpleAI v5 (supports protocols 1-3)
Client needs protocol 2
→ OK: SimpleAI can speak protocol 2

SimpleAI v5 (supports protocols 3-3, dropped 1-2)
Client needs protocol 2
→ Error: VERSION_TOO_OLD - SimpleAI dropped protocol 2, client must update

SimpleAI v5 (supports protocols 1-3)
Client needs protocol 4
→ Error: VERSION_TOO_OLD - SimpleAI doesn't know protocol 4 yet, user should update SimpleAI
```

**Build Config:**
```kotlin
buildConfigField("int", "SERVICE_VERSION", "5")
buildConfigField("int", "MIN_PROTOCOL_VERSION", "1")  // oldest protocol we still support
buildConfigField("int", "MAX_PROTOCOL_VERSION", "3")  // newest protocol we support
```

**Client passes:** `protocolVersion` (int) with each request. SimpleAI checks:
1. If `protocolVersion < MIN_PROTOCOL_VERSION` → client too old, needs update
2. If `protocolVersion > MAX_PROTOCOL_VERSION` → SimpleAI too old, needs update
3. Otherwise → respond using that protocol version

### Standard Response Format

All AIDL methods return JSON strings with this structure:

```json
// Success
{
  "status": "success",
  "protocolVersion": 1,
  "data": { ... }
}

// Error
{
  "status": "error",
  "protocolVersion": 1,
  "error": {
    "code": "CAPABILITY_NOT_READY",
    "message": "Voice Commands capability is not downloaded",
    "details": { ... }
  }
}
```

### Error Codes

| Code | Description |
|------|-------------|
| `VERSION_TOO_OLD` | SimpleAI version is older than client's minVersion |
| `UNSUPPORTED_PROTOCOL` | Requested protocol version not supported |
| `CAPABILITY_NOT_READY` | Required capability not downloaded |
| `CAPABILITY_DOWNLOADING` | Capability download in progress (includes progress %) |
| `CAPABILITY_ERROR` | Capability failed to initialize (includes error details) |
| `ADAPTER_LOAD_FAILED` | Failed to apply LoRA adapter (corrupt file, version mismatch, etc.) |
| `INVALID_REQUEST` | Malformed request parameters |
| `CLOUD_AUTH_FAILED` | Cloud endpoint rejected auth token |
| `CLOUD_UNAVAILABLE` | Cloud endpoint unreachable |
| `TRANSLATION_LANGUAGE_NOT_AVAILABLE` | Requested language not downloaded |
| `INTERNAL_ERROR` | Unexpected error |

### AIDL Interface

```aidl
package com.lelloman.simpleai;

interface ISimpleAI {

    // =========================================================================
    // SERVICE INFO
    // =========================================================================

    /**
     * Get service version and capabilities status.
     *
     * @param protocolVersion Client's protocol version (e.g., 1, 2, 3)
     * @return JSON response:
     *   {
     *     "status": "success",
     *     "data": {
     *       "serviceVersion": 5,
     *       "minProtocol": 1,
     *       "maxProtocol": 3,
     *       "capabilities": {
     *         "voiceCommands": {"status": "ready", "modelSize": 120000000},
     *         "translation": {"status": "downloading", "progress": 0.45, "languages": ["en", "it"]},
     *         "cloudAi": {"status": "ready"},
     *         "localAi": {"status": "not_downloaded", "modelSize": 1300000000}
     *       }
     *     }
     *   }
     */
    String getServiceInfo(int protocolVersion);

    // =========================================================================
    // VOICE COMMANDS (NLU)
    // =========================================================================

    /**
     * Classify text using NLU with a specific adapter.
     *
     * Adapter files are passed with every request. SimpleAI tracks the currently
     * applied adapter by (adapterId, adapterVersion). If it matches, inference runs
     * directly. If different, SimpleAI reverts the current patch, applies the new
     * one, and stores the revert data for future switches.
     *
     * @param protocolVersion Client's protocol version (e.g., 1, 2, 3)
     * @param text Text to classify
     * @param adapterId Unique identifier for the adapter (e.g., "simpleephem")
     * @param adapterVersion Version string for change detection (e.g., "1.0.3")
     * @param patchFd ParcelFileDescriptor for .lorapatch file
     * @param headsFd ParcelFileDescriptor for heads.bin file
     * @param tokenizerFd ParcelFileDescriptor for tokenizer.json
     * @param configFd ParcelFileDescriptor for config.json
     * @return JSON response:
     *   Success:
     *   {
     *     "status": "success",
     *     "data": {
     *       "intent": "add_subject",
     *       "intentConfidence": 0.94,
     *       "slots": {
     *         "person": ["John"],
     *         "date": ["tomorrow"]
     *       }
     *     }
     *   }
     */
    String classify(
        int protocolVersion,
        String text,
        String adapterId,
        String adapterVersion,
        in ParcelFileDescriptor patchFd,
        in ParcelFileDescriptor headsFd,
        in ParcelFileDescriptor tokenizerFd,
        in ParcelFileDescriptor configFd
    );

    /**
     * Remove currently applied adapter and restore pristine model.
     * Useful when client app is done and wants to free memory.
     *
     * @param protocolVersion Client's protocol version
     */
    String clearAdapter(int protocolVersion);

    // =========================================================================
    // TRANSLATION
    // =========================================================================

    /**
     * Translate text between languages.
     *
     * @param protocolVersion Client's protocol version
     * @param text Text to translate
     * @param sourceLang Source language code (e.g., "it") or "auto" for detection
     * @param targetLang Target language code (e.g., "en")
     * @return JSON response:
     *   {
     *     "status": "success",
     *     "data": {
     *       "translatedText": "Hello world",
     *       "detectedSourceLang": "it"
     *     }
     *   }
     */
    String translate(
        int protocolVersion,
        String text,
        String sourceLang,
        String targetLang
    );

    /**
     * Get list of downloaded translation languages.
     */
    String getTranslationLanguages(int protocolVersion);

    // =========================================================================
    // CLOUD AI
    // =========================================================================

    /**
     * Send chat request to cloud LLM endpoint.
     *
     * @param protocolVersion Client's protocol version
     * @param messagesJson JSON array of chat messages
     * @param toolsJson JSON array of tool definitions (optional, can be null)
     * @param systemPrompt System prompt (optional, can be null)
     * @param authToken Client's auth token for cloud service
     * @return JSON response with LLM response or tool calls
     */
    String cloudChat(
        int protocolVersion,
        String messagesJson,
        String toolsJson,
        String systemPrompt,
        String authToken
    );

    // =========================================================================
    // LOCAL AI
    // =========================================================================

    /**
     * Generate text using local LLM.
     *
     * @param protocolVersion Client's protocol version
     * @param prompt Text prompt
     * @param maxTokens Maximum tokens to generate
     * @param temperature Sampling temperature (0.0 - 2.0)
     * @return JSON response with generated text
     */
    String localGenerate(
        int protocolVersion,
        String prompt,
        int maxTokens,
        float temperature
    );

    /**
     * Chat using local LLM with tool support.
     *
     * @param protocolVersion Client's protocol version
     * @param messagesJson JSON array of chat messages
     * @param toolsJson JSON array of tool definitions (optional)
     * @param systemPrompt System prompt (optional)
     * @return JSON response with LLM response or tool calls
     */
    String localChat(
        int protocolVersion,
        String messagesJson,
        String toolsJson,
        String systemPrompt
    );
}
```

---

## UI Design

### Main Screen

Simple list of capability cards. No model selection, no advanced settings.

```
┌─────────────────────────────────────────┐
│  SimpleAI                          [?]  │
├─────────────────────────────────────────┤
│                                         │
│  ┌───────────────────────────────────┐  │
│  │ 🎤 Voice Commands                 │  │
│  │                                   │  │
│  │ [████████████░░░░░░░] 78%         │  │
│  │ Downloading... 94 MB / 120 MB     │  │
│  └───────────────────────────────────┘  │
│                                         │
│  ┌───────────────────────────────────┐  │
│  │ 🌐 Translation                    │  │
│  │                                   │  │
│  │ ✓ Ready                           │  │
│  │ Languages: English, Italian       │  │
│  │                        [Configure]│  │
│  └───────────────────────────────────┘  │
│                                         │
│  ┌───────────────────────────────────┐  │
│  │ ☁️ Cloud AI                       │  │
│  │                                   │  │
│  │ ✓ Ready (no download required)    │  │
│  └───────────────────────────────────┘  │
│                                         │
│  ┌───────────────────────────────────┐  │
│  │ 🤖 Local AI                       │  │
│  │                                   │  │
│  │ Not downloaded                    │  │
│  │ Size: 1.3 GB                      │  │
│  │                        [Download] │  │
│  └───────────────────────────────────┘  │
│                                         │
│  Storage used: 1.5 GB                   │
└─────────────────────────────────────────┘
```

### Capability States

- **Not Downloaded**: Shows size, download button
- **Downloading**: Progress bar, percentage, size progress, cancel button
- **Ready**: Green checkmark, usage stats if applicable
- **Error**: Red indicator, error message, retry button
- **Configuring** (Translation only): Language picker

### Translation Language Picker

```
┌─────────────────────────────────────────┐
│  ← Translation Languages                │
├─────────────────────────────────────────┤
│                                         │
│  Downloaded:                            │
│  ┌───────────────────────────────────┐  │
│  │ 🇬🇧 English (required)            │  │
│  │ 🇮🇹 Italian                    [x] │  │
│  │ 🇪🇸 Spanish                    [x] │  │
│  └───────────────────────────────────┘  │
│                                         │
│  Available:                             │
│  ┌───────────────────────────────────┐  │
│  │ 🇫🇷 French (~30 MB)           [+] │  │
│  │ 🇩🇪 German (~30 MB)           [+] │  │
│  │ 🇯🇵 Japanese (~30 MB)         [+] │  │
│  │ ...                               │  │
│  └───────────────────────────────────┘  │
│                                         │
│  Total: 90 MB                           │
└─────────────────────────────────────────┘
```

---

## Implementation Plan

### Phase 1: Core Infrastructure Refactor

#### 1.1 Define Response Types
- [ ] Create `ApiResponse` sealed class (Success/Error)
- [ ] Create `ErrorCode` enum with all error types
- [ ] Create JSON serialization for responses
- [ ] Create `ProtocolHandler` for version compatibility

**Files:**
- `app/src/main/java/com/lelloman/simpleai/api/ApiResponse.kt`
- `app/src/main/java/com/lelloman/simpleai/api/ErrorCode.kt`
- `app/src/main/java/com/lelloman/simpleai/api/ProtocolHandler.kt`

#### 1.2 Capability Manager
- [ ] Create `Capability` sealed class with states
- [ ] Create `CapabilityManager` to track all capabilities
- [ ] Implement download orchestration
- [ ] Implement status persistence across restarts

**Files:**
- `app/src/main/java/com/lelloman/simpleai/capability/Capability.kt`
- `app/src/main/java/com/lelloman/simpleai/capability/CapabilityManager.kt`
- `app/src/main/java/com/lelloman/simpleai/capability/CapabilityStatus.kt`

#### 1.3 Update AIDL Interface
- [ ] Rewrite `ISimpleAI.aidl` with new interface
- [ ] Update service to implement new interface
- [ ] Add protocol version handling to all methods

**Files:**
- `app/src/main/aidl/com/lelloman/simpleai/ISimpleAI.aidl`
- `app/src/main/java/com/lelloman/simpleai/service/SimpleAIService.kt`

### Phase 2: Voice Commands (NLU)

#### 2.1 Adapter State Tracking

The NLU engine can only hold ONE adapter at a time in memory. The "cache" is simply tracking
which adapter is currently applied so we can detect when we need to de-patch and re-patch.

**State Model:**
```
Base Model (in memory, ~120MB)
    │
    ├── No adapter applied (pristine)
    │
    └── Adapter applied:
            - adapterId: "simpleephem"
            - adapterVersion: "1.0.3"
            - revertPatch: <binary data to undo the patch>
```

**When classify() is called:**
1. Check if requested (adapterId, adapterVersion) matches currently applied adapter
2. If match → run inference directly
3. If no match:
   a. If adapter currently applied → revert patch (restore pristine model)
   b. Apply new patch from provided FDs
   c. Store revertPatch for future de-patching
   d. Run inference

- [ ] Create `AdapterState` data class to track current adapter
- [ ] Store revertPatch when applying adapter
- [ ] Implement de-patch → re-patch flow

**Files:**
- `app/src/main/java/com/lelloman/simpleai/nlu/AdapterState.kt`

#### 2.2 Refactor OnnxNLUEngine
- [ ] Update to use AdapterState (single adapter, not cache)
- [ ] Accept adapter files per-request
- [ ] Implement de-patch before re-patch logic
- [ ] Return standardized ApiResponse

**Files:**
- `app/src/main/java/com/lelloman/simpleai/nlu/OnnxNLUEngine.kt` (update)

### Phase 3: Translation (ML Kit)

#### 3.1 ML Kit Integration
- [ ] Add ML Kit translate dependency
- [ ] Create `TranslationManager`
- [ ] Implement language model download/delete
- [ ] Auto-manage English (add when first lang added, remove when all removed)

**Files:**
- `app/build.gradle.kts` (add dependency)
- `app/src/main/java/com/lelloman/simpleai/translation/TranslationManager.kt`

#### 3.2 Translation Capability
- [ ] Wire TranslationManager to CapabilityManager
- [ ] Track per-language download status
- [ ] Implement translate() AIDL method

### Phase 4: Cloud AI Proxy

#### 4.1 Cloud Client
- [ ] Create `CloudLLMClient` with OkHttp
- [ ] Use `BuildConfig.CLOUD_LLM_ENDPOINT`
- [ ] OpenAI-compatible request/response format
- [ ] Handle auth token pass-through

**Files:**
- `app/build.gradle.kts` (add buildConfigField)
- `app/src/main/java/com/lelloman/simpleai/cloud/CloudLLMClient.kt`

#### 4.2 Cloud Capability
- [ ] Always-ready capability (no downloads)
- [ ] Implement cloudChat() AIDL method

### Phase 5: Local AI

#### 5.1 Simplify LLM Engine
- [ ] Keep only LlamaEngine
- [ ] Remove ExecuTorchEngine (or keep disabled)
- [ ] Hardcode Qwen 3 1.7B as the only model
- [ ] Remove model selection logic

**Files:**
- `app/src/main/java/com/lelloman/simpleai/llm/LLMEngine.kt` (simplify)
- `app/src/main/java/com/lelloman/simpleai/model/AvailableModels.kt` (simplify to single model)

#### 5.2 Local AI Capability
- [ ] Wire LlamaEngine to CapabilityManager
- [ ] Implement localGenerate() and localChat() AIDL methods

### Phase 6: UI Refactor

#### 6.1 Main Screen
- [ ] Create `CapabilityCard` composable
- [ ] Show download progress, errors, retry
- [ ] Remove all model selection UI
- [ ] Remove generation test UI

**Files:**
- `app/src/main/java/com/lelloman/simpleai/ui/CapabilityCard.kt`
- `app/src/main/java/com/lelloman/simpleai/MainActivity.kt` (simplify)
- `app/src/main/java/com/lelloman/simpleai/MainViewModel.kt` (simplify)

#### 6.2 Translation Language Picker
- [ ] Create language picker screen/dialog
- [ ] Show downloaded vs available languages
- [ ] Download/delete individual languages

**Files:**
- `app/src/main/java/com/lelloman/simpleai/ui/TranslationLanguagesScreen.kt`

### Phase 7: Cleanup

#### 7.1 Remove Unused Code
- [ ] Delete ExecuTorchEngine (or move to disabled/)
- [ ] Delete chat formatters (HermesFormatter, LlamaFormatter, RawFormatter)
- [ ] Delete ChatModels.kt
- [ ] Clean up old AIDL interface references
- [ ] Remove old UI components

#### 7.2 Update SimpleEphem
- [ ] Update AIDL interface copy
- [ ] Update NluClient to use new API
- [ ] Update SimpleAiClient for cloud proxy
- [ ] Add version checking logic

### Phase 8: Testing & Documentation

#### 8.1 Unit Tests
- [ ] Test ApiResponse serialization
- [ ] Test ProtocolHandler version logic
- [ ] Test AdapterCache
- [ ] Test CapabilityManager state transitions

#### 8.2 Integration Tests
- [ ] Test AIDL binding and method calls
- [ ] Test capability download flows
- [ ] Test error scenarios

#### 8.3 Documentation
- [ ] Update README.md
- [ ] Document AIDL interface for client apps
- [ ] Document error codes and handling

---

## Build Configuration

```kotlin
// app/build.gradle.kts
android {
    defaultConfig {
        // Versioning
        buildConfigField("int", "SERVICE_VERSION", "1")       // Bump every release
        buildConfigField("int", "MIN_PROTOCOL_VERSION", "1")  // Bump when dropping old protocol support
        buildConfigField("int", "MAX_PROTOCOL_VERSION", "1")  // Bump when adding new protocol features
    }

    buildTypes {
        debug {
            buildConfigField("String", "CLOUD_LLM_ENDPOINT", "\"https://dev-api.example.com/v1\"")
        }
        release {
            buildConfigField("String", "CLOUD_LLM_ENDPOINT", "\"https://api.example.com/v1\"")
        }
    }
}

dependencies {
    // ML Kit Translation
    implementation("com.google.mlkit:translate:17.0.3")
}
```

---

## Migration Notes

### For Client Apps (SimpleEphem)

1. **Copy new AIDL file** to your project
2. **Add version checking**:
   ```kotlin
   val info = simpleAi.getServiceInfo("v1")
   if (info.serviceVersion < MIN_REQUIRED_VERSION) {
       // Prompt user to update SimpleAI
   }
   ```
3. **Update classify() calls** to pass adapter files every time (SimpleAI caches internally)
4. **Handle new error codes** in responses
5. **Use cloudChat()** instead of local LLM for tool calling

### Breaking Changes

- All AIDL methods now require `protocolVersion` parameter
- All responses are now JSON with `status`/`data`/`error` structure
- Adapter management is now per-request (pass files with classify())
- Model selection removed (fixed models per capability)

---

## Resolved Questions

1. **Adapter management**: Single adapter in memory at a time. Track current adapter state with revertPatch for de-patching when switching adapters.

2. **Translation**: Offline-only for now. Cloud translation may be added later for paying users.

3. **Cloud timeout**: 60 seconds.

4. **Versioning strategy**:
   - `SERVICE_VERSION`: Bump every release
   - `MIN_PROTOCOL_VERSION`: Bump only when dropping support for old protocols
   - `MAX_PROTOCOL_VERSION`: Bump when adding new protocol features
   - Clients pass their required protocol; SimpleAI checks if it's in the supported range
