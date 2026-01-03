# SimpleAI

An Android service that provides AI capabilities to other apps. SimpleAI acts as a capability manager where users opt-in to features and download required models.

## Capabilities

| Capability | Description | Download Required |
|------------|-------------|-------------------|
| **Voice Commands** | NLU intent classification + entity extraction | ~120 MB (XLM-RoBERTa) |
| **Translation** | On-device translation via ML Kit | ~30 MB per language |
| **Cloud AI** | Proxy to cloud LLM endpoint | None (requires client auth) |
| **Local AI** | On-device LLM inference | ~1.3 GB (Qwen 3 1.7B) |

## Requirements

- Android 7.0 (API 24) or higher
- ARM processor (arm64-v8a or armeabi-v7a)
- Storage space depends on capabilities:
  - Voice Commands: ~150 MB
  - Translation: ~30 MB per language (English required as pivot)
  - Local AI: ~1.5 GB

## Installation

### Build from Source

```bash
git clone https://github.com/user/simple-ai.git
cd simple-ai
./gradlew assembleDebug
./gradlew installDebug
```

## AIDL Integration

Client apps bind to SimpleAI via AIDL to use its capabilities.

### 1. Add the AIDL Interface

Copy `ISimpleAI.aidl` to your project:

```
app/src/main/aidl/com/lelloman/simpleai/ISimpleAI.aidl
```

### 2. Bind to the Service

```kotlin
class MyActivity : AppCompatActivity() {

    private var simpleAi: ISimpleAI? = null

    private val serviceConnection = object : ServiceConnection {
        override fun onServiceConnected(name: ComponentName?, service: IBinder?) {
            simpleAi = ISimpleAI.Stub.asInterface(service)
            checkCapabilities()
        }

        override fun onServiceDisconnected(name: ComponentName?) {
            simpleAi = null
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        val intent = Intent().apply {
            setClassName("com.lelloman.simpleai", "com.lelloman.simpleai.service.SimpleAIService")
        }
        bindService(intent, serviceConnection, Context.BIND_AUTO_CREATE)
    }

    override fun onDestroy() {
        super.onDestroy()
        unbindService(serviceConnection)
    }
}
```

### 3. Protocol Versioning

All AIDL methods require a `protocolVersion` parameter. This enables backward compatibility:

```kotlin
const val PROTOCOL_VERSION = 1

// Check service compatibility
val infoJson = simpleAi?.getServiceInfo(PROTOCOL_VERSION)
val info = Json.parseToJsonElement(infoJson).jsonObject

// Check if our protocol is supported
val minProtocol = info["data"]?.jsonObject?.get("minProtocol")?.jsonPrimitive?.int
val maxProtocol = info["data"]?.jsonObject?.get("maxProtocol")?.jsonPrimitive?.int

if (PROTOCOL_VERSION < minProtocol) {
    // Client too old, needs update
} else if (PROTOCOL_VERSION > maxProtocol) {
    // SimpleAI too old, prompt user to update
}
```

---

## API Reference

All methods return JSON strings with this structure:

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
    "code": "ERROR_CODE",
    "message": "Human-readable message",
    "details": { ... }
  }
}
```

### getServiceInfo

Get service version and capabilities status.

```kotlin
val result = simpleAi.getServiceInfo(protocolVersion = 1)
```

**Response:**
```json
{
  "status": "success",
  "protocolVersion": 1,
  "data": {
    "serviceVersion": 1,
    "minProtocol": 1,
    "maxProtocol": 1,
    "capabilities": {
      "voiceCommands": { "status": "ready" },
      "translation": { "status": "ready", "languages": ["en", "it", "fr"] },
      "cloudAi": { "status": "ready" },
      "localAi": { "status": "not_downloaded", "modelSize": 1342177280 }
    }
  }
}
```

**Capability Status Values:**
- `not_downloaded` - Requires download (includes `modelSize`)
- `downloading` - Download in progress (includes `progress` 0.0-1.0)
- `ready` - Available for use
- `error` - Failed (includes `message`, `canRetry`)

---

### Voice Commands (NLU)

#### classify

Classify text using a client-provided LoRA adapter.

```kotlin
// Open file descriptors for adapter files
val patchFd = contentResolver.openFileDescriptor(patchUri, "r")
val headsFd = contentResolver.openFileDescriptor(headsUri, "r")
val tokenizerFd = contentResolver.openFileDescriptor(tokenizerUri, "r")
val configFd = contentResolver.openFileDescriptor(configUri, "r")

try {
    val result = simpleAi.classify(
        protocolVersion = 1,
        text = "remind me to call John tomorrow",
        adapterId = "my-app",
        adapterVersion = "1.0.0",
        patchFd = patchFd,
        headsFd = headsFd,
        tokenizerFd = tokenizerFd,
        configFd = configFd
    )
} finally {
    // Client must close FDs after call returns
    patchFd?.close()
    headsFd?.close()
    tokenizerFd?.close()
    configFd?.close()
}
```

**Response:**
```json
{
  "status": "success",
  "protocolVersion": 1,
  "data": {
    "intent": "add_reminder",
    "intentConfidence": 0.94,
    "slots": {
      "person": ["John"],
      "date": ["tomorrow"]
    }
  }
}
```

**Notes:**
- Adapter files are passed with every request
- SimpleAI tracks the current adapter by (adapterId, adapterVersion)
- If adapter matches, inference runs directly
- If different, SimpleAI swaps adapters automatically

#### clearAdapter

Remove the currently applied adapter and restore the pristine model.

```kotlin
val result = simpleAi.clearAdapter(protocolVersion = 1)
```

---

### Translation

#### translate

Translate text between languages.

```kotlin
val result = simpleAi.translate(
    protocolVersion = 1,
    text = "Hello, how are you?",
    sourceLang = "en",      // or "auto" for detection
    targetLang = "it"
)
```

**Response:**
```json
{
  "status": "success",
  "protocolVersion": 1,
  "data": {
    "translatedText": "Ciao, come stai?",
    "detectedSourceLang": "en"
  }
}
```

#### getTranslationLanguages

Get list of downloaded translation languages.

```kotlin
val result = simpleAi.getTranslationLanguages(protocolVersion = 1)
```

**Response:**
```json
{
  "status": "success",
  "protocolVersion": 1,
  "data": {
    "languages": ["en", "it", "fr", "de"]
  }
}
```

**Supported Languages (59):**

| Code | Language | Code | Language |
|------|----------|------|----------|
| af | Afrikaans | ka | Georgian |
| ar | Arabic | kn | Kannada |
| be | Belarusian | ko | Korean |
| bg | Bulgarian | lt | Lithuanian |
| bn | Bengali | lv | Latvian |
| ca | Catalan | mk | Macedonian |
| cs | Czech | mr | Marathi |
| cy | Welsh | ms | Malay |
| da | Danish | mt | Maltese |
| de | German | nl | Dutch |
| el | Greek | no | Norwegian |
| en | English | pl | Polish |
| eo | Esperanto | pt | Portuguese |
| es | Spanish | ro | Romanian |
| et | Estonian | ru | Russian |
| fa | Persian | sk | Slovak |
| fi | Finnish | sl | Slovenian |
| fr | French | sq | Albanian |
| ga | Irish | sv | Swedish |
| gl | Galician | sw | Swahili |
| gu | Gujarati | ta | Tamil |
| he | Hebrew | te | Telugu |
| hi | Hindi | th | Thai |
| hr | Croatian | tl | Tagalog |
| ht | Haitian Creole | tr | Turkish |
| hu | Hungarian | uk | Ukrainian |
| id | Indonesian | ur | Urdu |
| is | Icelandic | vi | Vietnamese |
| it | Italian | zh | Chinese |
| ja | Japanese | | |

---

### Cloud AI

#### cloudChat

Send chat request to cloud LLM endpoint.

```kotlin
val messages = """[
    {"role": "user", "content": "What is the capital of France?"}
]"""

val result = simpleAi.cloudChat(
    protocolVersion = 1,
    messagesJson = messages,
    toolsJson = null,           // optional tool definitions
    systemPrompt = null,        // optional system prompt
    authToken = "your-api-key"  // client's auth token
)
```

**Response:**
```json
{
  "status": "success",
  "protocolVersion": 1,
  "data": {
    "role": "assistant",
    "content": "The capital of France is Paris.",
    "finishReason": "stop",
    "usage": {
      "promptTokens": 15,
      "completionTokens": 8,
      "totalTokens": 23
    }
  }
}
```

**With Tool Calls:**
```json
{
  "status": "success",
  "protocolVersion": 1,
  "data": {
    "role": "assistant",
    "content": null,
    "finishReason": "tool_calls",
    "toolCalls": [
      {
        "id": "call_abc123",
        "type": "function",
        "function": {
          "name": "get_weather",
          "arguments": "{\"location\": \"Paris\"}"
        }
      }
    ]
  }
}
```

---

### Local AI

#### localGenerate

Generate text using the local LLM.

```kotlin
val result = simpleAi.localGenerate(
    protocolVersion = 1,
    prompt = "Explain quantum computing in simple terms:",
    maxTokens = 256,
    temperature = 0.7f
)
```

**Response:**
```json
{
  "status": "success",
  "protocolVersion": 1,
  "data": {
    "text": "Quantum computing uses quantum mechanics principles..."
  }
}
```

#### localChat

Chat using local LLM with optional tool support.

```kotlin
val messages = """[
    {"role": "user", "content": "Hello!"}
]"""

val result = simpleAi.localChat(
    protocolVersion = 1,
    messagesJson = messages,
    toolsJson = null,
    systemPrompt = "You are a helpful assistant."
)
```

**Response:**
```json
{
  "status": "success",
  "protocolVersion": 1,
  "data": {
    "role": "assistant",
    "content": "Hello! How can I help you today?"
  }
}
```

---

## Error Codes

| Code | Description | Typical Action |
|------|-------------|----------------|
| `VERSION_TOO_OLD` | SimpleAI needs update | Prompt user to update SimpleAI |
| `UNSUPPORTED_PROTOCOL` | Client needs update | Update your app |
| `CAPABILITY_NOT_READY` | Capability not downloaded | Direct user to SimpleAI to download |
| `CAPABILITY_DOWNLOADING` | Download in progress | Show progress, retry later |
| `CAPABILITY_ERROR` | Capability failed to initialize | Check `message`, maybe retry |
| `ADAPTER_LOAD_FAILED` | LoRA adapter failed to load | Check adapter files |
| `INVALID_REQUEST` | Malformed request | Fix request parameters |
| `CLOUD_AUTH_FAILED` | Cloud rejected auth token | Check/refresh auth token |
| `CLOUD_UNAVAILABLE` | Cloud endpoint unreachable | Check network, retry |
| `TRANSLATION_LANGUAGE_NOT_AVAILABLE` | Language not downloaded | Direct user to download language |
| `INTERNAL_ERROR` | Unexpected error | Report bug, retry |

**Example Error Handling:**

```kotlin
val result = simpleAi.translate(1, "Hello", "en", "it")
val json = Json.parseToJsonElement(result).jsonObject

when (json["status"]?.jsonPrimitive?.content) {
    "success" -> {
        val translated = json["data"]?.jsonObject?.get("translatedText")?.jsonPrimitive?.content
        // Use translated text
    }
    "error" -> {
        val error = json["error"]?.jsonObject
        val code = error?.get("code")?.jsonPrimitive?.content
        val message = error?.get("message")?.jsonPrimitive?.content

        when (code) {
            "CAPABILITY_NOT_READY" -> showDownloadPrompt()
            "TRANSLATION_LANGUAGE_NOT_AVAILABLE" -> showLanguageDownloadPrompt()
            else -> showError(message)
        }
    }
}
```

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
│  └─────────────┴─────────────┴─────────────┴──────────────┘ │
│  ┌────────────────────────────────────────────────────────┐ │
│  │              Capability Manager                         │ │
│  │    (download status, progress, errors, retry)          │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
app/src/main/
├── aidl/com/lelloman/simpleai/
│   └── ISimpleAI.aidl              # AIDL interface
├── java/com/lelloman/simpleai/
│   ├── MainActivity.kt             # Compose UI
│   ├── api/
│   │   ├── ApiResponse.kt          # Response types
│   │   ├── ErrorCode.kt            # Error codes enum
│   │   └── ProtocolHandler.kt      # Version handling
│   ├── capability/
│   │   ├── Capability.kt           # Capability data classes
│   │   ├── CapabilityId.kt         # Capability identifiers
│   │   ├── CapabilityManager.kt    # State management
│   │   └── CapabilityStatus.kt     # Status sealed class
│   ├── cloud/
│   │   └── CloudLLMClient.kt       # Cloud API client
│   ├── download/
│   │   └── ModelDownloadManager.kt # Model downloads
│   ├── llm/
│   │   ├── LLMEngine.kt            # LLM interface
│   │   └── LlamaHelperWrapper.kt   # llama.cpp wrapper
│   ├── nlu/
│   │   ├── NLUEngine.kt            # NLU interface
│   │   ├── OnnxNLUEngine.kt        # ONNX implementation
│   │   └── LoraPatcher.kt          # LoRA patching
│   ├── service/
│   │   └── SimpleAIService.kt      # Main foreground service
│   ├── translation/
│   │   ├── Language.kt             # Language definitions
│   │   └── TranslationManager.kt   # ML Kit integration
│   └── ui/
│       ├── CapabilitiesScreen.kt   # Main UI
│       ├── CapabilitiesViewModel.kt
│       ├── CapabilityCard.kt       # Capability card component
│       └── TranslationLanguagesScreen.kt
└── res/
```

---

## Development

### Building

```bash
# Debug build
./gradlew assembleDebug

# Release build (requires signing config)
./gradlew assembleRelease

# Install on device
./gradlew installDebug
```

### Testing

```bash
# Run all unit tests
./gradlew testDebugUnitTest

# Run specific test class
./gradlew testDebugUnitTest --tests "com.lelloman.simpleai.api.ApiResponseTest"

# Run tests with coverage
./gradlew testDebugUnitTest jacocoTestReport
```

### Build Configuration

Protocol versioning is configured in `app/build.gradle.kts`:

```kotlin
buildConfigField("int", "SERVICE_VERSION", "1")       // Bump every release
buildConfigField("int", "MIN_PROTOCOL_VERSION", "1")  // Bump when dropping old protocols
buildConfigField("int", "MAX_PROTOCOL_VERSION", "1")  // Bump when adding new features
```

---

## Tech Stack

- **Language**: Kotlin 2.2
- **UI**: Jetpack Compose with Material 3
- **LLM**: llama.cpp via llamacpp-kotlin
- **NLU**: ONNX Runtime with XLM-RoBERTa
- **Translation**: Google ML Kit
- **Networking**: OkHttp 4.12
- **Serialization**: kotlinx.serialization
- **Async**: Kotlin Coroutines + Flow
- **Testing**: JUnit 4, MockK, Turbine

---

## License

TBD
