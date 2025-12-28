# SimpleAI

An Android application that runs a local Large Language Model (LLM) on-device, providing AI text generation and translation capabilities without requiring an internet connection.

## Features

- **Local LLM Inference** - Runs Qwen 3 1.7B model entirely on-device using llama.cpp
- **Text Generation** - Generate responses with configurable parameters (max tokens, temperature)
- **Translation** - Translate between 18 languages with auto-detect support
- **AIDL Service** - Expose LLM capabilities to other apps via Android IPC
- **Automatic Model Download** - Downloads and caches the model on first launch (~1.3GB)

## Requirements

- Android 7.0 (API 24) or higher
- ARM processor (arm64-v8a or armeabi-v7a)
- ~1.5GB free storage space for the model
- ~2GB RAM recommended

## Installation

### Build from Source

```bash
# Clone the repository
git clone https://github.com/user/simple-ai.git
cd simple-ai

# Build debug APK
./gradlew assembleDebug

# Install on connected device
./gradlew installDebug
```

### Install APK

1. Download the APK from releases
2. Enable "Install from unknown sources" in device settings
3. Open the APK file to install

## Usage

### First Launch

1. Open SimpleAI - the app will start downloading the Qwen 3 1.7B model (~1.3GB)
2. Wait for download and model loading to complete
3. Once ready, you can test generation using the built-in UI

### Service States

The service goes through these states:
- **Initializing** - Service starting up
- **Downloading** - Model being downloaded (shows progress)
- **Loading** - Model being loaded into memory
- **Ready** - Model loaded and ready for inference
- **Error** - Something went wrong (check logs)

## AIDL Integration

Other Android apps can bind to SimpleAI's LLM service to use its capabilities.

### 1. Add the AIDL Interface

Copy `ILLMService.aidl` to your project at:
```
app/src/main/aidl/com/lelloman/simpleai/ILLMService.aidl
```

```aidl
package com.lelloman.simpleai;

interface ILLMService {
    boolean isReady();
    String getStatus();
    String generate(String prompt);
    String generateWithParams(String prompt, int maxTokens, float temperature);
    String translate(String text, String sourceLanguage, String targetLanguage);
    String getSupportedLanguages();
    String getModelInfo();
}
```

### 2. Bind to the Service

```kotlin
class MyActivity : AppCompatActivity() {

    private var llmService: ILLMService? = null

    private val serviceConnection = object : ServiceConnection {
        override fun onServiceConnected(name: ComponentName?, service: IBinder?) {
            llmService = ILLMService.Stub.asInterface(service)
        }

        override fun onServiceDisconnected(name: ComponentName?) {
            llmService = null
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Bind to SimpleAI service
        val intent = Intent().apply {
            setClassName("com.lelloman.simpleai", "com.lelloman.simpleai.service.LLMService")
        }
        bindService(intent, serviceConnection, Context.BIND_AUTO_CREATE)
    }

    override fun onDestroy() {
        super.onDestroy()
        unbindService(serviceConnection)
    }
}
```

### 3. Use the API

```kotlin
// Check if ready
if (llmService?.isReady() == true) {

    // Generate text
    val response = llmService?.generate("Explain quantum computing in simple terms")

    // Generate with custom parameters
    val response2 = llmService?.generateWithParams(
        "Write a haiku about Android",
        maxTokens = 100,
        temperature = 0.8f
    )

    // Translate text
    val translated = llmService?.translate(
        "Hello, how are you?",
        "en",    // source language
        "es"     // target language (Spanish)
    )

    // Auto-detect source language
    val autoTranslated = llmService?.translate(
        "Bonjour le monde",
        "auto",  // auto-detect
        "en"     // translate to English
    )

    // Get supported languages (returns JSON)
    val languages = llmService?.getSupportedLanguages()

    // Get model info (returns JSON)
    val modelInfo = llmService?.getModelInfo()
}
```

### Supported Languages

| Code | Language |
|------|----------|
| en | English |
| es | Spanish |
| fr | French |
| de | German |
| it | Italian |
| pt | Portuguese |
| nl | Dutch |
| pl | Polish |
| ru | Russian |
| uk | Ukrainian |
| zh | Chinese |
| ja | Japanese |
| ko | Korean |
| ar | Arabic |
| hi | Hindi |
| tr | Turkish |
| vi | Vietnamese |
| th | Thai |
| id | Indonesian |

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      Client Apps                         │
│                    (via AIDL binding)                    │
└─────────────────────┬───────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────┐
│                    LLMService                            │
│              (Foreground Service)                        │
│  ┌─────────────────┬─────────────────┐                  │
│  │  ILLMService    │  Status         │                  │
│  │  (AIDL Binder)  │  Broadcasts     │                  │
│  └────────┬────────┴────────┬────────┘                  │
└───────────┼─────────────────┼───────────────────────────┘
            │                 │
┌───────────▼─────────────────▼───────────────────────────┐
│                    LLMEngine                             │
│            (LlamaEngine / StubEngine)                    │
│  ┌─────────────────────────────────────┐                │
│  │  llama.cpp (native inference)       │                │
│  └─────────────────────────────────────┘                │
└─────────────────────────────────────────────────────────┘
            │
┌───────────▼─────────────────────────────────────────────┐
│              ModelDownloadManager                        │
│         (Downloads model from HuggingFace)               │
└─────────────────────────────────────────────────────────┘
```

## Development

### Project Structure

```
app/src/main/
├── aidl/
│   └── com/lelloman/simpleai/ILLMService.aidl
├── java/com/lelloman/simpleai/
│   ├── MainActivity.kt          # Compose UI
│   ├── MainViewModel.kt         # UI state management
│   ├── service/
│   │   └── LLMService.kt        # Foreground service with AIDL
│   ├── llm/
│   │   ├── LLMEngine.kt         # LLM interface and implementations
│   │   └── LlamaHelperWrapper.kt
│   ├── download/
│   │   └── ModelDownloadManager.kt
│   ├── translation/
│   │   └── Language.kt
│   └── util/
│       └── Logger.kt
└── res/
```

### Building

```bash
# Debug build
./gradlew assembleDebug

# Release build
./gradlew assembleRelease

# Run tests
./gradlew testDebugUnitTest
```

### Running Tests

```bash
# Run all unit tests
./gradlew testDebugUnitTest

# Run specific test class
./gradlew testDebugUnitTest --tests "LanguageTest"
```

## Tech Stack

- **Language**: Kotlin 2.2
- **UI**: Jetpack Compose with Material 3
- **LLM**: llama.cpp via [llamacpp-kotlin](https://github.com/anthropics/llamacpp-kotlin)
- **Model**: Qwen 3 1.7B (Q4_K_M quantization)
- **Networking**: OkHttp 4.12
- **Async**: Kotlin Coroutines
- **Testing**: JUnit 4, MockK, Turbine

## License

TBD
