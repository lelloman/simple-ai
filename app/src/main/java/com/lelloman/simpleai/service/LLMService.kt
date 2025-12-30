package com.lelloman.simpleai.service

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Context
import android.content.Intent
import android.os.Build
import android.os.IBinder
import android.os.ParcelFileDescriptor
import androidx.core.app.NotificationCompat
import com.lelloman.simpleai.ILLMService
import com.lelloman.simpleai.R
import com.lelloman.simpleai.chat.ChatFormatter
import com.lelloman.simpleai.chat.ChatMessage
import com.lelloman.simpleai.chat.ChatResponse
import com.lelloman.simpleai.chat.FunctionDefinition
import com.lelloman.simpleai.chat.ToolCall
import com.lelloman.simpleai.chat.ToolDefinition
import com.lelloman.simpleai.download.DownloadState
import com.lelloman.simpleai.download.ModelDownloadManager
import com.lelloman.simpleai.llm.ExecuTorchEngine
import com.lelloman.simpleai.llm.GenerationParams
import com.lelloman.simpleai.llm.LLMEngine
import com.lelloman.simpleai.llm.LlamaEngine
import com.lelloman.simpleai.model.AvailableModel
import com.lelloman.simpleai.model.AvailableModels
import com.lelloman.simpleai.model.EngineType
import com.lelloman.simpleai.nlu.NLUEngine
import com.lelloman.simpleai.nlu.OnnxNLUEngine
import com.lelloman.simpleai.translation.Language
import android.util.Log
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.buildJsonArray
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.put
import kotlinx.serialization.json.JsonPrimitive

class LLMService : Service() {

    companion object {
        private const val NOTIFICATION_CHANNEL_ID = "llm_service_channel"
        private const val NOTIFICATION_ID = 1
        private const val PREFS_NAME = "simple_ai_prefs"
        private const val KEY_SELECTED_MODEL = "selected_model_id"
        const val ACTION_STATUS_UPDATE = "com.lelloman.simpleai.STATUS_UPDATE"
        const val EXTRA_STATUS = "status"
        const val EXTRA_PROGRESS = "progress"
        const val EXTRA_DOWNLOADED_BYTES = "downloaded_bytes"
        const val EXTRA_TOTAL_BYTES = "total_bytes"
    }

    sealed class ServiceStatus {
        data object Initializing : ServiceStatus()
        data class Downloading(
            val progress: Float,
            val downloadedBytes: Long = 0,
            val totalBytes: Long = 0
        ) : ServiceStatus()
        data object Loading : ServiceStatus()
        data object Ready : ServiceStatus()
        data class Error(val message: String) : ServiceStatus()
    }

    private val serviceScope = CoroutineScope(SupervisorJob() + Dispatchers.Main)

    private lateinit var downloadManager: ModelDownloadManager
    private var llmEngine: LLMEngine? = null
    private var llamaEngine: LlamaEngine? = null
    private var execuTorchEngine: ExecuTorchEngine? = null
    private var currentEngineType: EngineType? = null

    private val _status = MutableStateFlow<ServiceStatus>(ServiceStatus.Initializing)
    val status: StateFlow<ServiceStatus> = _status.asStateFlow()

    private var currentModel: AvailableModel? = null
    private var chatFormatter: ChatFormatter? = null

    // NLU Engine for classification
    private var nluEngine: NLUEngine? = null

    private val json = Json {
        ignoreUnknownKeys = true
        encodeDefaults = true
    }

    private val binder = object : ILLMService.Stub() {
        override fun isReady(): Boolean {
            return _status.value is ServiceStatus.Ready && llmEngine?.isLoaded == true
        }

        override fun getStatus(): String {
            return when (val s = _status.value) {
                is ServiceStatus.Initializing -> "initializing"
                is ServiceStatus.Downloading -> "downloading"
                is ServiceStatus.Loading -> "loading"
                is ServiceStatus.Ready -> "ready"
                is ServiceStatus.Error -> "error: ${s.message}"
            }
        }

        override fun getAvailableModels(): String = buildAvailableModelsJson()

        override fun getCurrentModel(): String? = buildCurrentModelJson()

        override fun setModel(modelId: String): String = switchModel(modelId)

        override fun generate(prompt: String): String {
            if (!isReady) {
                return "Error: Model not ready. Status: ${getStatus()}"
            }
            val engine = llmEngine ?: return "Error: No engine available"
            // Run generation on IO thread to avoid ANR
            return runBlocking(Dispatchers.IO) {
                engine.generate(prompt).getOrElse { "Error: ${it.message}" }
            }
        }

        override fun generateWithParams(prompt: String, maxTokens: Int, temperature: Float): String {
            if (!isReady) {
                return "Error: Model not ready. Status: ${getStatus()}"
            }
            val engine = llmEngine ?: return "Error: No engine available"
            val params = GenerationParams(
                maxTokens = if (maxTokens > 0) maxTokens else 512,
                temperature = temperature.coerceIn(0f, 2f)
            )
            // Run generation on IO thread to avoid ANR
            return runBlocking(Dispatchers.IO) {
                engine.generate(prompt, params).getOrElse { "Error: ${it.message}" }
            }
        }

        override fun chat(messagesJson: String, toolsJson: String?, systemPrompt: String?): String {
            return handleChat(messagesJson, toolsJson, systemPrompt)
        }

        override fun translate(text: String, sourceLanguage: String, targetLanguage: String): String {
            if (!isReady) {
                return "Error: Model not ready. Status: ${getStatus()}"
            }

            if (!Language.isValidCode(sourceLanguage)) {
                return "Error: Invalid source language code: $sourceLanguage"
            }
            if (!Language.isValidCode(targetLanguage) || targetLanguage == Language.AUTO_DETECT) {
                return "Error: Invalid target language code: $targetLanguage"
            }

            val sourceLang = if (sourceLanguage == Language.AUTO_DETECT) null else Language.fromCode(sourceLanguage)
            val targetLang = Language.fromCode(targetLanguage)!!

            val engine = llmEngine ?: return "Error: No engine available"
            val prompt = buildTranslationPrompt(text, sourceLang, targetLang)
            val params = GenerationParams(
                maxTokens = (text.length * 3).coerceIn(256, 2048),
                temperature = 0.3f
            )

            // Run generation on IO thread to avoid ANR
            return runBlocking(Dispatchers.IO) {
                engine.generate(prompt, params)
                    .map { extractTranslation(it) }
                    .getOrElse { "Error: ${it.message}" }
            }
        }

        override fun getSupportedLanguages(): String = Language.toJsonArray().toString()

        // ==================== Classification (NLU) ====================

        override fun applyClassificationAdapter(
            adapterId: String,
            adapterVersion: String,
            patchFd: ParcelFileDescriptor,
            headsFd: ParcelFileDescriptor,
            tokenizerFd: ParcelFileDescriptor,
            configFd: ParcelFileDescriptor
        ): String {
            return handleApplyAdapter(adapterId, adapterVersion, patchFd, headsFd, tokenizerFd, configFd)
        }

        override fun removeClassificationAdapter(): String {
            return handleRemoveAdapter()
        }

        override fun classify(text: String, adapterId: String): String {
            return handleClassify(text, adapterId)
        }

        override fun getCurrentClassificationAdapter(): String? {
            return buildCurrentAdapterJson()
        }

        override fun isClassificationReady(): Boolean {
            return nluEngine?.isReady == true
        }
    }

    // ==================== Model Discovery Helpers ====================

    private fun buildAvailableModelsJson(): String {
        val modelsArray = buildJsonArray {
            AvailableModels.ALL.forEach { model ->
                add(buildJsonObject {
                    put("id", model.id)
                    put("name", model.name)
                    put("description", model.description)
                    put("sizeMb", model.sizeMb)
                    put("supportsTools", model.supportsTools)
                    put("format", model.format.name)
                    put("downloaded", downloadManager.isModelDownloaded(model))
                })
            }
        }
        return modelsArray.toString()
    }

    private fun buildCurrentModelJson(): String? {
        val model = currentModel ?: return null
        return buildJsonObject {
            put("id", model.id)
            put("name", model.name)
            put("description", model.description)
            put("sizeMb", model.sizeMb)
            put("supportsTools", model.supportsTools)
            put("format", model.format.name)
        }.toString()
    }

    private fun switchModel(modelId: String): String {
        val model = AvailableModels.findById(modelId)
            ?: return "error: Unknown model ID: $modelId"

        getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            .edit()
            .putString(KEY_SELECTED_MODEL, modelId)
            .apply()

        // Unload current engine
        unloadCurrentEngine()
        currentModel = null
        chatFormatter = null

        serviceScope.launch {
            if (downloadManager.isModelDownloaded(model)) {
                loadModel(model)
            } else {
                downloadModel(model)
            }
        }

        return "ok"
    }

    private fun unloadCurrentEngine() {
        llmEngine?.unloadModel()
        llamaEngine?.release()
        execuTorchEngine?.release()
        llmEngine = null
        llamaEngine = null
        execuTorchEngine = null
        currentEngineType = null
    }

    // ==================== Chat Helpers ====================

    private fun handleChat(messagesJson: String, toolsJson: String?, systemPrompt: String?): String {
        val engine = llmEngine
        if (_status.value !is ServiceStatus.Ready || engine == null || !engine.isLoaded) {
            return errorResponse("Model not ready")
        }

        val formatter = chatFormatter ?: return errorResponse("No formatter available")

        return try {
            val messages = parseMessages(messagesJson)
            val tools = parseTools(toolsJson)

            if (tools.isNotEmpty() && !formatter.supportsTools) {
                return errorResponse("Current model does not support tool calling. Switch to Llama 3.2 or Qwen 3.")
            }

            val prompt = formatter.formatPrompt(messages, tools, systemPrompt)
            val params = GenerationParams(maxTokens = 1024, temperature = 0.7f)
            // Run generation on IO thread to avoid ANR
            val rawResponse = runBlocking(Dispatchers.IO) {
                engine.generate(prompt, params).getOrThrow()
            }
            val parsed = formatter.parseResponse(rawResponse)

            formatChatResponse(parsed)
        } catch (e: Exception) {
            errorResponse("Chat error: ${e.message}")
        }
    }

    private fun parseMessages(messagesJson: String): List<ChatMessage> {
        val messagesElement = json.parseToJsonElement(messagesJson)
        return messagesElement.jsonArray.map { msgElement ->
            val msgObj = msgElement.jsonObject
            ChatMessage(
                role = msgObj["role"]?.jsonPrimitive?.content ?: "user",
                content = msgObj["content"]?.jsonPrimitive?.content ?: "",
                toolCallId = msgObj["toolCallId"]?.jsonPrimitive?.content,
                toolCalls = msgObj["toolCalls"]?.jsonArray?.map { tcElement ->
                    val tcObj = tcElement.jsonObject
                    ToolCall(
                        id = tcObj["id"]?.jsonPrimitive?.content ?: "",
                        name = tcObj["name"]?.jsonPrimitive?.content ?: "",
                        arguments = tcObj["arguments"]?.jsonObject ?: JsonObject(emptyMap())
                    )
                }
            )
        }
    }

    private fun parseTools(toolsJson: String?): List<ToolDefinition> {
        if (toolsJson.isNullOrBlank()) return emptyList()

        val toolsElement = json.parseToJsonElement(toolsJson)
        return toolsElement.jsonArray.map { toolElement ->
            val toolObj = toolElement.jsonObject
            val funcObj = toolObj["function"]?.jsonObject
                ?: throw IllegalArgumentException("Tool missing 'function' field")
            ToolDefinition(
                type = toolObj["type"]?.jsonPrimitive?.content ?: "function",
                function = FunctionDefinition(
                    name = funcObj["name"]?.jsonPrimitive?.content ?: "",
                    description = funcObj["description"]?.jsonPrimitive?.content ?: "",
                    parameters = funcObj["parameters"]?.jsonObject ?: JsonObject(emptyMap())
                )
            )
        }
    }

    private fun formatChatResponse(parsed: ChatResponse): String {
        return when (parsed) {
            is ChatResponse.Text -> buildJsonObject {
                put("type", "text")
                put("content", parsed.content)
            }.toString()

            is ChatResponse.ToolCalls -> buildJsonObject {
                put("type", "tool_calls")
                put("toolCalls", buildToolCallsArray(parsed.toolCalls))
            }.toString()

            is ChatResponse.Mixed -> buildJsonObject {
                put("type", "mixed")
                put("content", parsed.content)
                put("toolCalls", buildToolCallsArray(parsed.toolCalls))
            }.toString()

            is ChatResponse.Error -> errorResponse(parsed.message)
        }
    }

    private fun buildToolCallsArray(toolCalls: List<ToolCall>) = buildJsonArray {
        toolCalls.forEach { tc ->
            add(buildJsonObject {
                put("id", tc.id)
                put("name", tc.name)
                put("arguments", tc.arguments)
            })
        }
    }

    private fun errorResponse(message: String): String {
        return buildJsonObject {
            put("type", "error")
            put("message", message)
        }.toString()
    }

    // ==================== Classification Helpers ====================

    private fun handleApplyAdapter(
        adapterId: String,
        adapterVersion: String,
        patchFd: ParcelFileDescriptor,
        headsFd: ParcelFileDescriptor,
        tokenizerFd: ParcelFileDescriptor,
        configFd: ParcelFileDescriptor
    ): String {
        val engine = nluEngine as? OnnxNLUEngine
        if (engine == null) {
            return "error: NLU engine not initialized"
        }

        return runBlocking(Dispatchers.IO) {
            engine.applyAdapter(adapterId, adapterVersion, patchFd, headsFd, tokenizerFd, configFd).fold(
                onSuccess = { "ok" },
                onFailure = { "error: ${it.message}" }
            )
        }
    }

    private fun handleRemoveAdapter(): String {
        val engine = nluEngine as? OnnxNLUEngine
        if (engine == null) {
            return "error: NLU engine not initialized"
        }

        return runBlocking(Dispatchers.IO) {
            engine.removeAdapter().fold(
                onSuccess = { "ok" },
                onFailure = { "error: ${it.message}" }
            )
        }
    }

    private fun handleClassify(text: String, adapterId: String): String {
        val engine = nluEngine
        if (engine == null || !engine.isReady) {
            return buildJsonObject {
                put("error", "NLU engine not ready")
            }.toString()
        }

        return runBlocking(Dispatchers.IO) {
            engine.classify(text, adapterId).fold(
                onSuccess = { result ->
                    buildJsonObject {
                        put("intent", result.intent)
                        put("intent_confidence", result.intentConfidence)
                        put("slots", buildJsonObject {
                            result.slots.forEach { (slotType, values) ->
                                put(slotType, buildJsonArray {
                                    values.forEach { add(JsonPrimitive(it)) }
                                })
                            }
                        })
                        put("raw_slot_labels", buildJsonArray {
                            result.rawSlotLabels.forEach { add(JsonPrimitive(it)) }
                        })
                    }.toString()
                },
                onFailure = { error ->
                    buildJsonObject {
                        put("error", error.message ?: "Classification failed")
                    }.toString()
                }
            )
        }
    }

    private fun buildCurrentAdapterJson(): String? {
        val engine = nluEngine ?: return null
        val adapter = engine.adapters.firstOrNull() ?: return null
        return buildJsonObject {
            put("id", adapter.id)
            put("version", adapter.version)
            put("intents", buildJsonArray { adapter.intents.forEach { add(JsonPrimitive(it)) } })
            put("slot_types", buildJsonArray { adapter.slotTypes.forEach { add(JsonPrimitive(it)) } })
        }.toString()
    }

    // ==================== Translation Helpers ====================

    private fun buildTranslationPrompt(text: String, source: Language?, target: Language): String {
        val sourceDesc = source?.displayName ?: "the source language"
        return """Translate the following text from $sourceDesc to ${target.displayName}.
Only output the translation, nothing else.

Text to translate:
$text

Translation:"""
    }

    private fun extractTranslation(response: String): String {
        return response.trim()
            .removePrefix("Translation:")
            .removePrefix("translation:")
            .trim()
    }

    // ==================== Service Lifecycle ====================

    override fun onCreate() {
        super.onCreate()
        downloadManager = ModelDownloadManager(this)
        // Engine is lazily initialized based on model type in loadModel()

        createNotificationChannel()
        startForeground(NOTIFICATION_ID, createNotification("Initializing..."))

        initialize()
        initializeNLU()
    }

    private fun initializeNLU() {
        serviceScope.launch(Dispatchers.IO) {
            try {
                Log.i("LLMService", "Initializing NLU engine...")
                val engine = OnnxNLUEngine(this@LLMService)
                engine.initialize().fold(
                    onSuccess = {
                        nluEngine = engine
                        Log.i("LLMService", "NLU engine ready with ${engine.adapters.size} adapters")
                    },
                    onFailure = { e ->
                        Log.e("LLMService", "Failed to initialize NLU engine", e)
                    }
                )
            } catch (e: Exception) {
                Log.e("LLMService", "Error initializing NLU", e)
            }
        }
    }

    override fun onBind(intent: Intent?): IBinder = binder

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int = START_STICKY

    override fun onDestroy() {
        super.onDestroy()
        unloadCurrentEngine()
        nluEngine?.release()
        nluEngine = null
        serviceScope.cancel()
    }

    private fun getSelectedModel(): AvailableModel {
        val prefs = getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        val selectedId = prefs.getString(KEY_SELECTED_MODEL, AvailableModels.DEFAULT_MODEL_ID)
            ?: AvailableModels.DEFAULT_MODEL_ID
        return AvailableModels.findById(selectedId) ?: AvailableModels.getDefault()
    }

    private fun initialize() {
        serviceScope.launch {
            val model = getSelectedModel()
            Log.i("LLMService", "Initializing with model: ${model.id} (${model.name}), engineType: ${model.engineType}")
            val isDownloaded = downloadManager.isModelDownloaded(model)
            Log.i("LLMService", "Model downloaded: $isDownloaded")
            if (isDownloaded) {
                loadModel(model)
            } else {
                Log.i("LLMService", "Starting download for model: ${model.id}")
                downloadModel(model)
            }
        }
    }

    private suspend fun downloadModel(model: AvailableModel) {
        try {
            Log.i("LLMService", "downloadModel starting collect for ${model.id}")
            downloadManager.downloadModel(model).collect { state ->
                Log.i("LLMService", "Download state: $state")
            when (state) {
                is DownloadState.Idle -> {
                    _status.value = ServiceStatus.Downloading(0f)
                    updateNotification("Starting download...")
                }
                is DownloadState.Downloading -> {
                    _status.value = ServiceStatus.Downloading(
                        progress = state.progress,
                        downloadedBytes = state.downloadedBytes,
                        totalBytes = state.totalBytes
                    )
                    val percent = (state.progress * 100).toInt()
                    val mbDownloaded = state.downloadedBytes / (1024 * 1024)
                    val mbTotal = state.totalBytes / (1024 * 1024)
                    updateNotification("Downloading: $percent% ($mbDownloaded/$mbTotal MB)")
                    broadcastStatus()
                }
                is DownloadState.Completed -> loadModel(model)
                is DownloadState.Error -> {
                    _status.value = ServiceStatus.Error(state.message)
                    updateNotification("Error: ${state.message}")
                    broadcastStatus()
                }
            }
            }
        } catch (e: Exception) {
            Log.e("LLMService", "Error in downloadModel", e)
            _status.value = ServiceStatus.Error("Download error: ${e.message}")
            updateNotification("Error: ${e.message}")
        }
    }

    private fun loadModel(model: AvailableModel) {
        _status.value = ServiceStatus.Loading
        updateNotification("Loading ${model.name}...")
        broadcastStatus()

        serviceScope.launch(Dispatchers.IO) {
            // Create the appropriate engine based on model type
            val engine = getOrCreateEngine(model)

            val modelFile = downloadManager.getModelFile(model)
            engine.loadModel(modelFile).fold(
                onSuccess = {
                    currentModel = model
                    chatFormatter = ChatFormatter.forFormat(model.format)
                    _status.value = ServiceStatus.Ready
                    updateNotification("Ready - ${model.name}")
                    broadcastStatus()
                },
                onFailure = {
                    _status.value = ServiceStatus.Error(it.message ?: "Failed to load model")
                    updateNotification("Error loading model")
                    broadcastStatus()
                }
            )
        }
    }

    private fun getOrCreateEngine(model: AvailableModel): LLMEngine {
        // If switching engine types, unload the current one
        if (currentEngineType != null && currentEngineType != model.engineType) {
            unloadCurrentEngine()
        }

        return when (model.engineType) {
            EngineType.EXECUTORCH -> {
                if (execuTorchEngine == null) {
                    execuTorchEngine = ExecuTorchEngine()
                }
                llmEngine = execuTorchEngine
                currentEngineType = EngineType.EXECUTORCH
                execuTorchEngine!!
            }
            EngineType.LLAMA_CPP -> {
                if (llamaEngine == null) {
                    llamaEngine = LlamaEngine(this@LLMService)
                }
                llmEngine = llamaEngine
                currentEngineType = EngineType.LLAMA_CPP
                llamaEngine!!
            }
        }
    }

    private fun broadcastStatus() {
        val intent = Intent(ACTION_STATUS_UPDATE).apply {
            setPackage(packageName)  // Explicit broadcast for Android 14+
            putExtra(EXTRA_STATUS, binder.status)
            when (val s = _status.value) {
                is ServiceStatus.Downloading -> {
                    putExtra(EXTRA_PROGRESS, s.progress)
                    putExtra(EXTRA_DOWNLOADED_BYTES, s.downloadedBytes)
                    putExtra(EXTRA_TOTAL_BYTES, s.totalBytes)
                }
                else -> {}
            }
        }
        Log.d("LLMService", "Broadcasting status: ${binder.status}")
        sendBroadcast(intent)
    }

    private fun createNotificationChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                NOTIFICATION_CHANNEL_ID,
                "LLM Service",
                NotificationManager.IMPORTANCE_LOW
            ).apply { description = "Shows the status of the local LLM service" }
            getSystemService(NotificationManager::class.java).createNotificationChannel(channel)
        }
    }

    private fun createNotification(text: String): Notification {
        return NotificationCompat.Builder(this, NOTIFICATION_CHANNEL_ID)
            .setContentTitle("SimpleAI")
            .setContentText(text)
            .setSmallIcon(R.drawable.ic_notification)
            .setOngoing(true)
            .build()
    }

    private fun updateNotification(text: String) {
        getSystemService(NotificationManager::class.java)
            .notify(NOTIFICATION_ID, createNotification(text))
    }
}
