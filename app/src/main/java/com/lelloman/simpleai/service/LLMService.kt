package com.lelloman.simpleai.service

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Intent
import android.os.Build
import android.os.IBinder
import androidx.core.app.NotificationCompat
import com.lelloman.simpleai.ILLMService
import com.lelloman.simpleai.R
import com.lelloman.simpleai.download.DefaultModel
import com.lelloman.simpleai.download.DownloadState
import com.lelloman.simpleai.download.ModelDownloadManager
import com.lelloman.simpleai.llm.GenerationParams
import com.lelloman.simpleai.llm.LLMEngine
import com.lelloman.simpleai.llm.LlamaEngine
import com.lelloman.simpleai.translation.Language
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import org.json.JSONObject

class LLMService : Service() {

    companion object {
        private const val NOTIFICATION_CHANNEL_ID = "llm_service_channel"
        private const val NOTIFICATION_ID = 1
        const val ACTION_STATUS_UPDATE = "com.lelloman.simpleai.STATUS_UPDATE"
        const val EXTRA_STATUS = "status"
        const val EXTRA_PROGRESS = "progress"
    }

    sealed class ServiceStatus {
        data object Initializing : ServiceStatus()
        data class Downloading(val progress: Float) : ServiceStatus()
        data object Loading : ServiceStatus()
        data object Ready : ServiceStatus()
        data class Error(val message: String) : ServiceStatus()
    }

    private val serviceScope = CoroutineScope(SupervisorJob() + Dispatchers.Main)

    private lateinit var downloadManager: ModelDownloadManager
    private lateinit var llmEngine: LLMEngine

    private val _status = MutableStateFlow<ServiceStatus>(ServiceStatus.Initializing)
    val status: StateFlow<ServiceStatus> = _status.asStateFlow()

    private val binder = object : ILLMService.Stub() {
        override fun isReady(): Boolean {
            return _status.value is ServiceStatus.Ready && llmEngine.isLoaded
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

        override fun generate(prompt: String): String {
            if (!isReady) {
                return "Error: Model not ready. Status: ${getStatus()}"
            }
            return llmEngine.generate(prompt).getOrElse { "Error: ${it.message}" }
        }

        override fun generateWithParams(prompt: String, maxTokens: Int, temperature: Float): String {
            if (!isReady) {
                return "Error: Model not ready. Status: ${getStatus()}"
            }
            val params = GenerationParams(
                maxTokens = if (maxTokens > 0) maxTokens else 512,
                temperature = temperature.coerceIn(0f, 2f)
            )
            return llmEngine.generate(prompt, params).getOrElse { "Error: ${it.message}" }
        }

        override fun translate(text: String, sourceLanguage: String, targetLanguage: String): String {
            if (!isReady) {
                return "Error: Model not ready. Status: ${getStatus()}"
            }

            // Validate languages
            if (!Language.isValidCode(sourceLanguage)) {
                return "Error: Invalid source language code: $sourceLanguage"
            }
            if (!Language.isValidCode(targetLanguage) || targetLanguage == Language.AUTO_DETECT) {
                return "Error: Invalid target language code: $targetLanguage"
            }

            val sourceLang = if (sourceLanguage == Language.AUTO_DETECT) {
                null
            } else {
                Language.fromCode(sourceLanguage)
            }
            val targetLang = Language.fromCode(targetLanguage)!!

            val prompt = buildTranslationPrompt(text, sourceLang, targetLang)

            // Use lower temperature for translation (more deterministic)
            val params = GenerationParams(
                maxTokens = (text.length * 3).coerceIn(256, 2048),
                temperature = 0.3f
            )

            return llmEngine.generate(prompt, params)
                .map { extractTranslation(it) }
                .getOrElse { "Error: ${it.message}" }
        }

        override fun getSupportedLanguages(): String {
            return Language.toJsonArray().toString()
        }

        override fun getModelInfo(): String {
            val info = llmEngine.modelInfo ?: return "{}"
            return JSONObject().apply {
                put("name", info.name)
                put("path", info.path)
                put("sizeBytes", info.sizeBytes)
                put("contextSize", info.contextSize)
            }.toString()
        }

        private fun buildTranslationPrompt(text: String, source: Language?, target: Language): String {
            val sourceDesc = source?.displayName ?: "the source language"
            return """Translate the following text from $sourceDesc to ${target.displayName}.
Only output the translation, nothing else.

Text to translate:
$text

Translation:"""
        }

        private fun extractTranslation(response: String): String {
            // Clean up the response - remove any prefixes the model might add
            return response
                .trim()
                .removePrefix("Translation:")
                .removePrefix("translation:")
                .trim()
        }
    }

    override fun onCreate() {
        super.onCreate()
        downloadManager = ModelDownloadManager(this)
        llmEngine = LlamaEngine(contentResolver)

        createNotificationChannel()
        startForeground(NOTIFICATION_ID, createNotification("Initializing..."))

        initialize()
    }

    override fun onBind(intent: Intent?): IBinder {
        return binder
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        return START_STICKY
    }

    override fun onDestroy() {
        super.onDestroy()
        (llmEngine as? LlamaEngine)?.release() ?: llmEngine.unloadModel()
        serviceScope.cancel()
    }

    private fun initialize() {
        serviceScope.launch {
            if (downloadManager.isModelDownloaded()) {
                loadModel()
            } else {
                downloadModel()
            }
        }
    }

    private suspend fun downloadModel() {
        downloadManager.downloadModel().collect { state ->
            when (state) {
                is DownloadState.Idle -> {
                    _status.value = ServiceStatus.Downloading(0f)
                    updateNotification("Starting download...")
                }
                is DownloadState.Downloading -> {
                    _status.value = ServiceStatus.Downloading(state.progress)
                    val percent = (state.progress * 100).toInt()
                    val mbDownloaded = state.downloadedBytes / (1024 * 1024)
                    val mbTotal = state.totalBytes / (1024 * 1024)
                    updateNotification("Downloading: $percent% ($mbDownloaded/$mbTotal MB)")
                    broadcastStatus()
                }
                is DownloadState.Completed -> {
                    loadModel()
                }
                is DownloadState.Error -> {
                    _status.value = ServiceStatus.Error(state.message)
                    updateNotification("Error: ${state.message}")
                    broadcastStatus()
                }
            }
        }
    }

    private fun loadModel() {
        _status.value = ServiceStatus.Loading
        updateNotification("Loading model...")
        broadcastStatus()

        serviceScope.launch(Dispatchers.IO) {
            val modelFile = downloadManager.getModelFile(DefaultModel.CONFIG)
            llmEngine.loadModel(modelFile).fold(
                onSuccess = {
                    _status.value = ServiceStatus.Ready
                    updateNotification("Ready")
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

    private fun broadcastStatus() {
        val intent = Intent(ACTION_STATUS_UPDATE).apply {
            putExtra(EXTRA_STATUS, binder.status)
            when (val s = _status.value) {
                is ServiceStatus.Downloading -> putExtra(EXTRA_PROGRESS, s.progress)
                else -> {}
            }
        }
        sendBroadcast(intent)
    }

    private fun createNotificationChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                NOTIFICATION_CHANNEL_ID,
                "LLM Service",
                NotificationManager.IMPORTANCE_LOW
            ).apply {
                description = "Shows the status of the local LLM service"
            }
            val manager = getSystemService(NotificationManager::class.java)
            manager.createNotificationChannel(channel)
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
        val notification = createNotification(text)
        val manager = getSystemService(NotificationManager::class.java)
        manager.notify(NOTIFICATION_ID, notification)
    }
}
