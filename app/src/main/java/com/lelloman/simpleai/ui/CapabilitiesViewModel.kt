package com.lelloman.simpleai.ui

import android.app.Application
import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.ServiceConnection
import android.os.IBinder
import android.util.Log
import androidx.core.content.ContextCompat
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.lelloman.simpleai.ISimpleAI
import com.lelloman.simpleai.capability.CapabilityStatus
import com.lelloman.simpleai.download.DownloadState
import com.lelloman.simpleai.download.ModelConfig
import com.lelloman.simpleai.download.ModelDownloadManager
import kotlinx.coroutines.flow.catch
import com.lelloman.simpleai.model.LocalAIModel
import com.lelloman.simpleai.service.SimpleAIService
import com.lelloman.simpleai.translation.TranslationManager
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive

/**
 * UI state for the capabilities screen.
 */
data class CapabilitiesState(
    val voiceCommandsStatus: CapabilityStatus = CapabilityStatus.NotDownloaded(534_000_000),
    val translationStatus: CapabilityStatus = CapabilityStatus.NotDownloaded(0),
    val cloudAiStatus: CapabilityStatus = CapabilityStatus.Ready,
    val localAiStatus: CapabilityStatus = CapabilityStatus.NotDownloaded(LocalAIModel.SIZE_BYTES),
    val downloadedLanguages: Set<String> = emptySet(),
    val downloadingLanguage: String? = null,
    val isServiceConnected: Boolean = false
)

/**
 * State for translation test screen.
 */
data class TranslationState(
    val isTranslating: Boolean = false,
    val translatedText: String? = null,
    val detectedLanguage: String? = null,
    val error: String? = null
)

/**
 * ViewModel for the capabilities screen.
 */
class CapabilitiesViewModel(application: Application) : AndroidViewModel(application) {

    companion object {
        private const val TAG = "CapabilitiesViewModel"
    }

    private val _state = MutableStateFlow(CapabilitiesState())
    val state: StateFlow<CapabilitiesState> = _state.asStateFlow()

    private val _translationState = MutableStateFlow(TranslationState())
    val translationState: StateFlow<TranslationState> = _translationState.asStateFlow()

    private var simpleAiService: ISimpleAI? = null
    private var isBound = false

    private val json = Json { ignoreUnknownKeys = true }

    private val translationManager = TranslationManager(application)
    private val downloadManager = ModelDownloadManager(application)

    private val serviceConnection = object : ServiceConnection {
        override fun onServiceConnected(name: ComponentName?, service: IBinder?) {
            Log.i(TAG, "Service connected")
            simpleAiService = ISimpleAI.Stub.asInterface(service)
            _state.update { it.copy(isServiceConnected = true) }
            // Refresh immediately and again after a short delay to catch initialization
            refreshCapabilities()
            viewModelScope.launch {
                kotlinx.coroutines.delay(500)
                refreshCapabilities()
            }
        }

        override fun onServiceDisconnected(name: ComponentName?) {
            Log.i(TAG, "Service disconnected")
            simpleAiService = null
            isBound = false
            _state.update { it.copy(isServiceConnected = false) }
        }
    }

    init {
        startAndBindService()
        initializeTranslationManager()
    }

    private fun startAndBindService() {
        val context = getApplication<Application>()
        val serviceIntent = Intent(context, SimpleAIService::class.java)

        // Start as foreground service
        ContextCompat.startForegroundService(context, serviceIntent)

        // Bind to it
        context.bindService(serviceIntent, serviceConnection, Context.BIND_AUTO_CREATE)
        isBound = true
    }

    private fun initializeTranslationManager() {
        viewModelScope.launch(Dispatchers.IO) {
            translationManager.initialize()

            // Observe downloaded languages
            translationManager.downloadedLanguages.collect { languages ->
                _state.update { it.copy(downloadedLanguages = languages) }
            }
        }
    }

    /**
     * Refresh capability status from the service.
     */
    fun refreshCapabilities() {
        viewModelScope.launch {
            val service = simpleAiService ?: return@launch

            try {
                val response = withContext(Dispatchers.IO) {
                    service.getServiceInfo(1)
                }

                parseServiceInfo(response)
            } catch (e: Exception) {
                Log.e(TAG, "Failed to refresh capabilities", e)
            }
        }
    }

    private fun parseServiceInfo(responseJson: String) {
        try {
            val response = json.parseToJsonElement(responseJson).jsonObject
            val status = response["status"]?.jsonPrimitive?.content

            if (status != "success") return

            val data = response["data"]?.jsonObject ?: return
            val capabilities = data["capabilities"]?.jsonObject ?: return

            // Parse voice commands
            val voiceCommands = capabilities["voiceCommands"]?.jsonObject
            val vcStatus = parseCapabilityStatus(voiceCommands)

            // Parse translation
            val translation = capabilities["translation"]?.jsonObject
            val transStatus = parseCapabilityStatus(translation)
            val languages = translation?.get("languages")?.jsonArray
                ?.map { it.jsonPrimitive.content }
                ?.toSet() ?: emptySet()

            // Parse cloud AI
            val cloudAi = capabilities["cloudAi"]?.jsonObject
            val cloudStatus = parseCapabilityStatus(cloudAi)

            // Parse local AI
            val localAi = capabilities["localAi"]?.jsonObject
            val localStatus = parseCapabilityStatus(localAi)

            _state.update {
                it.copy(
                    voiceCommandsStatus = vcStatus,
                    translationStatus = transStatus,
                    cloudAiStatus = cloudStatus,
                    localAiStatus = localStatus,
                    downloadedLanguages = languages
                )
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to parse service info", e)
        }
    }

    private fun parseCapabilityStatus(json: kotlinx.serialization.json.JsonObject?): CapabilityStatus {
        if (json == null) return CapabilityStatus.NotDownloaded(0)

        return when (json["status"]?.jsonPrimitive?.content) {
            "ready" -> CapabilityStatus.Ready
            "not_downloaded" -> {
                val size = json["modelSize"]?.jsonPrimitive?.content?.toLongOrNull() ?: 0
                CapabilityStatus.NotDownloaded(size)
            }
            "downloading" -> {
                val downloaded = json["downloadedBytes"]?.jsonPrimitive?.content?.toLongOrNull() ?: 0
                val total = json["totalBytes"]?.jsonPrimitive?.content?.toLongOrNull() ?: 0
                CapabilityStatus.Downloading(downloaded, total)
            }
            "error" -> {
                val message = json["message"]?.jsonPrimitive?.content ?: "Unknown error"
                val canRetry = json["canRetry"]?.jsonPrimitive?.content?.toBoolean() ?: true
                CapabilityStatus.Error(message, canRetry)
            }
            else -> CapabilityStatus.NotDownloaded(0)
        }
    }

    // =========================================================================
    // Local AI Download
    // =========================================================================

    fun downloadLocalAi() {
        viewModelScope.launch {
            val config = ModelConfig(
                name = LocalAIModel.NAME,
                url = LocalAIModel.URL,
                fileName = LocalAIModel.FILE_NAME,
                expectedSizeMb = LocalAIModel.SIZE_MB
            )

            downloadManager.downloadModel(config)
                .catch { e ->
                    _state.update {
                        it.copy(localAiStatus = CapabilityStatus.Error(
                            e.message ?: "Download failed",
                            canRetry = true
                        ))
                    }
                }
                .collect { downloadState ->
                    when (downloadState) {
                        is DownloadState.Idle -> {
                            _state.update {
                                it.copy(localAiStatus = CapabilityStatus.Downloading(0, LocalAIModel.SIZE_BYTES))
                            }
                        }
                        is DownloadState.Downloading -> {
                            _state.update {
                                it.copy(localAiStatus = CapabilityStatus.Downloading(
                                    downloadState.downloadedBytes,
                                    downloadState.totalBytes
                                ))
                            }
                        }
                        is DownloadState.Completed -> {
                            _state.update { it.copy(localAiStatus = CapabilityStatus.Ready) }
                            // Service will load the model on next initialization
                        }
                        is DownloadState.Error -> {
                            _state.update {
                                it.copy(localAiStatus = CapabilityStatus.Error(
                                    downloadState.message,
                                    canRetry = true
                                ))
                            }
                        }
                    }
                }
        }
    }

    // =========================================================================
    // Translation Languages
    // =========================================================================

    fun downloadTranslationLanguage(languageCode: String) {
        viewModelScope.launch {
            _state.update { it.copy(downloadingLanguage = languageCode) }

            translationManager.downloadLanguage(languageCode) { progress ->
                // Progress callback (ML Kit doesn't provide granular progress)
            }.fold(
                onSuccess = {
                    _state.update { it.copy(downloadingLanguage = null) }
                    refreshCapabilities()
                },
                onFailure = { e ->
                    Log.e(TAG, "Failed to download language: $languageCode", e)
                    _state.update { it.copy(downloadingLanguage = null) }
                }
            )
        }
    }

    fun deleteTranslationLanguage(languageCode: String) {
        viewModelScope.launch {
            translationManager.deleteLanguage(languageCode).fold(
                onSuccess = {
                    refreshCapabilities()
                },
                onFailure = { e ->
                    Log.e(TAG, "Failed to delete language: $languageCode", e)
                }
            )
        }
    }

    // =========================================================================
    // Translation Test
    // =========================================================================

    fun translate(text: String, sourceLang: String, targetLang: String) {
        viewModelScope.launch {
            _translationState.value = TranslationState(isTranslating = true)

            translationManager.translate(text, sourceLang, targetLang).fold(
                onSuccess = { result ->
                    _translationState.value = TranslationState(
                        isTranslating = false,
                        translatedText = result.translatedText,
                        detectedLanguage = result.detectedSourceLang
                    )
                },
                onFailure = { e ->
                    Log.e(TAG, "Translation failed", e)
                    _translationState.value = TranslationState(
                        isTranslating = false,
                        error = e.message ?: "Translation failed"
                    )
                }
            )
        }
    }

    override fun onCleared() {
        super.onCleared()
        val context = getApplication<Application>()
        if (isBound) {
            context.unbindService(serviceConnection)
            isBound = false
        }
        translationManager.release()
    }
}
