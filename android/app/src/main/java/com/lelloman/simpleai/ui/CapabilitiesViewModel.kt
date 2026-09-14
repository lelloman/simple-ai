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
import com.lelloman.simpleai.api.ServiceInfoClient
import com.lelloman.simpleai.capability.CapabilityStatus
import com.lelloman.simpleai.download.ModelDownloadManager
import com.lelloman.simpleai.model.ModelRepository
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
    val languageDownloadError: String? = null,
    val isServiceConnected: Boolean = false,
    val serviceError: String? = null
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

    private val models = ModelRepository.get(application)

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
        viewModelScope.launch {
            models.capabilities.voiceCommandsStatus.collect { status ->
                _state.update { it.copy(voiceCommandsStatus = status) }
            }
        }
        viewModelScope.launch {
            models.capabilities.localAiStatus.collect { status ->
                _state.update { it.copy(localAiStatus = status) }
            }
        }
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
                    ServiceInfoClient.request(service::getServiceInfo)
                }

                parseServiceInfo(response)
            } catch (e: Exception) {
                Log.e(TAG, "Failed to refresh capabilities", e)
                _state.update { it.copy(serviceError = e.message ?: "Could not connect to SimpleAI") }
            }
        }
    }

    private fun parseServiceInfo(capabilities: kotlinx.serialization.json.JsonObject) {
        try {

            // Parse voice commands
            val voiceCommands = capabilities["voiceCommands"]?.jsonObject
            val vcStatus = parseCapabilityStatus(voiceCommands)

            // Parse translation
            val translation = capabilities["translation"]?.jsonObject
            val transStatus = parseCapabilityStatus(translation)
            // Note: downloadedLanguages comes from TranslationManager's StateFlow,
            // not from the service, to ensure we have the most up-to-date data

            // Parse cloud AI
            val cloudAi = capabilities["cloudAi"]?.jsonObject
            val cloudStatus = parseCapabilityStatus(cloudAi)

            // Parse local AI
            val localAi = capabilities["localAi"]?.jsonObject
            val localStatus = parseCapabilityStatus(localAi)

            _state.update { currentState ->
                currentState.copy(
                    serviceError = null,
                    // Don't overwrite if currently downloading (ViewModel manages download progress)
                    voiceCommandsStatus = if (currentState.voiceCommandsStatus is CapabilityStatus.Downloading) {
                        currentState.voiceCommandsStatus
                    } else {
                        vcStatus
                    },
                    translationStatus = transStatus,
                    cloudAiStatus = cloudStatus,
                    localAiStatus = if (currentState.localAiStatus is CapabilityStatus.Downloading) {
                        currentState.localAiStatus
                    } else {
                        localStatus
                    }
                )
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to parse service info", e)
            _state.update { it.copy(serviceError = "Could not read service status: ${e.message}") }
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

    fun downloadLocalAi() = models.downloadLocal()

    fun deleteLocalAi() {
        viewModelScope.launch(Dispatchers.IO) {
            val success = downloadManager.deleteLocalAi()
            if (success) {
                _state.update { it.copy(localAiStatus = CapabilityStatus.NotDownloaded(LocalAIModel.SIZE_BYTES)) }
            } else {
                Log.e(TAG, "Failed to delete Local AI model")
            }
        }
    }

    fun deleteVoiceCommands() {
        viewModelScope.launch(Dispatchers.IO) {
            val success = downloadManager.deleteVoiceCommands()
            if (success) {
                _state.update { it.copy(voiceCommandsStatus = CapabilityStatus.NotDownloaded(534_000_000)) }
            } else {
                Log.e(TAG, "Failed to delete Voice Commands model")
            }
        }
    }

    fun downloadVoiceCommands() = models.downloadVoice()

    // =========================================================================
    // Translation Languages
    // =========================================================================

    fun downloadTranslationLanguage(languageCode: String) {
        viewModelScope.launch {
            _state.update { it.copy(downloadingLanguage = languageCode, languageDownloadError = null) }

            translationManager.downloadLanguage(languageCode) { progress ->
                // Progress callback (ML Kit doesn't provide granular progress)
            }.fold(
                onSuccess = {
                    _state.update { it.copy(downloadingLanguage = null, languageDownloadError = null) }
                    refreshCapabilities()
                },
                onFailure = { e ->
                    Log.e(TAG, "Failed to download language: $languageCode", e)
                    val errorMessage = when {
                        e.message?.contains("wifi", ignoreCase = true) == true ->
                            "WiFi required for download"
                        e.message?.contains("network", ignoreCase = true) == true ->
                            "Network error. Check your connection."
                        else -> e.message ?: "Download failed"
                    }
                    _state.update { it.copy(downloadingLanguage = null, languageDownloadError = errorMessage) }
                }
            )
        }
    }

    fun clearLanguageDownloadError() {
        _state.update { it.copy(languageDownloadError = null) }
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
