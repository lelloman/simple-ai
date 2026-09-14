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
import androidx.work.WorkManager
import com.lelloman.simpleai.download.DownloadPolicy
import com.lelloman.simpleai.download.StorageInfo
import com.lelloman.simpleai.download.ModelDownloadWorker
import com.lelloman.simpleai.ISimpleAI
import com.lelloman.simpleai.api.ServiceInfoClient
import com.lelloman.simpleai.capability.CapabilityStatus
import com.lelloman.simpleai.download.ModelDownloadManager
import com.lelloman.simpleai.model.ModelRepository
import com.lelloman.simpleai.model.LocalAIModel
import com.lelloman.simpleai.service.SimpleAIService
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
    val voiceCommandsStatus: CapabilityStatus = CapabilityStatus.Checking,
    val translationStatus: CapabilityStatus = CapabilityStatus.Checking,
    val cloudAiStatus: CapabilityStatus = CapabilityStatus.Checking,
    val localAiStatus: CapabilityStatus = CapabilityStatus.Checking,
    val downloadedLanguages: Set<String> = emptySet(),
    val downloadingLanguages: Set<String> = emptySet(),
    val languageDownloadErrors: Map<String, String> = emptyMap(),
    val isServiceConnected: Boolean = false,
    val serviceError: String? = null,
    val downloadJobs: Map<String, String> = emptyMap(),
    val storage: StorageInfo? = null,
    val allowMeteredDownloads: Boolean = false
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

    private val translationManager = models.translation
    private val downloadManager = ModelDownloadManager(application)

    private val serviceConnection = object : ServiceConnection {
        override fun onServiceConnected(name: ComponentName?, service: IBinder?) {
            Log.i(TAG, "Service connected")
            simpleAiService = ISimpleAI.Stub.asInterface(service)
            _state.update { it.copy(isServiceConnected = true) }
            refreshCapabilities()
        }

        override fun onServiceDisconnected(name: ComponentName?) {
            Log.i(TAG, "Service disconnected")
            simpleAiService = null
            isBound = false
            _state.update { it.copy(isServiceConnected = false) }
        }
    }

    init {
        viewModelScope.launch { models.languageDownloads.active.collect { active -> _state.update { it.copy(downloadingLanguages = active) } } }
        viewModelScope.launch { models.languageDownloads.errors.collect { errors -> _state.update { it.copy(languageDownloadErrors = errors) } } }

        _state.update { it.copy(allowMeteredDownloads = DownloadPolicy.allowsMetered(application)) }
        viewModelScope.launch(Dispatchers.IO) {
            while (true) {
                runCatching { downloadManager.getStorageInfo() }.onSuccess { info -> _state.update { it.copy(storage = info) } }
                kotlinx.coroutines.delay(5000)
            }
        }

        for (model in listOf(ModelDownloadWorker.VOICE, ModelDownloadWorker.LOCAL)) {
            viewModelScope.launch {
                WorkManager.getInstance(application).getWorkInfosForUniqueWorkFlow(ModelDownloadWorker.name(model)).collect { jobs ->
                    val job = jobs.firstOrNull { !it.state.isFinished } ?: jobs.firstOrNull()
                    _state.update { it.copy(downloadJobs = it.downloadJobs + (model to (job?.state?.name ?: ""))) }
                }
            }
        }
        startAndBindService()
        initializeTranslationManager()
        viewModelScope.launch {
            models.capabilities.translationStatus.collect { status ->
                _state.update { it.copy(translationStatus = status) }
            }
        }
        viewModelScope.launch {
            models.capabilities.cloudAiStatus.collect { status ->
                _state.update { it.copy(cloudAiStatus = status) }
            }
        }
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
                withContext(Dispatchers.IO) {
                    ServiceInfoClient.request(service::getServiceInfo)
                }

                _state.update { it.copy(serviceError = null) }
            } catch (e: Exception) {
                Log.e(TAG, "Failed to refresh capabilities", e)
                _state.update { it.copy(serviceError = e.message ?: "Could not connect to SimpleAI") }
            }
        }
    }

    // =========================================================================
    // Local AI Download
    // =========================================================================

    fun downloadLocalAi() = models.downloadLocal()
    fun setAllowMeteredDownloads(allowed: Boolean) {
        DownloadPolicy.setAllowsMetered(getApplication(), allowed)
        _state.update { it.copy(allowMeteredDownloads = allowed) }
    }
    fun pauseDownload(model: String) = models.pauseDownload(model)

    fun deleteLocalAi() = models.deleteLocal()

    fun deleteVoiceCommands() = models.deleteVoice()

    fun downloadVoiceCommands() = models.downloadVoice()

    // =========================================================================
    // Translation Languages
    // =========================================================================

    fun downloadTranslationLanguage(languageCode: String) = models.languageDownloads.start(languageCode)

    fun clearLanguageDownloadError(languageCode: String) = models.languageDownloads.clearError(languageCode)

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
    }
}
