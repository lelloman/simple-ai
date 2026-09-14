package com.lelloman.simpleai.ui

import com.lelloman.simpleai.R

import android.app.Application
import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.ServiceConnection
import android.os.IBinder
import android.util.Log
import androidx.core.content.ContextCompat
import androidx.lifecycle.SavedStateHandle
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
    val languageOperationMessage: String? = null,
    val languageOperationError: String? = null,
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
    val draft: TranslationDraft = TranslationDraft(),
    val isTranslating: Boolean = false,
    val translatedText: String? = null,
    val detectedLanguage: String? = null,
    val error: String? = null
)

/**
 * ViewModel for the capabilities screen.
 */
class CapabilitiesViewModel(application: Application, savedStateHandle: SavedStateHandle) : AndroidViewModel(application) {

    companion object {
        private const val TAG = "CapabilitiesViewModel"
    }

    private val _state = MutableStateFlow(CapabilitiesState())
    val state: StateFlow<CapabilitiesState> = _state.asStateFlow()

    private var simpleAiService: ISimpleAI? = null

    private val models = ModelRepository.get(application)

    val cloudEndpoint = models.cloudSettings.endpoint
    val gatewaySignedIn = models.gatewayAuth.signedIn
    val gatewayAuthError = models.gatewayAuth.error
    fun signOutGateway() { viewModelScope.launch { models.gatewayAuth.signOut() } }
    suspend fun saveCloudEndpoint(endpoint: String): Boolean = withContext(Dispatchers.IO) {
        models.gatewayAuth.changeServer(endpoint)
    }

    private val translationManager = models.translation
    private val translationSession = TranslationSession(viewModelScope, savedStateHandle) {
        withContext(Dispatchers.IO) {
            val service = simpleAiService
            if (service == null) Result.failure(IllegalStateException(getApplication<Application>().getString(R.string.ui_service_disconnected_recovery)))
            else com.lelloman.simpleai.api.ServiceTranslationClient.request(it.text, it.source, it.target, service::translate)
        }
    }
    val translationState = translationSession.state
    private val downloadManager = ModelDownloadManager(application)

    private val serviceBinding = ServiceBinding(application) { service, error ->
        simpleAiService = service
        _state.update { it.copy(isServiceConnected = service != null, serviceError = error) }
        if (service != null) refreshCapabilities()
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
        try {
            serviceBinding.connect()
        } catch (e: Exception) {
            _state.update { it.copy(isServiceConnected = false, serviceError = getApplication<Application>().getString(R.string.ui_could_not_start, e.message.orEmpty())) }
        }
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
            val service = simpleAiService ?: run { startAndBindService(); return@launch }

            try {
                withContext(Dispatchers.IO) {
                    ServiceInfoClient.request(service::getServiceInfo)
                }

                _state.update { it.copy(serviceError = null) }
            } catch (e: Exception) {
                Log.e(TAG, "Failed to refresh capabilities", e)
                _state.update { it.copy(serviceError = e.message ?: getApplication<Application>().getString(R.string.ui_could_not_connect)) }
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

    fun clearLanguageOperationMessage() { _state.update { it.copy(languageOperationMessage = null) } }

    fun clearLanguageOperationError() { _state.update { it.copy(languageOperationError = null) } }

    fun clearLanguageDownloadError(languageCode: String) = models.languageDownloads.clearError(languageCode)

    fun deleteTranslationLanguage(languageCode: String) {
        viewModelScope.launch {
            translationManager.deleteLanguage(languageCode).fold(
                onSuccess = {
                    _state.update { it.copy(languageOperationMessage = getApplication<Application>().getString(R.string.ui_language_deleted)) }
                    refreshCapabilities()
                },
                onFailure = { e ->
                    Log.e(TAG, "Failed to delete language: $languageCode", e)
                    _state.update { it.copy(languageOperationError = getApplication<Application>().getString(R.string.ui_could_not_delete_language, languageCode, e.message.orEmpty())) }
                }
            )
        }
    }

    // =========================================================================
    // Translation Test
    // =========================================================================

    fun editTranslation(draft: TranslationDraft) = translationSession.edit(draft)
    fun translate() = translationSession.submit()

    override fun onCleared() {
        super.onCleared()
        serviceBinding.close()
    }
}
