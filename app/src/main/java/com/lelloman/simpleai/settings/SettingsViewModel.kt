package com.lelloman.simpleai.settings

import android.app.Application
import android.content.Context
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.lelloman.simpleai.download.DownloadState
import com.lelloman.simpleai.download.ModelDownloadManager
import com.lelloman.simpleai.download.StorageInfo
import com.lelloman.simpleai.model.AvailableModel
import com.lelloman.simpleai.model.AvailableModels
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

data class ModelState(
    val model: AvailableModel,
    val isDownloaded: Boolean,
    val isSelected: Boolean,
    val sizeBytes: Long
)

data class SettingsState(
    val storageInfo: StorageInfo = StorageInfo(0, 0, 0),
    val models: List<ModelState> = emptyList(),
    val downloadingModelId: String? = null,
    val downloadProgress: Float = 0f,
    val showDeleteConfirmation: Boolean = false,
    val modelToDelete: AvailableModel? = null
)

class SettingsViewModel(application: Application) : AndroidViewModel(application) {

    companion object {
        private const val PREFS_NAME = "simple_ai_prefs"
        private const val KEY_SELECTED_MODEL = "selected_model_id"
    }

    private val downloadManager = ModelDownloadManager(application)
    private val prefs = application.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)

    private val _state = MutableStateFlow(SettingsState())
    val state: StateFlow<SettingsState> = _state.asStateFlow()

    init {
        refreshState()
    }

    fun refreshState() {
        val storageInfo = downloadManager.getStorageInfo()
        val selectedModelId = getSelectedModelId()

        val modelStates = AvailableModels.ALL.map { model ->
            ModelState(
                model = model,
                isDownloaded = downloadManager.isModelDownloaded(model),
                isSelected = model.id == selectedModelId,
                sizeBytes = downloadManager.getModelSizeBytes(model)
            )
        }

        _state.value = _state.value.copy(
            storageInfo = storageInfo,
            models = modelStates
        )
    }

    fun getSelectedModelId(): String {
        return prefs.getString(KEY_SELECTED_MODEL, AvailableModels.DEFAULT_MODEL_ID)
            ?: AvailableModels.DEFAULT_MODEL_ID
    }

    fun selectModel(model: AvailableModel) {
        if (!downloadManager.isModelDownloaded(model)) {
            // Can't select a model that's not downloaded
            return
        }
        prefs.edit().putString(KEY_SELECTED_MODEL, model.id).apply()
        refreshState()
    }

    fun downloadModel(model: AvailableModel) {
        if (_state.value.downloadingModelId != null) {
            // Already downloading
            return
        }

        _state.value = _state.value.copy(
            downloadingModelId = model.id,
            downloadProgress = 0f
        )

        viewModelScope.launch {
            downloadManager.downloadModel(model).collect { downloadState ->
                when (downloadState) {
                    is DownloadState.Idle -> {
                        _state.value = _state.value.copy(downloadProgress = 0f)
                    }
                    is DownloadState.Downloading -> {
                        _state.value = _state.value.copy(downloadProgress = downloadState.progress)
                    }
                    is DownloadState.Completed -> {
                        _state.value = _state.value.copy(
                            downloadingModelId = null,
                            downloadProgress = 0f
                        )
                        // Auto-select the newly downloaded model
                        prefs.edit().putString(KEY_SELECTED_MODEL, model.id).apply()
                        refreshState()
                    }
                    is DownloadState.Error -> {
                        _state.value = _state.value.copy(
                            downloadingModelId = null,
                            downloadProgress = 0f
                        )
                        refreshState()
                    }
                }
            }
        }
    }

    fun showDeleteConfirmation(model: AvailableModel) {
        _state.value = _state.value.copy(
            showDeleteConfirmation = true,
            modelToDelete = model
        )
    }

    fun dismissDeleteConfirmation() {
        _state.value = _state.value.copy(
            showDeleteConfirmation = false,
            modelToDelete = null
        )
    }

    fun confirmDelete() {
        val model = _state.value.modelToDelete ?: return
        downloadManager.deleteModel(model)

        // If we deleted the selected model, switch to another one
        val selectedId = getSelectedModelId()
        if (model.id == selectedId) {
            // Find another downloaded model or clear selection
            val anotherDownloaded = AvailableModels.ALL.firstOrNull {
                it.id != model.id && downloadManager.isModelDownloaded(it)
            }
            if (anotherDownloaded != null) {
                prefs.edit().putString(KEY_SELECTED_MODEL, anotherDownloaded.id).apply()
            }
        }

        dismissDeleteConfirmation()
        refreshState()
    }
}
