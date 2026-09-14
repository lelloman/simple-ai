package com.lelloman.simpleai.model

import android.content.Context
import androidx.work.WorkManager
import com.lelloman.simpleai.download.ModelDownloadWorker
import com.lelloman.simpleai.download.KeyedDownloads
import com.lelloman.simpleai.capability.CapabilityManager
import com.lelloman.simpleai.download.ModelConfig
import com.lelloman.simpleai.download.ModelDownloadManager
import com.lelloman.simpleai.llm.LlamaEngine
import com.lelloman.simpleai.nlu.OnnxNLUEngine
import com.lelloman.simpleai.translation.TranslationManager
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.launch

/** Application-owned model work survives the activity that requested it. */
class ModelRepository private constructor(private val context: Context) {
    companion object {
        @Volatile private var instance: ModelRepository? = null
        fun get(context: Context): ModelRepository = instance ?: synchronized(this) {
            instance ?: ModelRepository(context.applicationContext).also { instance = it }
        }
    }
    val cloudSettings = com.lelloman.simpleai.cloud.CloudSettings(context.getSharedPreferences("cloud_settings", Context.MODE_PRIVATE))
    val capabilities = CapabilityManager(context, cloudSettings.endpoint.value)
    val translation = TranslationManager(context, capabilities::syncTranslationLanguages)
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
    init {
        scope.launch { cloudSettings.endpoint.collect { capabilities.updateCloudEndpoint(it) } }
    }
    val languageDownloads = KeyedDownloads(scope) { translation.downloadLanguage(it) }
    private val downloads = ModelDownloadManager(context)
    val voice = ManagedModel(
        CapabilityManager.VOICE_COMMANDS_MODEL_SIZE,
        downloads::isVoiceCommandsDownloaded,
        downloads::downloadVoiceCommands,
        load = {
            val engine = OnnxNLUEngine(context)
            engine.initialize().fold(
                onSuccess = { Result.success(engine) },
                onFailure = { engine.release(); Result.failure(it) }
            )
        },
        dispose = OnnxNLUEngine::release,
        publish = capabilities::updateVoiceCommandsStatus
    )
    val local = ManagedModel(
        LocalAIModel.SIZE_BYTES,
        downloads::isLocalAiDownloaded,
        download = { downloads.downloadModel(ModelConfig(LocalAIModel.NAME, LocalAIModel.URL, LocalAIModel.FILE_NAME, LocalAIModel.SIZE_MB, LocalAIModel.SIZE_BYTES, LocalAIModel.SHA256)) },
        load = {
            val engine = LlamaEngine(context)
            engine.loadModel(downloads.getLocalAiModelFile()).fold(
                onSuccess = { Result.success(engine) },
                onFailure = { engine.release(); Result.failure(it) }
            )
        },
        dispose = LlamaEngine::release,
        publish = capabilities::updateLocalAiStatus
    )

    private val idleResources = IdleResources(scope) { voice.unload(); local.unload() }
    suspend fun <T> withWork(block: suspend () -> T): T = idleResources.withWork(block)

    fun downloadVoice() = ModelDownloadWorker.enqueue(context, ModelDownloadWorker.VOICE)
    fun downloadLocal() = ModelDownloadWorker.enqueue(context, ModelDownloadWorker.LOCAL)
    fun pauseDownload(model: String) { WorkManager.getInstance(context).cancelUniqueWork(ModelDownloadWorker.name(model)) }
    fun deleteVoice() { scope.launch { voice.delete(downloads::deleteVoiceCommands) } }
    fun deleteLocal() { scope.launch { local.delete(downloads::deleteLocalAi) } }
    fun initialize() {
        scope.launch { translation.initialize() }
        scope.launch { voice.inspect() }
        scope.launch { local.inspect() }
    }
}
