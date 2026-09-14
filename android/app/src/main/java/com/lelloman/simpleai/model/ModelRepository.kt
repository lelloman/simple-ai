package com.lelloman.simpleai.model

import android.content.Context
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
    val capabilities = CapabilityManager(context)
    val translation = TranslationManager(context, capabilities::syncTranslationLanguages)
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
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
        download = { downloads.downloadModel(ModelConfig(LocalAIModel.NAME, LocalAIModel.URL, LocalAIModel.FILE_NAME, LocalAIModel.SIZE_MB)) },
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

    fun downloadVoice() { scope.launch { voice.downloadAndActivate() } }
    fun downloadLocal() { scope.launch { local.downloadAndActivate() } }
    fun deleteVoice() { scope.launch { voice.delete(downloads::deleteVoiceCommands) } }
    fun deleteLocal() { scope.launch { local.delete(downloads::deleteLocalAi) } }
    fun initialize() {
        scope.launch { translation.initialize() }
        scope.launch { voice.initialize() }
        scope.launch { local.initialize() }
    }
}
