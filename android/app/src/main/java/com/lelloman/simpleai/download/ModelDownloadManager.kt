package com.lelloman.simpleai.download

import android.content.Context
import android.os.StatFs
import com.lelloman.simpleai.model.LocalAIModel
import com.lelloman.simpleai.model.NluModel
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.flowOn
import okhttp3.OkHttpClient
import okhttp3.Request
import java.io.File
import java.io.FileOutputStream
import java.util.concurrent.TimeUnit

data class StorageInfo(
    val usedBytes: Long,
    val availableBytes: Long,
    val totalBytes: Long
)

sealed class DownloadState {
    data object Idle : DownloadState()
    data class Downloading(val progress: Float, val downloadedBytes: Long, val totalBytes: Long) : DownloadState()
    data object Completed : DownloadState()
    data class Error(val message: String) : DownloadState()
}

data class ModelConfig(
    val name: String,
    val url: String,
    val fileName: String,
    val expectedSizeMb: Int,
    val expectedBytes: Long? = null,
    val sha256: String? = null
)

class ModelDownloadManager(
    private val context: Context,
    private val client: OkHttpClient = OkHttpClient.Builder()
        .connectTimeout(30, TimeUnit.SECONDS)
        .readTimeout(5, TimeUnit.MINUTES)
        .writeTimeout(5, TimeUnit.MINUTES)
        .build(),
    private val modelsDirProvider: () -> File = { File(context.filesDir, "models").also { it.mkdirs() } }
) {

    private val modelsDir: File
        get() = modelsDirProvider()

    fun getModelFile(config: ModelConfig): File {
        return File(modelsDir, config.fileName)
    }

    fun isModelDownloaded(config: ModelConfig): Boolean {
        val file = getModelFile(config)
        return file.exists() && file.length() > 0
    }

    /**
     * Check if the local AI model is downloaded.
     */
    fun isLocalAiDownloaded(): Boolean {
        val file = File(modelsDir, LocalAIModel.FILE_NAME)
        return file.exists() && file.length() > 0
    }

    /**
     * Get the local AI model file.
     */
    fun getLocalAiModelFile(): File {
        return File(modelsDir, LocalAIModel.FILE_NAME)
    }

    fun downloadModel(config: ModelConfig): Flow<DownloadState> =
        ResumableDownload(client).download(config, getModelFile(config))

    fun deleteModel(config: ModelConfig): Boolean {
        val file = getModelFile(config)
        val tempFile = File(modelsDir, "${config.fileName}.tmp")
        tempFile.delete()
        return file.delete()
    }

    /**
     * Delete the local AI model.
     */
    fun deleteLocalAi(): Boolean {
        val file = File(modelsDir, LocalAIModel.FILE_NAME)
        val tempFile = File(modelsDir, "${LocalAIModel.FILE_NAME}.tmp")
        val tempDeleted = !tempFile.exists() || tempFile.delete()
        val identity = File(tempFile.path + ".identity")
        val identityDeleted = !identity.exists() || identity.delete()
        val modelDeleted = !file.exists() || file.delete()
        return tempDeleted && modelDeleted && identityDeleted
    }

    /**
     * Delete the voice commands (NLU) model.
     */
    fun deleteVoiceCommands(): Boolean {
        val nluDir = File(context.filesDir, "nlu_models")
        return if (nluDir.exists()) {
            nluDir.deleteRecursively()
        } else {
            true
        }
    }

    /**
     * Check if the voice commands model is downloaded.
     */
    fun isVoiceCommandsDownloaded(): Boolean {
        val nluDir = File(context.filesDir, "nlu_models")
        val modelFile = File(nluDir, "xlm_roberta_base_int8.onnx")
        return modelFile.exists() && modelFile.length() > 0
    }

    /**
     * Download the voice commands (NLU) model.
     */
    fun downloadVoiceCommands(): Flow<DownloadState> = ResumableDownload(client).download(
        ModelConfig("Voice Commands", NluModel.URL, NluModel.FILE_NAME, 509, NluModel.SIZE_BYTES, NluModel.SHA256),
        File(context.filesDir, "nlu_models/${NluModel.FILE_NAME}")
    )

    fun getStorageInfo(): StorageInfo {
        // App data includes NLU working copies, ML Kit private models, and partial files.
        val modelsUsed = DownloadPolicy.usedBytes(context.dataDir)

        val statFs = StatFs(context.filesDir.absolutePath)
        val availableBytes = statFs.availableBytes
        val totalBytes = statFs.totalBytes

        return StorageInfo(
            usedBytes = modelsUsed,
            availableBytes = availableBytes,
            totalBytes = totalBytes
        )
    }
}
