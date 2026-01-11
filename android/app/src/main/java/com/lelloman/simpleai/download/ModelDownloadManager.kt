package com.lelloman.simpleai.download

import android.content.Context
import android.os.StatFs
import com.lelloman.simpleai.model.LocalAIModel
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
    val expectedSizeMb: Int
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

    fun downloadModel(config: ModelConfig): Flow<DownloadState> = flow {
        emit(DownloadState.Idle)

        val targetFile = getModelFile(config)
        val tempFile = File(modelsDir, "${config.fileName}.tmp")

        try {
            // Check if we can resume
            val existingBytes = if (tempFile.exists()) tempFile.length() else 0L

            val requestBuilder = Request.Builder().url(config.url)
            if (existingBytes > 0) {
                requestBuilder.addHeader("Range", "bytes=$existingBytes-")
            }

            val response = client.newCall(requestBuilder.build()).execute()

            if (!response.isSuccessful && response.code != 206) {
                emit(DownloadState.Error("Download failed: HTTP ${response.code}"))
                return@flow
            }

            val body = response.body ?: run {
                emit(DownloadState.Error("Empty response body"))
                return@flow
            }

            val contentLength = body.contentLength()
            val totalBytes = if (response.code == 206) {
                existingBytes + contentLength
            } else {
                contentLength
            }

            val outputStream = FileOutputStream(tempFile, response.code == 206)
            val buffer = ByteArray(8192)
            var downloadedBytes = existingBytes
            var lastEmitTime = System.currentTimeMillis()

            body.byteStream().use { inputStream ->
                outputStream.use { output ->
                    while (true) {
                        val bytesRead = inputStream.read(buffer)
                        if (bytesRead == -1) break

                        output.write(buffer, 0, bytesRead)
                        downloadedBytes += bytesRead

                        // Emit progress at most every 100ms to avoid flooding
                        val now = System.currentTimeMillis()
                        if (now - lastEmitTime >= 100) {
                            val progress = if (totalBytes > 0) {
                                downloadedBytes.toFloat() / totalBytes
                            } else {
                                0f
                            }
                            emit(DownloadState.Downloading(progress, downloadedBytes, totalBytes))
                            lastEmitTime = now
                        }
                    }
                }
            }

            // Rename temp file to final file
            if (tempFile.renameTo(targetFile)) {
                emit(DownloadState.Completed)
            } else {
                emit(DownloadState.Error("Failed to finalize download"))
            }

        } catch (e: Exception) {
            emit(DownloadState.Error("Download error: ${e.message}"))
        }
    }.flowOn(Dispatchers.IO)

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
        tempFile.delete()
        return file.delete()
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
    fun downloadVoiceCommands(): Flow<DownloadState> = flow {
        emit(DownloadState.Idle)

        val nluDir = File(context.filesDir, "nlu_models")
        nluDir.mkdirs()
        val targetFile = File(nluDir, "xlm_roberta_base_int8.onnx")
        val tempFile = File(nluDir, "xlm_roberta_base_int8.onnx.tmp")
        val url = "https://huggingface.co/lelloman/xlm-roberta-base-onnx-int8/resolve/main/xlm_roberta_base_int8.onnx"

        try {
            val existingBytes = if (tempFile.exists()) tempFile.length() else 0L

            val requestBuilder = Request.Builder().url(url)
            if (existingBytes > 0) {
                requestBuilder.addHeader("Range", "bytes=$existingBytes-")
            }

            val response = client.newCall(requestBuilder.build()).execute()

            if (!response.isSuccessful && response.code != 206) {
                emit(DownloadState.Error("Download failed: HTTP ${response.code}"))
                return@flow
            }

            val body = response.body ?: run {
                emit(DownloadState.Error("Empty response body"))
                return@flow
            }

            val contentLength = body.contentLength()
            val totalBytes = if (response.code == 206) {
                existingBytes + contentLength
            } else {
                contentLength
            }

            val outputStream = FileOutputStream(tempFile, response.code == 206)
            val buffer = ByteArray(8192)
            var downloadedBytes = existingBytes
            var lastEmitTime = System.currentTimeMillis()

            body.byteStream().use { inputStream ->
                outputStream.use { output ->
                    while (true) {
                        val bytesRead = inputStream.read(buffer)
                        if (bytesRead == -1) break

                        output.write(buffer, 0, bytesRead)
                        downloadedBytes += bytesRead

                        val now = System.currentTimeMillis()
                        if (now - lastEmitTime >= 100) {
                            val progress = if (totalBytes > 0) {
                                downloadedBytes.toFloat() / totalBytes
                            } else {
                                0f
                            }
                            emit(DownloadState.Downloading(progress, downloadedBytes, totalBytes))
                            lastEmitTime = now
                        }
                    }
                }
            }

            if (tempFile.renameTo(targetFile)) {
                emit(DownloadState.Completed)
            } else {
                emit(DownloadState.Error("Failed to finalize download"))
            }

        } catch (e: Exception) {
            emit(DownloadState.Error("Download error: ${e.message}"))
        }
    }.flowOn(Dispatchers.IO)

    fun getStorageInfo(): StorageInfo {
        val modelsUsed = modelsDir.listFiles()
            ?.filter { it.isFile && it.name.endsWith(".gguf") }
            ?.sumOf { it.length() } ?: 0L

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
