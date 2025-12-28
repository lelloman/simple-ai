package com.lelloman.simpleai.download

import android.content.Context
import android.os.StatFs
import com.lelloman.simpleai.model.AvailableModel
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

object DefaultModel {
    val CONFIG = ModelConfig(
        name = "SmolLM2-1.7B",
        url = "https://huggingface.co/bartowski/SmolLM2-1.7B-Instruct-GGUF/resolve/main/SmolLM2-1.7B-Instruct-Q4_K_M.gguf",
        fileName = "SmolLM2-1.7B-Instruct-Q4_K_M.gguf",
        expectedSizeMb = 1060
    )
}

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

    fun getModelFile(config: ModelConfig = DefaultModel.CONFIG): File {
        return File(modelsDir, config.fileName)
    }

    fun isModelDownloaded(config: ModelConfig = DefaultModel.CONFIG): Boolean {
        val file = getModelFile(config)
        return file.exists() && file.length() > 0
    }

    fun downloadModel(config: ModelConfig = DefaultModel.CONFIG): Flow<DownloadState> = flow {
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

    fun deleteModel(config: ModelConfig = DefaultModel.CONFIG): Boolean {
        val file = getModelFile(config)
        val tempFile = File(modelsDir, "${config.fileName}.tmp")
        tempFile.delete()
        return file.delete()
    }

    // ==================== AvailableModel support ====================

    fun getModelFile(model: AvailableModel): File {
        return File(modelsDir, model.fileName)
    }

    fun isModelDownloaded(model: AvailableModel): Boolean {
        val file = getModelFile(model)
        return file.exists() && file.length() > 0
    }

    fun downloadModel(model: AvailableModel): Flow<DownloadState> {
        val config = ModelConfig(
            name = model.name,
            url = model.url,
            fileName = model.fileName,
            expectedSizeMb = model.sizeMb
        )
        return downloadModel(config)
    }

    fun deleteModel(model: AvailableModel): Boolean {
        val file = getModelFile(model)
        val tempFile = File(modelsDir, "${model.fileName}.tmp")
        tempFile.delete()
        return file.delete()
    }

    fun getModelSizeBytes(model: AvailableModel): Long {
        val file = getModelFile(model)
        return if (file.exists()) file.length() else 0L
    }

    // ==================== Storage info ====================

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
