package com.lelloman.simpleai.download

import kotlinx.coroutines.*
import kotlinx.coroutines.flow.*
import okhttp3.OkHttpClient
import okhttp3.Request
import java.io.File
import java.io.FileOutputStream
import java.security.MessageDigest

internal class ResumableDownload(private val client: OkHttpClient) {
    fun download(config: ModelConfig, target: File): Flow<DownloadState> = flow {
        emit(DownloadState.Idle)
        target.parentFile?.mkdirs()
        val partial = File(target.path + ".tmp")
        val identity = File(partial.path + ".identity")
        try {
            val saved = if (identity.exists()) identity.readLines() else emptyList()
            val etag = saved.getOrNull(1)?.takeIf { saved.firstOrNull() == config.url && it.startsWith('"') }
            if (config.sha256 == null && etag == null && partial.exists()) check(partial.delete())
            var restarted = false
            while (true) {
                val offset = if (partial.exists()) partial.length() else 0L
                var restart = false
                val request = Request.Builder().url(config.url).header("Accept-Encoding", "identity")
                if (offset > 0) {
                    request.header("Range", "bytes=$offset-")
                    etag?.let { request.header("If-Range", it) }
                }
                client.newCall(request.build()).withResponse { response ->
                    if (response.code == 416 && !restarted) {
                        // Even an apparently complete partial is re-requested so
                        // a server-side artifact change cannot be mistaken for success.
                        check(!partial.exists() || partial.delete())
                        identity.delete()
                        restart = true
                        return@withResponse
                    }
                    check(response.code == 200 || response.code == 206) { "Download failed: HTTP ${response.code}" }
                    val body = checkNotNull(response.body) { "Empty response body" }
                    val length = body.contentLength()
                    val append = response.code == 206
                    val total: Long
                    if (append) {
                        val range = Regex("bytes (\\d+)-(\\d+)/(\\d+)").matchEntire(response.header("Content-Range") ?: "")
                            ?: error("Invalid Content-Range")
                        val (start, end, size) = range.destructured.toList().map { it.toLong() }
                        check(start == offset && end >= start && end < size) { "Mismatched Content-Range" }
                        check(length < 0 || length == end - start + 1) { "Range length mismatch" }
                        check(end == size - 1) { "Incomplete range response" }
                        if (config.sha256 == null) check(etag != null && response.header("ETag") == etag) { "Artifact identity changed" }
                        total = size
                    } else total = config.expectedBytes ?: length
                    check(total > 0) { "Unknown or empty model size" }
                    config.expectedBytes?.let { check(total == it && (append || length < 0 || length == it)) { "Unexpected model size" } }
                    val responseTag = response.header("ETag")
                    if (responseTag != null && responseTag.startsWith('"')) identity.writeText(config.url + "\n" + responseTag)
                    else identity.delete()
                    var received = if (append) offset else 0L
                    emit(DownloadState.Downloading(received.toFloat() / total, received, total))
                    var lastEmit = System.currentTimeMillis()
                    FileOutputStream(partial, append).use { output ->
                        body.byteStream().use { input ->
                            val buffer = ByteArray(64 * 1024)
                            while (true) {
                                currentCoroutineContext().ensureActive()
                                val count = input.read(buffer)
                                if (count < 0) break
                                check(received + count <= total) { "Model exceeds expected size" }
                                output.write(buffer, 0, count)
                                received += count
                                if (System.currentTimeMillis() - lastEmit >= 100) {
                                    emit(DownloadState.Downloading(received.toFloat() / total, received, total))
                                    lastEmit = System.currentTimeMillis()
                                }
                            }
                        }
                        output.fd.sync()
                    }
                    check(received == total) { "Incomplete download: $received of $total bytes" }
                    config.sha256?.let { expected ->
                        if (sha256(partial) != expected) {
                            partial.delete()
                            identity.delete()
                            error("Model checksum mismatch. Retry to download a clean copy.")
                        }
                    }
                    check(partial.renameTo(target)) { "Failed to finalize download" }
                    identity.delete()
                    emit(DownloadState.Completed)
                }
                if (!restart) break
                restarted = true
            }
        } catch (e: CancellationException) { throw e }
        catch (e: Exception) {
            currentCoroutineContext().ensureActive()
            emit(DownloadState.Error(e.message ?: "Download failed"))
        }
    }.flowOn(Dispatchers.IO)

    private suspend fun sha256(file: File): String {
        val digest = MessageDigest.getInstance("SHA-256")
        file.inputStream().use { input ->
            val buffer = ByteArray(64 * 1024)
            while (true) {
                currentCoroutineContext().ensureActive()
                val count = input.read(buffer)
                if (count < 0) break
                digest.update(buffer, 0, count)
            }
        }
        return digest.digest().joinToString("") { "%02x".format(it) }
    }
}
