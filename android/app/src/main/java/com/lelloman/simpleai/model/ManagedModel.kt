package com.lelloman.simpleai.model

import com.lelloman.simpleai.capability.CapabilityStatus
import com.lelloman.simpleai.download.DownloadState
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock

/** One serialized download/activation lifecycle, shared by UI and service clients. */
class ManagedModel<E>(
    private val size: Long,
    private val exists: () -> Boolean,
    private val download: () -> Flow<DownloadState>,
    private val load: suspend () -> Result<E>,
    private val dispose: (E) -> Unit,
    private val publish: (CapabilityStatus) -> Unit
) {
    private val mutex = Mutex()
    @Volatile var engine: E? = null
        private set
    private var initialized = false

    suspend fun inspect() = mutex.withLock {
        if (!initialized && engine == null) publish(if (exists()) CapabilityStatus.Downloaded else CapabilityStatus.NotDownloaded(size))
    }

    suspend fun initialize() = mutex.withLock {
        if (!initialized) activate()
    }

    suspend fun downloadAndActivate() = mutex.withLock {
        try {
            if (engine != null) return@withLock
            download().collect { state ->
                when (state) {
                    DownloadState.Idle -> publish(CapabilityStatus.Downloading(0, size))
                    is DownloadState.Downloading -> publish(CapabilityStatus.Downloading(state.downloadedBytes, state.totalBytes))
                    DownloadState.Completed -> activate()
                    is DownloadState.Error -> publish(CapabilityStatus.Error(state.message))
                }
            }
        } catch (e: CancellationException) {
            publish(CapabilityStatus.Error("Download paused. Retry to resume from saved progress."))
            throw e
        } catch (e: Exception) {
            publish(CapabilityStatus.Error(e.message ?: "Model download failed"))
        }
    }

    private suspend fun activate() {
        initialized = true
        if (!exists()) {
            publish(CapabilityStatus.NotDownloaded(size))
            return
        }
        publish(CapabilityStatus.Loading)
        try {
            load().fold(
                onSuccess = { engine = it; publish(CapabilityStatus.Ready) },
                onFailure = { publish(CapabilityStatus.Error(it.message ?: "Model loading failed")) }
            )
        } catch (e: CancellationException) { initialized = false; throw e }
        catch (e: Exception) { publish(CapabilityStatus.Error(e.message ?: "Model loading failed")) }
    }

    suspend fun unload() = mutex.withLock {
        val old = engine
        engine = null
        old?.let(dispose)
        initialized = false
        publish(if (exists()) CapabilityStatus.Downloaded else CapabilityStatus.NotDownloaded(size))
    }

    suspend fun delete(deleteFiles: () -> Boolean) = mutex.withLock {
        val oldEngine = engine
        engine = null // Prevent new clients obtaining a native session during deletion.
        initialized = true
        try {
            oldEngine?.let(dispose) // Engine locks wait for any current inference to finish.
            check(deleteFiles()) { "Could not delete model files. Try again." }
            publish(CapabilityStatus.NotDownloaded(size))
        } catch (e: Exception) {
            publish(CapabilityStatus.Error(e.message ?: "Model deletion failed"))
        }
    }
}
