package com.lelloman.simpleai.model

import kotlinx.coroutines.*
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock

/** Serializes the transition to idle against newly admitted work. */
internal class IdleResources(
    private val scope: CoroutineScope,
    private val idleMs: Long = 60_000,
    private val release: suspend () -> Unit
) {
    private val mutex = Mutex()
    private var active = 0
    private var idle: Job? = null
    suspend fun <T> withWork(block: suspend () -> T): T {
        mutex.withLock { idle?.cancel(); active++ }
        return try { block() } finally {
            withContext(NonCancellable) {
                mutex.withLock {
                    active--
                    if (active == 0) idle = scope.launch {
                        delay(idleMs)
                        mutex.withLock { if (active == 0) release() }
                    }
                }
            }
        }
    }
}
