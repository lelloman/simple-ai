package com.lelloman.simpleai.nlu

import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock

/** Owns the full adapter selection/inference transaction and resource lifetime. */
internal class AdapterAccess {
    private val mutex = Mutex()
    suspend fun <T> withLock(block: suspend () -> T): T = mutex.withLock { block() }

    suspend fun <T> classify(
        requested: Pair<String, String>,
        current: () -> Pair<String, String>?,
        switch: suspend () -> Result<Unit>,
        infer: suspend () -> Result<T>
    ): Result<T> = withLock {
        if (current() != requested) {
            switch().exceptionOrNull()?.let { return@withLock Result.failure(it) }
        }
        infer()
    }
}
