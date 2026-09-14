package com.lelloman.simpleai.download

import kotlinx.coroutines.*
import okhttp3.Call
import okhttp3.Response

/** Cancels blocking socket reads as soon as the owning job is cancelled. */
internal suspend fun <T> Call.withResponse(block: suspend (Response) -> T): T = coroutineScope {
    val watcher = launch(Dispatchers.IO, start = CoroutineStart.UNDISPATCHED) {
        try { awaitCancellation() } finally { this@withResponse.cancel() }
    }
    try {
        execute().use { block(it) }
    } finally {
        watcher.cancel()
    }
}
