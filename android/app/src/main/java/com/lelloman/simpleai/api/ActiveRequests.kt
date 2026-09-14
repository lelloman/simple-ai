package com.lelloman.simpleai.api

import kotlinx.coroutines.*
import java.util.concurrent.ConcurrentHashMap

/** One active request per UID, enforced by CallerBudget before entering here. */
internal class ActiveRequests(private val timeoutMs: Long = 120_000) {
    private val jobs = ConcurrentHashMap<Int, Job>()
    fun cancel(uid: Int): Boolean = jobs[uid]?.let { it.cancel(); true } ?: false
    fun run(uid: Int, block: suspend () -> String): String {
        val job = Job()
        check(jobs.putIfAbsent(uid, job) == null)
        return try {
            runBlocking(job + Dispatchers.IO) { withTimeout(timeoutMs) { block() } }
        } finally {
            jobs.remove(uid, job)
            job.cancel()
        }
    }
}
