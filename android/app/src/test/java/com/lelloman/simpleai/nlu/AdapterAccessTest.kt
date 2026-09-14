package com.lelloman.simpleai.nlu

import kotlinx.coroutines.*
import org.junit.Assert.*
import org.junit.Test

class AdapterAccessTest {
    private fun concurrent(secondId: String, secondVersion: String) = runBlocking {
        val access = AdapterAccess()
        var selected: Pair<String, String>? = null
        val entered = CompletableDeferred<Unit>()
        val finish = CompletableDeferred<Unit>()
        val first = async {
            access.classify("a" to "1", { selected }, {
                selected = "a" to "1"; Result.success(Unit)
            }) {
                entered.complete(Unit)
                finish.await()
                Result.success(selected)
            }
        }
        entered.await()
        val second = async {
            access.classify(secondId to secondVersion, { selected }, {
                selected = secondId to secondVersion; Result.success(Unit)
            }) { Result.success(selected) }
        }
        yield()
        assertEquals("a" to "1", selected)
        finish.complete(Unit)
        assertEquals("a" to "1", first.await().getOrThrow())
        assertEquals(secondId to secondVersion, second.await().getOrThrow())
    }

    @Test fun differentAdapters() = concurrent("b", "1")
    @Test fun differentVersions() = concurrent("a", "2")

    @Test fun cancellationReleasesAccessForCleanup() = runBlocking {
        val access = AdapterAccess()
        val entered = CompletableDeferred<Unit>()
        val task = launch {
            access.classify<Unit>("a" to "1", { null }, { Result.success(Unit) }) {
                entered.complete(Unit)
                awaitCancellation()
            }
        }
        entered.await()
        task.cancelAndJoin()
        var released = false
        withTimeout(1000) { access.withLock { released = true } }
        assertTrue(released)
    }
}
