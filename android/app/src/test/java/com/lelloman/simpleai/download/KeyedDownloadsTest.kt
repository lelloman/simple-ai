package com.lelloman.simpleai.download

import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.test.*
import org.junit.Assert.*
import org.junit.Test

@OptIn(kotlinx.coroutines.ExperimentalCoroutinesApi::class)
class KeyedDownloadsTest {
    @Test fun completionDoesNotClearAnotherLanguageOrAllowDuplicates() = runTest {
        val gates = mapOf("it" to CompletableDeferred<Unit>(), "fr" to CompletableDeferred<Unit>())
        val calls = mutableListOf<String>()
        val downloads = KeyedDownloads(backgroundScope) { language -> calls.add(language); gates.getValue(language).await(); Result.success(Unit) }
        downloads.start("it")
        downloads.start("fr")
        downloads.start("it")
        runCurrent()
        assertEquals(listOf("it", "fr"), calls)
        gates.getValue("it").complete(Unit)
        runCurrent()
        assertEquals(setOf("fr"), downloads.active.value)
        gates.getValue("fr").complete(Unit)
        runCurrent()
        assertTrue(downloads.active.value.isEmpty())
    }

    @Test fun acknowledgingOneErrorRetainsOtherLanguageErrors() = runTest {
        val downloads = KeyedDownloads(backgroundScope) { Result.failure(IllegalStateException("network")) }
        downloads.start("it")
        downloads.start("fr")
        runCurrent()
        downloads.clearError("it")
        assertEquals(mapOf("fr" to "network"), downloads.errors.value)
        assertTrue(downloads.active.value.isEmpty())
    }
}
