package com.lelloman.simpleai.model

import com.lelloman.simpleai.capability.CapabilityStatus
import com.lelloman.simpleai.download.DownloadState
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.launch
import kotlinx.coroutines.test.runCurrent
import kotlinx.coroutines.test.runTest
import org.junit.Assert.*
import org.junit.Test

@OptIn(kotlinx.coroutines.ExperimentalCoroutinesApi::class)
class ManagedModelTest {
    @Test fun `download activates an initially missing model for clients without restarting`() = runTest {
        for (size in listOf(534_000_000L, LocalAIModel.SIZE_BYTES)) {
            var exists = false
            var status: CapabilityStatus = CapabilityStatus.NotDownloaded(size)
            val loaded = CompletableDeferred<Unit>()
            val model = ManagedModel(
                size, { exists },
                { flow { emit(DownloadState.Idle); exists = true; emit(DownloadState.Completed) } },
                load = { loaded.await(); Result.success({ "client response" }) },
                dispose = {}, publish = { status = it }
            )
            model.initialize()
            assertTrue(status is CapabilityStatus.NotDownloaded)
            val job = launch { model.downloadAndActivate() }
            runCurrent()
            assertNull(model.engine)
            assertFalse(status is CapabilityStatus.Ready)
            loaded.complete(Unit)
            job.join()
            assertEquals(CapabilityStatus.Ready, status)
            assertEquals("client response", model.engine!!())
        }
    }

    @Test fun `loading failure is never advertised ready and can retry`() = runTest {
        var attempts = 0
        val states = mutableListOf<CapabilityStatus>()
        val model = ManagedModel(1, { true }, { flow { emit(DownloadState.Completed) } },
            load = { if (++attempts == 1) Result.failure(IllegalStateException("bad model")) else Result.success("loaded") },
            dispose = {}, publish = states::add)
        model.downloadAndActivate()
        assertTrue(states.last() is CapabilityStatus.Error)
        assertNull(model.engine)
        model.downloadAndActivate()
        assertEquals("loaded", model.engine)
        assertEquals(CapabilityStatus.Ready, states.last())
    }
}
