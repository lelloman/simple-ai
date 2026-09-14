package com.lelloman.simpleai.model

import com.lelloman.simpleai.capability.CapabilityStatus
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.emptyFlow
import kotlinx.coroutines.test.*
import org.junit.Assert.*
import org.junit.Test

@OptIn(ExperimentalCoroutinesApi::class)
class IdleResourcesTest {
    @Test fun resourcesStayResidentDuringWorkAndReleaseAfterIdle() = runTest {
        var releases = 0
        val owner = IdleResources(backgroundScope, 100) { releases++ }
        val done = CompletableDeferred<Unit>()
        val job = launch { owner.withWork { done.await() } }
        runCurrent()
        advanceTimeBy(200)
        assertEquals(0, releases)
        done.complete(Unit)
        job.join()
        advanceTimeBy(90)
        owner.withWork { }
        advanceTimeBy(90)
        assertEquals(0, releases)
        advanceTimeBy(11)
        runCurrent()
        assertEquals(1, releases)
    }
    @Test fun inspectionDoesNotLoadAndIdleUnloadKeepsDownloadVisible() = runTest {
        var loads = 0
        var disposed = 0
        var status: CapabilityStatus = CapabilityStatus.Checking
        val model = ManagedModel(1, { true }, { emptyFlow() },
            load = { loads++; Result.success("engine") }, dispose = { disposed++ }, publish = { status = it })
        model.inspect()
        assertEquals(0, loads)
        assertEquals(CapabilityStatus.Downloaded, status)
        model.initialize()
        model.unload()
        assertNull(model.engine)
        assertEquals(1, disposed)
        assertEquals(CapabilityStatus.Downloaded, status)
        model.initialize()
        assertEquals(2, loads)
    }
}
