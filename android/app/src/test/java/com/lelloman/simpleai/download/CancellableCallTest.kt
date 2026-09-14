package com.lelloman.simpleai.download

import io.mockk.every
import io.mockk.mockk
import io.mockk.verify
import kotlinx.coroutines.*
import okhttp3.Call
import org.junit.Test
import java.io.IOException
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import org.junit.Assert.*

class CancellableCallTest {
    @Test fun cancellingOwnerUnblocksSocketRead() = runBlocking {
        val entered = CountDownLatch(1)
        val cancelled = CountDownLatch(1)
        val call = mockk<Call>()
        every { call.execute() } answers {
            entered.countDown()
            check(cancelled.await(2, TimeUnit.SECONDS)) { "Socket was not cancelled" }
            throw IOException("cancelled")
        }
        every { call.cancel() } answers { cancelled.countDown(); Unit }
        val job = launch(Dispatchers.IO) {
            try { call.withResponse { error("Unexpected response") } }
            catch (e: IOException) { currentCoroutineContext().ensureActive(); throw e }
        }
        assertTrue(entered.await(1, TimeUnit.SECONDS))
        withTimeout(1000) { job.cancelAndJoin() }
        verify(exactly = 1) { call.cancel() }
    }
}
