package com.lelloman.simpleai.download

import io.mockk.every
import io.mockk.mockk
import kotlinx.coroutines.*
import okhttp3.*
import okio.Buffer
import okio.ForwardingSource
import okio.buffer
import org.junit.Assert.*
import org.junit.Test

class ResponseCleanupTest {
    private class TrackedBody : ResponseBody() {
        var closed = false
        private val input = object : ForwardingSource(Buffer().writeUtf8("body")) {
            override fun close() { closed = true; super.close() }
        }.buffer()
        override fun contentType(): MediaType? = null
        override fun contentLength() = 4L
        override fun source() = input
    }

    private fun call(body: ResponseBody, code: Int = 200): Call {
        val response = Response.Builder().request(Request.Builder().url("https://model.example").build())
            .protocol(Protocol.HTTP_1_1).code(code).message("test").body(body).build()
        return mockk<Call>(relaxed = true).also { every { it.execute() } returns response }
    }

    @Test fun successClosesResponse() = runBlocking {
        val body = TrackedBody()
        call(body).withResponse { assertEquals("body", it.body!!.string()) }
        assertTrue(body.closed)
    }

    @Test fun earlyHttpFailureClosesUnreadBody() = runBlocking {
        val body = TrackedBody()
        call(body, 404).withResponse { if (!it.isSuccessful) return@withResponse }
        assertTrue(body.closed)
    }

    @Test fun exceptionBeforeReadingClosesBody() = runBlocking {
        val body = TrackedBody()
        try { call(body).withResponse { throw IllegalArgumentException("bad range") } }
        catch (_: IllegalArgumentException) { }
        assertTrue(body.closed)
    }

    @Test fun cancellationDuringResponseHandlingClosesBody() = runBlocking {
        val body = TrackedBody()
        val entered = CompletableDeferred<Unit>()
        val job = launch {
            call(body).withResponse { entered.complete(Unit); awaitCancellation() }
        }
        entered.await()
        job.cancelAndJoin()
        assertTrue(body.closed)
    }
}
