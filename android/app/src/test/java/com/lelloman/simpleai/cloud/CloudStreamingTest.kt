package com.lelloman.simpleai.cloud

import kotlinx.coroutines.*
import kotlinx.serialization.json.JsonArray
import okhttp3.HttpUrl.Companion.toHttpUrl
import org.junit.Assert.*
import org.junit.Test
import java.net.ServerSocket
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import kotlin.concurrent.thread

class CloudStreamingTest {
    private fun serve(socket: java.net.Socket, afterFirst: (java.io.BufferedReader, java.io.OutputStream) -> Unit) {
        socket.use {
            socket.soTimeout = 5_000
            val reader = socket.getInputStream().bufferedReader()
            var length = 0
            while (true) {
                val line = reader.readLine() ?: error("Missing request")
                if (line.isEmpty()) break
                if (line.startsWith("Content-Length:", ignoreCase = true)) {
                    length = line.substringAfter(':').trim().toInt()
                }
            }
            val request = CharArray(length)
            var offset = 0
            while (offset < length) {
                val read = reader.read(request, offset, length - offset)
                check(read > 0)
                offset += read
            }
            assertTrue(String(request).contains("\"stream\":true"))
            val output = socket.getOutputStream()
            output.write(("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n" +
                "data: first\n\n").toByteArray())
            output.flush()
            afterFirst(reader, output)
        }
    }

    @Test fun deliversFirstEventBeforeServerCompletes() = runBlocking {
        ServerSocket(0).use { server ->
            val release = CountDownLatch(1)
            val serverResult = CompletableDeferred<Unit>()
            val worker = thread {
                try {
                    serve(server.accept()) { _, output ->
                        check(release.await(5, TimeUnit.SECONDS))
                        output.write("data: [DONE]\n\n".toByteArray())
                        output.flush()
                    }
                    serverResult.complete(Unit)
                } catch (e: Throwable) { serverResult.completeExceptionally(e) }
            }
            try {
                val first = CompletableDeferred<String>()
                val events = mutableListOf<String>()
                val call = async {
                    CloudLLMClient({ "https://unused.example" }, { "http://127.0.0.1:${server.localPort}/v1/chat/completions".toHttpUrl() }).streamChat(
                        JsonArray(emptyList()), null, null, null, "test-token"
                    ) {
                        events.add(it)
                        first.complete(it)
                    }
                }
                assertEquals("first", withTimeout(5_000) { first.await() })
                assertFalse(call.isCompleted)
                release.countDown()
                withTimeout(5_000) { call.await(); serverResult.await() }
                assertEquals(listOf("first", "[DONE]"), events)
            } finally {
                release.countDown()
                worker.join(5_000)
            }
        }
    }

    @Test fun cancellationClosesBlockedHttpRead() = runBlocking {
        ServerSocket(0).use { server ->
            val disconnected = CompletableDeferred<Unit>()
            val worker = thread {
                try {
                    serve(server.accept()) { reader, _ ->
                        assertEquals(-1, reader.read())
                    }
                    disconnected.complete(Unit)
                } catch (e: Throwable) { disconnected.completeExceptionally(e) }
            }
            try {
                val first = CompletableDeferred<Unit>()
                val call = launch {
                    CloudLLMClient({ "https://unused.example" }, { "http://127.0.0.1:${server.localPort}/v1/chat/completions".toHttpUrl() }).streamChat(
                        JsonArray(emptyList()), null, null, null, "test-token"
                    ) { first.complete(Unit) }
                }
                withTimeout(5_000) { first.await() }
                withTimeout(5_000) { call.cancelAndJoin(); disconnected.await() }
                assertTrue(call.isCancelled)
            } finally {
                worker.join(5_000)
            }
        }
    }
}
