package com.lelloman.simpleai.api

import com.lelloman.simpleai.cloud.CloudLLMClient
import kotlinx.coroutines.*
import org.junit.Assert.*
import org.junit.Test
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import kotlin.concurrent.thread

class RequestContractTest {
    @Test fun rejectsMalformedMessagesAndUnboundedGeneration() {
        listOf("[1]", "[{}]", "[]", "[{\"role\":\"user\",\"content\":{}}]").forEach {
            assertTrue(runCatching { RequestValidation.messages(it) }.isFailure)
        }
        listOf(Float.NaN, Float.POSITIVE_INFINITY, -1f, 3f).forEach {
            assertTrue(runCatching { RequestValidation.generation("hello", 12, it) }.isFailure)
        }
        assertTrue(runCatching { RequestValidation.generation("hello", 0, 1f) }.isFailure)
        assertTrue(runCatching { RequestValidation.generation("hello", 2049, 1f) }.isFailure)
        RequestValidation.generation("hello", 2048, 0f)
    }
    @Test fun cloudRequestsUseConfiguredFastClassInsteadOfAccountDependentDefault() {
        val messages = kotlinx.serialization.json.Json.parseToJsonElement("""[{"role":"user","content":"hello"}]""") as kotlinx.serialization.json.JsonArray
        val tools = kotlinx.serialization.json.JsonArray(emptyList())
        val body = CloudLLMClient { "" }.buildRequestBody(messages, tools, "conversation-1")
        assertEquals(kotlinx.serialization.json.JsonPrimitive("class:fast"), body["model"])
        assertEquals(messages, body["messages"])
        assertEquals(kotlinx.serialization.json.JsonPrimitive("conversation-1"), body["prompt_cache_key"])
    }
    @Test fun nullUsageIsOptional() {
        val response = CloudLLMClient { "" }.parseResponse("""{"choices":[{"message":{"role":"assistant","content":"ok"}}],"usage":null}""")
        assertNull(response.usage)
        assertEquals("ok", response.content)
    }
    @Test fun callerCanOnlyCancelOwnJobAndRegistryCleansUp() {
        val requests = ActiveRequests()
        val started = CountDownLatch(1)
        var result: Throwable? = null
        val worker = thread {
            result = runCatching { requests.run(12) { started.countDown(); awaitCancellation() } }.exceptionOrNull()
        }
        assertTrue(started.await(2, TimeUnit.SECONDS))
        assertFalse(requests.cancel(13))
        assertTrue(requests.cancel(12))
        worker.join(2000)
        assertFalse(worker.isAlive)
        assertTrue(result is CancellationException)
        assertFalse(requests.cancel(12))
        assertEquals("next", requests.run(12) { "next" })
    }
    @Test fun deadlineCancelsSuspendedWork() {
        val requests = ActiveRequests(20)
        assertTrue(runCatching { requests.run(1) { awaitCancellation() } }.exceptionOrNull() is TimeoutCancellationException)
        assertFalse(requests.cancel(1))
    }
}
