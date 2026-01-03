package com.lelloman.simpleai.api

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.put
import org.junit.Assert.*
import org.junit.Test

class ProtocolHandlerTest {

    private val json = Json { ignoreUnknownKeys = true }

    // Note: These tests use the actual BuildConfig values set during compilation.
    // Current config: SERVICE_VERSION=1, MIN_PROTOCOL_VERSION=1, MAX_PROTOCOL_VERSION=1

    @Test
    fun `validateProtocol returns null for supported protocol`() {
        // Protocol 1 should be supported (min=1, max=1)
        val result = ProtocolHandler.validateProtocol(1)
        assertNull(result)
    }

    @Test
    fun `validateProtocol returns error for protocol below minimum`() {
        // Protocol 0 should be rejected
        val result = ProtocolHandler.validateProtocol(0)

        assertNotNull(result)
        val parsed = json.parseToJsonElement(result!!).jsonObject
        assertEquals("error", parsed["status"]?.jsonPrimitive?.content)
        assertEquals("UNSUPPORTED_PROTOCOL", parsed["error"]?.jsonObject?.get("code")?.jsonPrimitive?.content)
    }

    @Test
    fun `validateProtocol returns error for protocol above maximum`() {
        // Protocol 999 should be rejected as too new
        val result = ProtocolHandler.validateProtocol(999)

        assertNotNull(result)
        val parsed = json.parseToJsonElement(result!!).jsonObject
        assertEquals("error", parsed["status"]?.jsonPrimitive?.content)
        assertEquals("VERSION_TOO_OLD", parsed["error"]?.jsonObject?.get("code")?.jsonPrimitive?.content)
    }

    @Test
    fun `success creates valid JSON response`() {
        val data = buildJsonObject { put("test", "data") }
        val result = ProtocolHandler.success(1, data)

        val parsed = json.parseToJsonElement(result).jsonObject
        assertEquals("success", parsed["status"]?.jsonPrimitive?.content)
        assertEquals(1, parsed["protocolVersion"]?.jsonPrimitive?.content?.toInt())
        assertEquals("data", parsed["data"]?.jsonObject?.get("test")?.jsonPrimitive?.content)
    }

    @Test
    fun `error creates valid JSON response`() {
        val result = ProtocolHandler.error(1, ErrorCode.CAPABILITY_NOT_READY, "Not ready")

        val parsed = json.parseToJsonElement(result).jsonObject
        assertEquals("error", parsed["status"]?.jsonPrimitive?.content)
        assertEquals(1, parsed["protocolVersion"]?.jsonPrimitive?.content?.toInt())
        assertEquals("CAPABILITY_NOT_READY", parsed["error"]?.jsonObject?.get("code")?.jsonPrimitive?.content)
        assertEquals("Not ready", parsed["error"]?.jsonObject?.get("message")?.jsonPrimitive?.content)
    }

    @Test
    fun `error with details includes details in JSON`() {
        val details = buildJsonObject { put("progress", 0.5) }
        val result = ProtocolHandler.error(
            1,
            ErrorCode.CAPABILITY_DOWNLOADING,
            "Downloading",
            details
        )

        val parsed = json.parseToJsonElement(result).jsonObject
        val errorDetails = parsed["error"]?.jsonObject?.get("details")?.jsonObject
        assertNotNull(errorDetails)
        assertEquals(0.5, errorDetails?.get("progress")?.jsonPrimitive?.content?.toDouble())
    }

    @Test
    fun `clampProtocol returns protocol within range`() {
        // With min=1, max=1, all values should clamp to 1
        assertEquals(1, ProtocolHandler.clampProtocol(0))
        assertEquals(1, ProtocolHandler.clampProtocol(1))
        assertEquals(1, ProtocolHandler.clampProtocol(999))
    }

    @Test
    fun `serviceVersion is accessible`() {
        // Should be 1 based on build config
        assertTrue(ProtocolHandler.serviceVersion >= 1)
    }

    @Test
    fun `minProtocolVersion is accessible`() {
        assertTrue(ProtocolHandler.minProtocolVersion >= 1)
    }

    @Test
    fun `maxProtocolVersion is accessible`() {
        assertTrue(ProtocolHandler.maxProtocolVersion >= ProtocolHandler.minProtocolVersion)
    }

    @Test
    fun `validateProtocol error message for old client is helpful`() {
        val result = ProtocolHandler.validateProtocol(0)
        assertNotNull(result)
        assertTrue(result!!.contains("too old"))
        assertTrue(result.contains("update your app"))
    }

    @Test
    fun `validateProtocol error message for old server is helpful`() {
        val result = ProtocolHandler.validateProtocol(999)
        assertNotNull(result)
        assertTrue(result!!.contains("update SimpleAI"))
    }
}
