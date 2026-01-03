package com.lelloman.simpleai.api

import com.lelloman.simpleai.api.ApiResponse.Companion.toJson
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.put
import org.junit.Assert.*
import org.junit.Test

class ApiResponseTest {

    private val json = Json { ignoreUnknownKeys = true }

    @Test
    fun `success response has correct structure`() {
        val data = buildJsonObject { put("test", "value") }
        val response = ApiResponse.success(protocolVersion = 1, data = data)

        assertEquals("success", response.status)
        assertEquals(1, response.protocolVersion)
        assertNotNull(response.data)
        assertNull(response.error)
    }

    @Test
    fun `error response has correct structure`() {
        val response = ApiResponse.error(
            protocolVersion = 1,
            code = ErrorCode.CAPABILITY_NOT_READY,
            message = "Test error"
        )

        assertEquals("error", response.status)
        assertEquals(1, response.protocolVersion)
        assertNull(response.data)
        assertNotNull(response.error)
        assertEquals(ErrorCode.CAPABILITY_NOT_READY, response.error?.code)
        assertEquals("Test error", response.error?.message)
    }

    @Test
    fun `error response with details includes details`() {
        val details = buildJsonObject { put("extra", "info") }
        val response = ApiResponse.error(
            protocolVersion = 2,
            code = ErrorCode.INTERNAL_ERROR,
            message = "Something went wrong",
            details = details
        )

        assertNotNull(response.error?.details)
        assertEquals("info", response.error?.details?.jsonObject?.get("extra")?.jsonPrimitive?.content)
    }

    @Test
    fun `success response serializes to valid JSON`() {
        val data = buildJsonObject {
            put("intent", "test_intent")
            put("confidence", 0.95)
        }
        val response = ApiResponse.success(protocolVersion = 1, data = data)

        val jsonString = response.toJson()
        val parsed = json.parseToJsonElement(jsonString).jsonObject

        assertEquals("success", parsed["status"]?.jsonPrimitive?.content)
        assertEquals(1, parsed["protocolVersion"]?.jsonPrimitive?.content?.toInt())
        assertNotNull(parsed["data"])
        assertEquals("test_intent", parsed["data"]?.jsonObject?.get("intent")?.jsonPrimitive?.content)
    }

    @Test
    fun `error response serializes to valid JSON`() {
        val response = ApiResponse.error(
            protocolVersion = 1,
            code = ErrorCode.CLOUD_AUTH_FAILED,
            message = "Invalid token"
        )

        val jsonString = response.toJson()
        val parsed = json.parseToJsonElement(jsonString).jsonObject

        assertEquals("error", parsed["status"]?.jsonPrimitive?.content)
        assertEquals(1, parsed["protocolVersion"]?.jsonPrimitive?.content?.toInt())
        assertNotNull(parsed["error"])
        assertEquals("CLOUD_AUTH_FAILED", parsed["error"]?.jsonObject?.get("code")?.jsonPrimitive?.content)
        assertEquals("Invalid token", parsed["error"]?.jsonObject?.get("message")?.jsonPrimitive?.content)
    }

    @Test
    fun `ApiError contains all fields`() {
        val details = buildJsonObject { put("key", "value") }
        val error = ApiError(
            code = ErrorCode.ADAPTER_LOAD_FAILED,
            message = "Adapter corrupt",
            details = details
        )

        assertEquals(ErrorCode.ADAPTER_LOAD_FAILED, error.code)
        assertEquals("Adapter corrupt", error.message)
        assertNotNull(error.details)
    }

    @Test
    fun `ApiError details is optional`() {
        val error = ApiError(
            code = ErrorCode.INVALID_REQUEST,
            message = "Bad request"
        )

        assertNull(error.details)
    }

    @Test
    fun `all ErrorCode values can be serialized`() {
        for (code in ErrorCode.entries) {
            val response = ApiResponse.error(
                protocolVersion = 1,
                code = code,
                message = "Test"
            )
            val jsonString = response.toJson()
            assertTrue("Failed to serialize $code", jsonString.contains(code.name))
        }
    }

    @Test
    fun `response can be deserialized back`() {
        val original = ApiResponse.success(
            protocolVersion = 2,
            data = buildJsonObject { put("result", "ok") }
        )

        val jsonString = original.toJson()
        val deserialized = json.decodeFromString<ApiResponse>(jsonString)

        assertEquals(original.status, deserialized.status)
        assertEquals(original.protocolVersion, deserialized.protocolVersion)
        assertEquals(
            original.data?.jsonObject?.get("result")?.jsonPrimitive?.content,
            deserialized.data?.jsonObject?.get("result")?.jsonPrimitive?.content
        )
    }
}
