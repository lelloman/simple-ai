package com.lelloman.simpleai.api

import kotlinx.serialization.Serializable
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonElement

/**
 * Standard response format for all SimpleAI AIDL methods.
 *
 * All responses are JSON strings with either:
 * - Success: {"status": "success", "protocolVersion": 1, "data": {...}}
 * - Error: {"status": "error", "protocolVersion": 1, "error": {"code": "...", "message": "...", "details": {...}}}
 */
@Serializable
data class ApiResponse(
    val status: String,
    val protocolVersion: Int,
    val data: JsonElement? = null,
    val error: ApiError? = null
) {
    companion object {
        private val json = Json {
            encodeDefaults = true
            ignoreUnknownKeys = true
        }

        fun success(protocolVersion: Int, data: JsonElement): ApiResponse {
            return ApiResponse(
                status = "success",
                protocolVersion = protocolVersion,
                data = data
            )
        }

        fun error(protocolVersion: Int, code: ErrorCode, message: String, details: JsonElement? = null): ApiResponse {
            return ApiResponse(
                status = "error",
                protocolVersion = protocolVersion,
                error = ApiError(code = code, message = message, details = details)
            )
        }

        fun ApiResponse.toJson(): String = json.encodeToString(this)
    }
}

@Serializable
data class ApiError(
    val code: ErrorCode,
    val message: String,
    val details: JsonElement? = null
)
