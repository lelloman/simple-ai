package com.lelloman.simpleai.api

import com.lelloman.simpleai.BuildConfig
import com.lelloman.simpleai.api.ApiResponse.Companion.toJson
import kotlinx.serialization.json.JsonElement

/**
 * Handles protocol version validation and response formatting.
 *
 * Version strategy:
 * - SERVICE_VERSION: Bumped every release
 * - MIN_PROTOCOL_VERSION: Bumped when dropping old protocol support
 * - MAX_PROTOCOL_VERSION: Bumped when adding new protocol features
 *
 * Client passes protocolVersion with each request:
 * - If protocolVersion < MIN_PROTOCOL_VERSION → client too old
 * - If protocolVersion > MAX_PROTOCOL_VERSION → SimpleAI too old
 * - Otherwise → respond using that protocol version
 */
object ProtocolHandler {

    val serviceVersion: Int
        get() = BuildConfig.SERVICE_VERSION

    val minProtocolVersion: Int
        get() = BuildConfig.MIN_PROTOCOL_VERSION

    val maxProtocolVersion: Int
        get() = BuildConfig.MAX_PROTOCOL_VERSION

    /**
     * Validate protocol version and return error response if invalid.
     *
     * @param requestedProtocol The protocol version requested by the client
     * @return Error response JSON if invalid, null if valid
     */
    fun validateProtocol(requestedProtocol: Int): String? {
        return when {
            requestedProtocol < minProtocolVersion -> {
                ApiResponse.error(
                    protocolVersion = minProtocolVersion,
                    code = ErrorCode.UNSUPPORTED_PROTOCOL,
                    message = "Client protocol $requestedProtocol is too old. Minimum supported: $minProtocolVersion. Please update your app."
                ).toJson()
            }
            requestedProtocol > maxProtocolVersion -> {
                ApiResponse.error(
                    protocolVersion = maxProtocolVersion,
                    code = ErrorCode.VERSION_TOO_OLD,
                    message = "SimpleAI protocol $maxProtocolVersion is older than requested $requestedProtocol. Please update SimpleAI."
                ).toJson()
            }
            else -> null
        }
    }

    /**
     * Create a success response JSON.
     */
    fun success(protocolVersion: Int, data: JsonElement): String {
        return ApiResponse.success(protocolVersion, data).toJson()
    }

    /**
     * Create an error response JSON.
     */
    fun error(protocolVersion: Int, code: ErrorCode, message: String, details: JsonElement? = null): String {
        return ApiResponse.error(protocolVersion, code, message, details).toJson()
    }

    /**
     * Clamp protocol version to supported range.
     * Use this to determine which protocol version to use in responses.
     */
    fun clampProtocol(requestedProtocol: Int): Int {
        return requestedProtocol.coerceIn(minProtocolVersion, maxProtocolVersion)
    }
}
