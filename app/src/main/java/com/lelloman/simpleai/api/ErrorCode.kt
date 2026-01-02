package com.lelloman.simpleai.api

import kotlinx.serialization.Serializable

/**
 * Standard error codes for SimpleAI AIDL responses.
 */
@Serializable
enum class ErrorCode {
    /** SimpleAI version is older than client's required protocol */
    VERSION_TOO_OLD,

    /** Requested protocol version not supported by this SimpleAI */
    UNSUPPORTED_PROTOCOL,

    /** Required capability not downloaded */
    CAPABILITY_NOT_READY,

    /** Capability download in progress */
    CAPABILITY_DOWNLOADING,

    /** Capability failed to initialize */
    CAPABILITY_ERROR,

    /** Failed to apply LoRA adapter */
    ADAPTER_LOAD_FAILED,

    /** Malformed request parameters */
    INVALID_REQUEST,

    /** Cloud endpoint rejected auth token */
    CLOUD_AUTH_FAILED,

    /** Cloud endpoint unreachable */
    CLOUD_UNAVAILABLE,

    /** Requested translation language not downloaded */
    TRANSLATION_LANGUAGE_NOT_AVAILABLE,

    /** Unexpected internal error */
    INTERNAL_ERROR
}
