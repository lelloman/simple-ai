package com.lelloman.simpleai.capability

import kotlinx.serialization.Serializable

/**
 * Status of a capability.
 */
@Serializable
sealed class CapabilityStatus {

    /** Capability requires download before use */
    @Serializable
    data class NotDownloaded(
        val totalBytes: Long
    ) : CapabilityStatus()

    /** Capability is currently downloading */
    @Serializable
    data class Downloading(
        val downloadedBytes: Long,
        val totalBytes: Long
    ) : CapabilityStatus() {
        val progress: Float
            get() = if (totalBytes > 0) downloadedBytes.toFloat() / totalBytes else 0f
    }

    /** Capability is ready to use */
    @Serializable
    data object Ready : CapabilityStatus()

    /** Capability failed to initialize or download */
    @Serializable
    data class Error(
        val message: String,
        val canRetry: Boolean = true
    ) : CapabilityStatus()
}
