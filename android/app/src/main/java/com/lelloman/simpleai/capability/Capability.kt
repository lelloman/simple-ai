package com.lelloman.simpleai.capability

import kotlinx.serialization.Serializable

/**
 * Represents a SimpleAI capability with its current status.
 */
@Serializable
data class Capability(
    val id: CapabilityId,
    val name: String,
    val description: String,
    val status: CapabilityStatus
) {
    val isReady: Boolean
        get() = status is CapabilityStatus.Ready

    val isDownloading: Boolean
        get() = status is CapabilityStatus.Downloading

    val canDownload: Boolean
        get() = status is CapabilityStatus.NotDownloaded || status is CapabilityStatus.Error
}

/**
 * Translation capability has additional state for managing languages.
 */
@Serializable
data class TranslationCapability(
    val baseCapability: Capability,
    val downloadedLanguages: Set<String>,
    val availableLanguages: Set<String>
) {
    val hasLanguages: Boolean
        get() = downloadedLanguages.isNotEmpty()
}
