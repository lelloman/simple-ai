package com.lelloman.simpleai.capability

import kotlinx.serialization.Serializable

/**
 * Identifiers for SimpleAI capabilities.
 */
@Serializable
enum class CapabilityId {
    VOICE_COMMANDS,
    TRANSLATION,
    CLOUD_AI,
    LOCAL_AI
}
