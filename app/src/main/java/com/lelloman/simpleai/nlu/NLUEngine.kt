package com.lelloman.simpleai.nlu

import kotlinx.serialization.Serializable
import kotlinx.serialization.json.JsonObject

/**
 * Classification result from the NLU model.
 */
@Serializable
data class ClassificationResult(
    val intent: String,
    val intentConfidence: Float,
    val slots: Map<String, List<String>>,
    val rawSlotLabels: List<String>
)

/**
 * Adapter configuration for app-specific NLU models.
 */
@Serializable
data class NLUAdapter(
    val id: String,
    val name: String,
    val version: String,
    val baseModel: String,  // e.g., "xlm-roberta-base"
    val maxLength: Int,
    val intents: List<String>,
    val slotLabels: List<String>
) {
    val slotTypes: List<String>
        get() = slotLabels.filter { it.startsWith("B-") }
            .map { it.removePrefix("B-") }
}

/**
 * Interface for NLU classification engines.
 */
interface NLUEngine {
    /**
     * Whether the engine is ready for inference.
     */
    val isReady: Boolean

    /**
     * List of loaded adapters.
     */
    val adapters: List<NLUAdapter>

    /**
     * Load the base model and all bundled adapters.
     * @return Result indicating success or failure
     */
    suspend fun initialize(): Result<Unit>

    /**
     * Classify input text using a specific adapter.
     * @param text Input text to classify
     * @param adapterId ID of the adapter to use
     * @return Classification result with intent and slots
     */
    suspend fun classify(text: String, adapterId: String): Result<ClassificationResult>

    /**
     * Release all resources.
     */
    fun release()
}
