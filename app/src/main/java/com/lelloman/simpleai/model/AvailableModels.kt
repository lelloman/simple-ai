package com.lelloman.simpleai.model

import com.lelloman.simpleai.chat.ModelFormat

// Legacy EngineType for backward compatibility with LLMService
// TODO: Remove in Phase 7 cleanup when LLMService is removed
@Deprecated("Legacy - will be removed with LLMService")
enum class EngineType {
    LLAMA_CPP,
    EXECUTORCH
}

/**
 * Local AI model configuration.
 *
 * SimpleAI uses a single fixed model (Qwen 3 1.7B) for local AI capability.
 * Users cannot choose models - they can only download or delete the one model.
 */
object LocalAIModel {
    const val ID = "qwen3-1.7b"
    const val NAME = "Qwen 3 1.7B"
    const val DESCRIPTION = "On-device language model with tool support"
    const val URL = "https://huggingface.co/bartowski/Qwen_Qwen3-1.7B-GGUF/resolve/main/Qwen_Qwen3-1.7B-Q4_K_M.gguf"
    const val FILE_NAME = "Qwen_Qwen3-1.7B-Q4_K_M.gguf"
    const val SIZE_MB = 1280
    const val SIZE_BYTES = SIZE_MB * 1024L * 1024L
    val FORMAT = ModelFormat.HERMES
    const val SUPPORTS_TOOLS = true
}

// Legacy support for existing code that references AvailableModels
// TODO: Remove in Phase 7 cleanup
@Suppress("DEPRECATION")
@Deprecated("Use LocalAIModel instead", ReplaceWith("LocalAIModel"))
data class AvailableModel(
    val id: String,
    val name: String,
    val description: String,
    val url: String,
    val fileName: String,
    val sizeMb: Int,
    val format: ModelFormat,
    val supportsTools: Boolean,
    val engineType: EngineType = EngineType.LLAMA_CPP,
    val tokenizerUrl: String? = null
)

@Deprecated("Use LocalAIModel instead")
object AvailableModels {
    val DEFAULT_MODEL_ID = LocalAIModel.ID

    @Suppress("DEPRECATION")
    val ALL = listOf(
        AvailableModel(
            id = LocalAIModel.ID,
            name = LocalAIModel.NAME,
            description = LocalAIModel.DESCRIPTION,
            url = LocalAIModel.URL,
            fileName = LocalAIModel.FILE_NAME,
            sizeMb = LocalAIModel.SIZE_MB,
            format = LocalAIModel.FORMAT,
            supportsTools = LocalAIModel.SUPPORTS_TOOLS
        )
    )

    @Suppress("DEPRECATION")
    fun findById(id: String): AvailableModel? = ALL.find { it.id == id }

    @Suppress("DEPRECATION")
    fun getDefault(): AvailableModel = ALL.first()
}
