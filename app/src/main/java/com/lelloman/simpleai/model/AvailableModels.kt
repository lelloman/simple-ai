package com.lelloman.simpleai.model

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
    const val SUPPORTS_TOOLS = true
}
