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
    const val URL = "https://huggingface.co/bartowski/Qwen_Qwen3-1.7B-GGUF/resolve/dcb19155b962dbb6389f4691a982043a8e651022/Qwen_Qwen3-1.7B-Q4_K_M.gguf"
    const val FILE_NAME = "Qwen_Qwen3-1.7B-Q4_K_M.gguf"
    const val SIZE_MB = 1280
    const val SIZE_BYTES = 1282439584L
    const val SHA256 = "72c5c3cb38fa32d5256e2fe30d03e7a64c6c79e668ad84057e3bd66e250b24fb"
    const val SUPPORTS_TOOLS = true
}
