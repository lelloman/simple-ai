package com.lelloman.simpleai.model

data class AvailableModel(
    val id: String,
    val name: String,
    val description: String,
    val url: String,
    val fileName: String,
    val sizeMb: Int
)

object AvailableModels {
    val DEFAULT_MODEL_ID = "qwen3-1.7b"

    val ALL = listOf(
        AvailableModel(
            id = "qwen3-0.6b",
            name = "Qwen 3 0.6B",
            description = "Smallest, fastest. Good for simple tasks.",
            url = "https://huggingface.co/bartowski/Qwen_Qwen3-0.6B-GGUF/resolve/main/Qwen_Qwen3-0.6B-Q4_K_M.gguf",
            fileName = "Qwen_Qwen3-0.6B-Q4_K_M.gguf",
            sizeMb = 430
        ),
        AvailableModel(
            id = "qwen3-1.7b",
            name = "Qwen 3 1.7B",
            description = "Balanced size and quality. Recommended.",
            url = "https://huggingface.co/bartowski/Qwen_Qwen3-1.7B-GGUF/resolve/main/Qwen_Qwen3-1.7B-Q4_K_M.gguf",
            fileName = "Qwen_Qwen3-1.7B-Q4_K_M.gguf",
            sizeMb = 1280
        ),
        AvailableModel(
            id = "qwen3-4b",
            name = "Qwen 3 4B",
            description = "Best quality. Requires more RAM.",
            url = "https://huggingface.co/bartowski/Qwen_Qwen3-4B-GGUF/resolve/main/Qwen_Qwen3-4B-Q4_K_M.gguf",
            fileName = "Qwen_Qwen3-4B-Q4_K_M.gguf",
            sizeMb = 2900
        )
    )

    fun findById(id: String): AvailableModel? = ALL.find { it.id == id }

    fun getDefault(): AvailableModel = findById(DEFAULT_MODEL_ID)!!
}
