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
    val DEFAULT_MODEL_ID = "smollm2-1.7b"

    val ALL = listOf(
        AvailableModel(
            id = "smollm2-360m",
            name = "SmolLM2 360M",
            description = "Smallest, fastest. Good for simple tasks.",
            url = "https://huggingface.co/bartowski/SmolLM2-360M-Instruct-GGUF/resolve/main/SmolLM2-360M-Instruct-Q4_K_M.gguf",
            fileName = "SmolLM2-360M-Instruct-Q4_K_M.gguf",
            sizeMb = 271
        ),
        AvailableModel(
            id = "smollm2-1.7b",
            name = "SmolLM2 1.7B",
            description = "Balanced size and quality. Recommended.",
            url = "https://huggingface.co/bartowski/SmolLM2-1.7B-Instruct-GGUF/resolve/main/SmolLM2-1.7B-Instruct-Q4_K_M.gguf",
            fileName = "SmolLM2-1.7B-Instruct-Q4_K_M.gguf",
            sizeMb = 1060
        )
    )

    fun findById(id: String): AvailableModel? = ALL.find { it.id == id }

    fun getDefault(): AvailableModel = findById(DEFAULT_MODEL_ID)!!
}
