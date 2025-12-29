package com.lelloman.simpleai.model

import com.lelloman.simpleai.chat.ModelFormat
import kotlinx.serialization.Serializable

enum class EngineType {
    LLAMA_CPP,   // Uses llamacpp-kotlin with GGUF files
    EXECUTORCH   // Uses ExecuTorch with PTE files
}

@Serializable
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
    val tokenizerUrl: String? = null  // For ExecuTorch models that need separate tokenizer
)

object AvailableModels {
    val DEFAULT_MODEL_ID = "llama3.2-3b"  // GGUF - ExecuTorch models only have 1024 seq_len

    val ALL = listOf(
        // ==================== Llama 3.2 GGUF (llama.cpp) ====================
        AvailableModel(
            id = "llama3.2-1b",
            name = "Llama 3.2 1B",
            description = "Smallest Llama. Fast, good tool support.",
            url = "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf",
            fileName = "Llama-3.2-1B-Instruct-Q4_K_M.gguf",
            sizeMb = 776,
            format = ModelFormat.LLAMA,
            supportsTools = true
        ),
        AvailableModel(
            id = "llama3.2-3b",
            name = "Llama 3.2 3B",
            description = "Best balance of size and quality. Recommended.",
            url = "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf",
            fileName = "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
            sizeMb = 2020,
            format = ModelFormat.LLAMA,
            supportsTools = true
        ),

        // ==================== Qwen 3 (Hermes format) ====================
        AvailableModel(
            id = "qwen3-1.7b",
            name = "Qwen 3 1.7B",
            description = "Good general model with tool support.",
            url = "https://huggingface.co/bartowski/Qwen_Qwen3-1.7B-GGUF/resolve/main/Qwen_Qwen3-1.7B-Q4_K_M.gguf",
            fileName = "Qwen_Qwen3-1.7B-Q4_K_M.gguf",
            sizeMb = 1280,
            format = ModelFormat.HERMES,
            supportsTools = true
        ),
        AvailableModel(
            id = "qwen3-4b",
            name = "Qwen 3 4B",
            description = "Higher quality Qwen. Needs more RAM.",
            url = "https://huggingface.co/bartowski/Qwen_Qwen3-4B-GGUF/resolve/main/Qwen_Qwen3-4B-Q4_K_M.gguf",
            fileName = "Qwen_Qwen3-4B-Q4_K_M.gguf",
            sizeMb = 2900,
            format = ModelFormat.HERMES,
            supportsTools = true
        ),

        // ==================== SmolLM2 (Fast, limited tool support) ====================
        AvailableModel(
            id = "smollm2-360m",
            name = "SmolLM2 360M",
            description = "Ultra fast. Limited capabilities.",
            url = "https://huggingface.co/bartowski/SmolLM2-360M-Instruct-GGUF/resolve/main/SmolLM2-360M-Instruct-Q4_K_M.gguf",
            fileName = "SmolLM2-360M-Instruct-Q4_K_M.gguf",
            sizeMb = 271,
            format = ModelFormat.RAW,  // SmolLM2 doesn't have reliable tool calling
            supportsTools = false
        ),
        AvailableModel(
            id = "smollm2-1.7b",
            name = "SmolLM2 1.7B",
            description = "Fast general assistant. No tool support.",
            url = "https://huggingface.co/bartowski/SmolLM2-1.7B-Instruct-GGUF/resolve/main/SmolLM2-1.7B-Instruct-Q4_K_M.gguf",
            fileName = "SmolLM2-1.7B-Instruct-Q4_K_M.gguf",
            sizeMb = 1060,
            format = ModelFormat.RAW,
            supportsTools = false
        ),

        // ==================== Functionary (Best for tools) ====================
        AvailableModel(
            id = "functionary-3b",
            name = "Functionary 3B",
            description = "Specialized for function calling. Best tool accuracy.",
            url = "https://huggingface.co/meetkai/functionary-small-v3.2-GGUF/resolve/main/functionary-small-v3.2.Q4_K_M.gguf",
            fileName = "functionary-small-v3.2.Q4_K_M.gguf",
            sizeMb = 4650,
            format = ModelFormat.HERMES,
            supportsTools = true
        ),

        // ==================== Hermes 3 (Good all-around) ====================
        AvailableModel(
            id = "hermes3-3b",
            name = "Hermes 3 Llama 3B",
            description = "Excellent instruction following and tools.",
            url = "https://huggingface.co/NousResearch/Hermes-3-Llama-3.2-3B-GGUF/resolve/main/Hermes-3-Llama-3.2-3B.Q4_K_M.gguf",
            fileName = "Hermes-3-Llama-3.2-3B.Q4_K_M.gguf",
            sizeMb = 2020,
            format = ModelFormat.HERMES,
            supportsTools = true
        ),

        // ==================== ExecuTorch (Fast, but limited context) ====================
        AvailableModel(
            id = "llama3.2-3b-et",
            name = "Llama 3.2 3B (ExecuTorch)",
            description = "Fast inference, but only 1024 token limit.",
            url = "https://huggingface.co/software-mansion/react-native-executorch-llama-3.2/resolve/main/llama-3.2-3B/spinquant/llama3_2_3B_spinquant.pte",
            fileName = "llama3_2_3B_spinquant.pte",
            sizeMb = 2550,
            format = ModelFormat.LLAMA,
            supportsTools = true,
            engineType = EngineType.EXECUTORCH,
            tokenizerUrl = "https://huggingface.co/executorch-community/Llama-3.2-1B-Instruct-QLORA_INT4_EO8-ET/resolve/main/tokenizer.model"
        )
    )

    fun findById(id: String): AvailableModel? = ALL.find { it.id == id }

    fun getDefault(): AvailableModel = findById(DEFAULT_MODEL_ID)!!

    /** Get models that support tool calling */
    fun getToolCapableModels(): List<AvailableModel> = ALL.filter { it.supportsTools }
}
