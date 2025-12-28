package com.lelloman.simpleai.chat

import kotlinx.serialization.Serializable
import kotlinx.serialization.json.JsonObject

/**
 * A message in a chat conversation.
 */
@Serializable
data class ChatMessage(
    val role: String,  // "user", "assistant", "tool", "system"
    val content: String,
    val toolCallId: String? = null,      // For tool response messages
    val toolCalls: List<ToolCall>? = null // For assistant messages requesting tools
)

/**
 * A tool call requested by the model.
 */
@Serializable
data class ToolCall(
    val id: String,
    val name: String,
    val arguments: JsonObject
)

/**
 * A tool definition in OpenAI format.
 */
@Serializable
data class ToolDefinition(
    val type: String = "function",
    val function: FunctionDefinition
)

@Serializable
data class FunctionDefinition(
    val name: String,
    val description: String,
    val parameters: JsonObject
)

/**
 * Response from chat generation.
 */
sealed class ChatResponse {
    /** Plain text response */
    data class Text(val content: String) : ChatResponse()

    /** Model wants to call tools */
    data class ToolCalls(val toolCalls: List<ToolCall>) : ChatResponse()

    /** Both text and tool calls */
    data class Mixed(val content: String, val toolCalls: List<ToolCall>) : ChatResponse()

    /** An error occurred */
    data class Error(val message: String) : ChatResponse()
}

/**
 * The prompt format a model uses for tool calling.
 */
enum class ModelFormat {
    /** Hermes format with <tool_call> XML tags - used by Qwen, Hermes, Functionary */
    HERMES,

    /** Llama 3.1+ native format */
    LLAMA,

    /** Mistral format */
    MISTRAL,

    /** No tool support - raw text generation only */
    RAW
}
