package com.lelloman.simpleai.chat

/**
 * Fallback formatter for models without tool calling support.
 *
 * Simply concatenates messages into a plain text prompt.
 * Tool definitions are ignored.
 */
class RawFormatter : ChatFormatter {

    override val format = ModelFormat.RAW
    override val supportsTools = false

    override fun formatPrompt(
        messages: List<ChatMessage>,
        tools: List<ToolDefinition>,
        systemPrompt: String?
    ): String {
        val sb = StringBuilder()

        // System prompt
        if (systemPrompt != null) {
            sb.appendLine("System: $systemPrompt")
            sb.appendLine()
        }

        // Messages
        for (message in messages) {
            when (message.role) {
                "user" -> sb.appendLine("User: ${message.content}")
                "assistant" -> sb.appendLine("Assistant: ${message.content}")
                "tool" -> sb.appendLine("Tool result: ${message.content}")
                "system" -> sb.appendLine("System: ${message.content}")
            }
        }

        sb.appendLine("Assistant:")

        return sb.toString()
    }

    override fun parseResponse(response: String): ChatResponse {
        // No tool parsing - just return as text
        return ChatResponse.Text(response.trim())
    }
}
