package com.lelloman.simpleai.chat

/**
 * Formats chat messages and tools into model-specific prompt format,
 * and parses model responses back into structured format.
 *
 * Each model family (Qwen/Hermes, Llama, Mistral) has its own implementation.
 */
interface ChatFormatter {

    /**
     * The format this formatter handles.
     */
    val format: ModelFormat

    /**
     * Whether this format supports tool/function calling.
     */
    val supportsTools: Boolean

    /**
     * Format a chat conversation into a prompt string for the model.
     *
     * @param messages The conversation history
     * @param tools Available tool definitions (may be empty)
     * @param systemPrompt Optional system prompt (null = use model default)
     * @return The formatted prompt string to send to the model
     */
    fun formatPrompt(
        messages: List<ChatMessage>,
        tools: List<ToolDefinition>,
        systemPrompt: String?
    ): String

    /**
     * Parse the model's raw response into a structured ChatResponse.
     *
     * @param response The raw text output from the model
     * @return Parsed response (text, tool calls, mixed, or error)
     */
    fun parseResponse(response: String): ChatResponse

    companion object {
        /**
         * Get the appropriate formatter for a model format.
         */
        fun forFormat(format: ModelFormat): ChatFormatter {
            return when (format) {
                ModelFormat.HERMES -> HermesFormatter()
                ModelFormat.LLAMA -> LlamaFormatter()
                ModelFormat.MISTRAL -> HermesFormatter() // Mistral also works with Hermes format
                ModelFormat.RAW -> RawFormatter()
            }
        }
    }
}
