package com.lelloman.simpleai;

interface ILLMService {
    /**
     * Check if the model is loaded and ready to generate responses.
     */
    boolean isReady();

    /**
     * Get the current status of the service.
     * Returns: "downloading", "loading", "ready", or "error: <message>"
     */
    String getStatus();

    // ==================== Model Discovery ====================

    /**
     * Get available models that can be downloaded/used.
     * Returns JSON array:
     * [{"id": "qwen3-1.7b", "name": "Qwen 3 1.7B", "sizeMb": 1280,
     *   "supportsTools": true, "downloaded": true}, ...]
     */
    String getAvailableModels();

    /**
     * Get currently loaded model info.
     * Returns JSON:
     * {"id": "qwen3-1.7b", "name": "Qwen 3 1.7B", "supportsTools": true, ...}
     * Or null if no model is loaded.
     */
    String getCurrentModel();

    /**
     * Request to switch to a different model.
     * This may trigger a download if the model is not cached.
     * The switch happens asynchronously - monitor getStatus() for progress.
     *
     * @param modelId The model ID from getAvailableModels()
     * @return "ok" if switch initiated, or "error: <message>"
     */
    String setModel(String modelId);

    // ==================== Generation ====================

    /**
     * Generate a response for the given prompt.
     * This is a blocking call that returns when generation is complete.
     *
     * @param prompt The input text to generate a response for
     * @return The generated response text
     */
    String generate(String prompt);

    /**
     * Generate a response with custom parameters.
     *
     * @param prompt The input text
     * @param maxTokens Maximum tokens to generate (0 = use default)
     * @param temperature Sampling temperature (0.0-2.0, default 0.7)
     * @return The generated response text
     */
    String generateWithParams(String prompt, int maxTokens, float temperature);

    // ==================== Chat with Tools ====================

    /**
     * Chat with tool/function calling support.
     * SimpleAI automatically uses the correct prompt format for the current model.
     *
     * @param messagesJson JSON array of messages in standard format:
     *   [{"role": "user"|"assistant"|"tool", "content": "...",
     *     "toolCallId": "..." (for tool results),
     *     "toolCalls": [...] (for assistant tool requests)}]
     *
     * @param toolsJson JSON array of tool definitions (OpenAI format):
     *   [{"type": "function", "function": {"name": "...", "description": "...",
     *     "parameters": {"type": "object", "properties": {...}, "required": [...]}}}]
     *   Pass null or empty string if no tools.
     *
     * @param systemPrompt Optional system prompt. Pass null for model default.
     *
     * @return JSON response in standard format:
     *   {"type": "text", "content": "..."} - plain text response
     *   {"type": "tool_calls", "toolCalls": [{"id": "...", "name": "...", "arguments": {...}}]}
     *   {"type": "mixed", "content": "...", "toolCalls": [...]} - text + tool calls
     *   {"type": "error", "message": "..."} - error occurred
     */
    String chat(String messagesJson, String toolsJson, String systemPrompt);

    // ==================== Translation ====================

    /**
     * Translate text from one language to another.
     *
     * @param text The text to translate
     * @param sourceLanguage Source language code (e.g., "en", "es", "auto" for auto-detect)
     * @param targetLanguage Target language code (e.g., "en", "es")
     * @return The translated text
     */
    String translate(String text, String sourceLanguage, String targetLanguage);

    /**
     * Get list of supported language codes for translation.
     * Returns JSON array of language objects with "code" and "name" fields.
     */
    String getSupportedLanguages();
}
