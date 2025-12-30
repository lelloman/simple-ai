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

    // ==================== Classification (NLU) ====================

    /**
     * Apply a classification adapter provided by the caller app.
     *
     * The caller provides adapter files via ParcelFileDescriptors.
     * SimpleAI keeps the base model in memory and patches it with the adapter.
     * When switching adapters, the previous patch is reverted before applying the new one.
     *
     * @param adapterId Unique identifier for this adapter (e.g., "simpleephem")
     * @param adapterVersion Version string for cache invalidation (e.g., "1.0")
     * @param patchFd ParcelFileDescriptor for the .lorapatch file
     * @param headsFd ParcelFileDescriptor for the heads.bin file
     * @param tokenizerFd ParcelFileDescriptor for the tokenizer.json file
     * @param configFd ParcelFileDescriptor for the config.json file
     *
     * @return "ok" if adapter applied successfully, or "error: <message>"
     */
    String applyClassificationAdapter(
        String adapterId,
        String adapterVersion,
        in ParcelFileDescriptor patchFd,
        in ParcelFileDescriptor headsFd,
        in ParcelFileDescriptor tokenizerFd,
        in ParcelFileDescriptor configFd
    );

    /**
     * Remove the current classification adapter, reverting to pristine model state.
     *
     * @return "ok" if adapter removed successfully, or "error: <message>"
     */
    String removeClassificationAdapter();

    /**
     * Classify text using the currently applied adapter.
     *
     * @param text The input text to classify
     * @param adapterId Must match the currently applied adapter
     *
     * @return JSON result:
     *   {
     *     "intent": "add_subject",
     *     "intent_confidence": 0.95,
     *     "slots": {
     *       "planet": ["mars", "venus"],
     *       "date": "tomorrow"
     *     },
     *     "raw_slot_labels": ["O", "B-planet", "O", ...]  // BIO tags per token
     *   }
     *   Or error: {"error": "No adapter applied"} or {"error": "Adapter mismatch: ..."}
     */
    String classify(String text, String adapterId);

    /**
     * Get info about the currently applied classification adapter.
     *
     * @return JSON object with adapter info, or null if no adapter applied:
     *   {"id": "simpleephem", "version": "1.0", "intents": [...], "slot_types": [...]}
     */
    String getCurrentClassificationAdapter();

    /**
     * Check if classification is ready (base model loaded).
     * Note: An adapter must be applied before calling classify().
     */
    boolean isClassificationReady();
}
