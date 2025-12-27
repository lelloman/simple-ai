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

    // ==================== Translation ====================

    /**
     * Translate text from one language to another.
     *
     * @param text The text to translate
     * @param sourceLanguage Source language code (e.g., "en", "es", "auto" for auto-detect)
     * @param targetLanguage Target language code (e.g., "en", "es")
     * @return The translated text
     *
     * Supported language codes:
     *   en - English
     *   es - Spanish
     *   fr - French
     *   de - German
     *   it - Italian
     *   pt - Portuguese
     *   nl - Dutch
     *   pl - Polish
     *   ru - Russian
     *   uk - Ukrainian
     *   zh - Chinese (Simplified)
     *   ja - Japanese
     *   ko - Korean
     *   ar - Arabic
     *   hi - Hindi
     *   tr - Turkish
     *   vi - Vietnamese
     *   th - Thai
     *   id - Indonesian
     *   auto - Auto-detect source language
     */
    String translate(String text, String sourceLanguage, String targetLanguage);

    /**
     * Get list of supported language codes for translation.
     * Returns JSON array of language objects with "code" and "name" fields.
     * Example: [{"code":"en","name":"English"},{"code":"es","name":"Spanish"}]
     */
    String getSupportedLanguages();

    // ==================== Model Info ====================

    /**
     * Get information about the loaded model.
     * Returns JSON string with model name, size, etc.
     */
    String getModelInfo();
}
