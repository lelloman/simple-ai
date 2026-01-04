package com.lelloman.simpleai;

interface ISimpleAI {

    // =========================================================================
    // SERVICE INFO
    // =========================================================================

    /**
     * Get service version and capabilities status.
     *
     * @param protocolVersion Client's protocol version (e.g., 1, 2, 3)
     * @return JSON response:
     *   {
     *     "status": "success",
     *     "protocolVersion": 1,
     *     "data": {
     *       "serviceVersion": 5,
     *       "minProtocol": 1,
     *       "maxProtocol": 3,
     *       "capabilities": {
     *         "voiceCommands": {"status": "ready", "modelSize": 120000000},
     *         "translation": {"status": "downloading", "progress": 0.45, "languages": ["en", "it"]},
     *         "cloudAi": {"status": "ready"},
     *         "localAi": {"status": "not_downloaded", "modelSize": 1300000000}
     *       }
     *     }
     *   }
     */
    String getServiceInfo(int protocolVersion);

    // =========================================================================
    // VOICE COMMANDS (NLU)
    // =========================================================================

    /**
     * Classify text using NLU with a specific adapter.
     *
     * Adapter files are passed with every request. SimpleAI tracks the currently
     * applied adapter by (adapterId, adapterVersion). If it matches, inference runs
     * directly. If different, SimpleAI reverts the current patch (if any), applies
     * the new one, and stores the revert data for future switches.
     *
     * NOTE: The caller is responsible for closing the ParcelFileDescriptors after
     * this call returns. SimpleAI reads from them during this call but does not
     * take ownership.
     *
     * @param protocolVersion Client's protocol version (e.g., 1, 2, 3)
     * @param text Text to classify
     * @param adapterId Unique identifier for the adapter (e.g., "simpleephem")
     * @param adapterVersion Version string for change detection (e.g., "1.0.3")
     * @param patchFd ParcelFileDescriptor for .lorapatch file (optional, can be null
     *                for adapters that only use classification heads without LoRA)
     * @param headsFd ParcelFileDescriptor for heads.bin file (caller must close)
     * @param tokenizerFd ParcelFileDescriptor for tokenizer.json (caller must close)
     * @param configFd ParcelFileDescriptor for config.json (caller must close)
     * @return JSON response:
     *   Success:
     *   {
     *     "status": "success",
     *     "protocolVersion": 1,
     *     "data": {
     *       "intent": "add_subject",
     *       "intentConfidence": 0.94,
     *       "slots": {
     *         "person": ["John"],
     *         "date": ["tomorrow"]
     *       }
     *     }
     *   }
     */
    String classify(
        int protocolVersion,
        String text,
        String adapterId,
        String adapterVersion,
        in ParcelFileDescriptor patchFd,
        in ParcelFileDescriptor headsFd,
        in ParcelFileDescriptor tokenizerFd,
        in ParcelFileDescriptor configFd
    );

    /**
     * Remove currently applied adapter and restore pristine model.
     * Useful when client app is done and wants to free memory.
     *
     * @param protocolVersion Client's protocol version
     * @return JSON response with success or error
     */
    String clearAdapter(int protocolVersion);

    // =========================================================================
    // TRANSLATION
    // =========================================================================

    /**
     * Translate text between languages.
     *
     * @param protocolVersion Client's protocol version
     * @param text Text to translate
     * @param sourceLang Source language code (e.g., "it") or "auto" for detection
     * @param targetLang Target language code (e.g., "en")
     * @return JSON response:
     *   {
     *     "status": "success",
     *     "protocolVersion": 1,
     *     "data": {
     *       "translatedText": "Hello world",
     *       "detectedSourceLang": "it"
     *     }
     *   }
     */
    String translate(
        int protocolVersion,
        String text,
        String sourceLang,
        String targetLang
    );

    /**
     * Get list of downloaded translation languages.
     *
     * @param protocolVersion Client's protocol version
     * @return JSON response with languages array
     */
    String getTranslationLanguages(int protocolVersion);

    // =========================================================================
    // CLOUD AI
    // =========================================================================

    /**
     * Send chat request to cloud LLM endpoint.
     *
     * @param protocolVersion Client's protocol version
     * @param messagesJson JSON array of chat messages
     * @param toolsJson JSON array of tool definitions (optional, can be null)
     * @param systemPrompt System prompt (optional, can be null)
     * @param authToken Client's auth token for cloud service
     * @return JSON response with LLM response or tool calls
     */
    String cloudChat(
        int protocolVersion,
        String messagesJson,
        String toolsJson,
        String systemPrompt,
        String authToken
    );

    // =========================================================================
    // LOCAL AI
    // =========================================================================

    /**
     * Generate text using local LLM.
     *
     * @param protocolVersion Client's protocol version
     * @param prompt Text prompt
     * @param maxTokens Maximum tokens to generate
     * @param temperature Sampling temperature (0.0 - 2.0)
     * @return JSON response with generated text
     */
    String localGenerate(
        int protocolVersion,
        String prompt,
        int maxTokens,
        float temperature
    );

    /**
     * Chat using local LLM with tool support.
     *
     * @param protocolVersion Client's protocol version
     * @param messagesJson JSON array of chat messages
     * @param toolsJson JSON array of tool definitions (optional)
     * @param systemPrompt System prompt (optional)
     * @return JSON response with LLM response or tool calls
     */
    String localChat(
        int protocolVersion,
        String messagesJson,
        String toolsJson,
        String systemPrompt
    );
}
