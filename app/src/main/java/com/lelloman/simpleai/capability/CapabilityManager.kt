package com.lelloman.simpleai.capability

import android.content.Context
import android.content.SharedPreferences
import android.util.Log
import com.lelloman.simpleai.model.LocalAIModel
import com.lelloman.simpleai.translation.TranslationManager
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json

/**
 * Manages all SimpleAI capabilities, their download status, and persistence.
 */
class CapabilityManager(
    private val context: Context
) {
    companion object {
        private const val TAG = "CapabilityManager"
        private const val PREFS_NAME = "simple_ai_capabilities"
        private const val KEY_TRANSLATION_LANGUAGES = "translation_languages"

        // Model sizes
        const val VOICE_COMMANDS_MODEL_SIZE = 120_000_000L  // ~120 MB XLM-RoBERTa
        val LOCAL_AI_MODEL_SIZE = LocalAIModel.SIZE_BYTES   // Qwen 3 1.7B (~1.3 GB)
        const val TRANSLATION_LANGUAGE_SIZE = 30_000_000L   // ~30 MB per language

        // ML Kit supported languages - delegate to TranslationManager as single source of truth
        val SUPPORTED_LANGUAGES: Set<String>
            get() = TranslationManager.SUPPORTED_LANGUAGES
    }

    private val prefs: SharedPreferences by lazy {
        context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
    }

    private val json = Json { ignoreUnknownKeys = true }

    // Voice Commands capability
    private val _voiceCommandsStatus = MutableStateFlow<CapabilityStatus>(
        CapabilityStatus.NotDownloaded(VOICE_COMMANDS_MODEL_SIZE)
    )
    val voiceCommandsStatus: StateFlow<CapabilityStatus> = _voiceCommandsStatus.asStateFlow()

    // Translation capability
    private val _translationStatus = MutableStateFlow<CapabilityStatus>(
        CapabilityStatus.NotDownloaded(0)
    )
    val translationStatus: StateFlow<CapabilityStatus> = _translationStatus.asStateFlow()

    private val _downloadedLanguages = MutableStateFlow<Set<String>>(emptySet())
    val downloadedLanguages: StateFlow<Set<String>> = _downloadedLanguages.asStateFlow()

    // Cloud AI capability (always ready, no download needed)
    private val _cloudAiStatus = MutableStateFlow<CapabilityStatus>(CapabilityStatus.Ready)
    val cloudAiStatus: StateFlow<CapabilityStatus> = _cloudAiStatus.asStateFlow()

    // Local AI capability
    private val _localAiStatus = MutableStateFlow<CapabilityStatus>(
        CapabilityStatus.NotDownloaded(LOCAL_AI_MODEL_SIZE)
    )
    val localAiStatus: StateFlow<CapabilityStatus> = _localAiStatus.asStateFlow()

    init {
        loadPersistedState()
    }

    private fun loadPersistedState() {
        // Load downloaded translation languages
        val languagesJson = prefs.getString(KEY_TRANSLATION_LANGUAGES, null)
        if (languagesJson != null) {
            try {
                val languages = json.decodeFromString<Set<String>>(languagesJson)
                _downloadedLanguages.value = languages
                if (languages.isNotEmpty()) {
                    _translationStatus.value = CapabilityStatus.Ready
                }
            } catch (e: Exception) {
                Log.e(TAG, "Failed to load translation languages", e)
            }
        }

        // Model download status is detected when engines initialize in SimpleAIService
    }

    // =========================================================================
    // Voice Commands
    // =========================================================================

    fun updateVoiceCommandsStatus(status: CapabilityStatus) {
        _voiceCommandsStatus.value = status
    }

    // =========================================================================
    // Translation
    // =========================================================================

    fun addTranslationLanguage(languageCode: String) {
        val updated = _downloadedLanguages.value + languageCode
        // Always include English as pivot
        val withEnglish = if ("en" !in updated && updated.isNotEmpty()) {
            updated + "en"
        } else {
            updated
        }
        _downloadedLanguages.value = withEnglish
        persistTranslationLanguages(withEnglish)

        if (withEnglish.isNotEmpty()) {
            _translationStatus.value = CapabilityStatus.Ready
        }
    }

    fun removeTranslationLanguage(languageCode: String) {
        if (languageCode == "en") {
            // Can't remove English directly, it's removed when all others are removed
            return
        }

        val updated = _downloadedLanguages.value - languageCode
        // Remove English if no other languages remain
        val withoutEnglish = if (updated == setOf("en")) {
            emptySet()
        } else {
            updated
        }
        _downloadedLanguages.value = withoutEnglish
        persistTranslationLanguages(withoutEnglish)

        if (withoutEnglish.isEmpty()) {
            _translationStatus.value = CapabilityStatus.NotDownloaded(0)
        }
    }

    fun updateTranslationStatus(status: CapabilityStatus) {
        _translationStatus.value = status
    }

    /**
     * Sync downloaded languages from TranslationManager.
     * This replaces the current set entirely to ensure consistency.
     */
    fun syncTranslationLanguages(languages: Set<String>) {
        _downloadedLanguages.value = languages
        persistTranslationLanguages(languages)
    }

    private fun persistTranslationLanguages(languages: Set<String>) {
        prefs.edit()
            .putString(KEY_TRANSLATION_LANGUAGES, json.encodeToString(languages))
            .apply()
    }

    // =========================================================================
    // Local AI
    // =========================================================================

    fun updateLocalAiStatus(status: CapabilityStatus) {
        _localAiStatus.value = status
    }

    // =========================================================================
    // Capability Queries
    // =========================================================================

    fun getCapability(id: CapabilityId): Capability {
        return when (id) {
            CapabilityId.VOICE_COMMANDS -> Capability(
                id = CapabilityId.VOICE_COMMANDS,
                name = "Voice Commands",
                description = "Intent classification and entity extraction for voice control",
                status = _voiceCommandsStatus.value
            )
            CapabilityId.TRANSLATION -> Capability(
                id = CapabilityId.TRANSLATION,
                name = "Translation",
                description = "On-device translation between languages",
                status = _translationStatus.value
            )
            CapabilityId.CLOUD_AI -> Capability(
                id = CapabilityId.CLOUD_AI,
                name = "Cloud AI",
                description = "Cloud-based LLM for advanced conversations",
                status = _cloudAiStatus.value
            )
            CapabilityId.LOCAL_AI -> Capability(
                id = CapabilityId.LOCAL_AI,
                name = "Local AI",
                description = "On-device LLM for offline conversations",
                status = _localAiStatus.value
            )
        }
    }

    fun getAllCapabilities(): List<Capability> {
        return CapabilityId.entries.map { getCapability(it) }
    }

    fun getTranslationCapability(): TranslationCapability {
        return TranslationCapability(
            baseCapability = getCapability(CapabilityId.TRANSLATION),
            downloadedLanguages = _downloadedLanguages.value,
            availableLanguages = SUPPORTED_LANGUAGES
        )
    }
}
