package com.lelloman.simpleai.translation

import android.content.Context
import android.util.Log
import com.google.mlkit.common.model.DownloadConditions
import com.google.mlkit.common.model.RemoteModelManager
import com.google.mlkit.nl.languageid.LanguageIdentification
import com.google.mlkit.nl.translate.TranslateLanguage
import com.google.mlkit.nl.translate.TranslateRemoteModel
import com.google.mlkit.nl.translate.Translation
import com.google.mlkit.nl.translate.Translator
import com.google.mlkit.nl.translate.TranslatorOptions
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlinx.coroutines.withContext
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException

/**
 * Manages ML Kit translation models and performs translations.
 *
 * All translations go through English as a pivot language, so:
 * - User must download English + their target language(s)
 * - IT -> ES becomes IT -> EN -> ES
 */
class TranslationManager(
    private val context: Context
) {
    companion object {
        private const val TAG = "TranslationManager"

        // Map our language codes to ML Kit codes
        private val LANGUAGE_MAP = mapOf(
            "af" to TranslateLanguage.AFRIKAANS,
            "ar" to TranslateLanguage.ARABIC,
            "be" to TranslateLanguage.BELARUSIAN,
            "bg" to TranslateLanguage.BULGARIAN,
            "bn" to TranslateLanguage.BENGALI,
            "ca" to TranslateLanguage.CATALAN,
            "cs" to TranslateLanguage.CZECH,
            "cy" to TranslateLanguage.WELSH,
            "da" to TranslateLanguage.DANISH,
            "de" to TranslateLanguage.GERMAN,
            "el" to TranslateLanguage.GREEK,
            "en" to TranslateLanguage.ENGLISH,
            "eo" to TranslateLanguage.ESPERANTO,
            "es" to TranslateLanguage.SPANISH,
            "et" to TranslateLanguage.ESTONIAN,
            "fa" to TranslateLanguage.PERSIAN,
            "fi" to TranslateLanguage.FINNISH,
            "fr" to TranslateLanguage.FRENCH,
            "ga" to TranslateLanguage.IRISH,
            "gl" to TranslateLanguage.GALICIAN,
            "gu" to TranslateLanguage.GUJARATI,
            "he" to TranslateLanguage.HEBREW,
            "hi" to TranslateLanguage.HINDI,
            "hr" to TranslateLanguage.CROATIAN,
            "ht" to TranslateLanguage.HAITIAN_CREOLE,
            "hu" to TranslateLanguage.HUNGARIAN,
            "id" to TranslateLanguage.INDONESIAN,
            "is" to TranslateLanguage.ICELANDIC,
            "it" to TranslateLanguage.ITALIAN,
            "ja" to TranslateLanguage.JAPANESE,
            "ka" to TranslateLanguage.GEORGIAN,
            "kn" to TranslateLanguage.KANNADA,
            "ko" to TranslateLanguage.KOREAN,
            "lt" to TranslateLanguage.LITHUANIAN,
            "lv" to TranslateLanguage.LATVIAN,
            "mk" to TranslateLanguage.MACEDONIAN,
            "mr" to TranslateLanguage.MARATHI,
            "ms" to TranslateLanguage.MALAY,
            "mt" to TranslateLanguage.MALTESE,
            "nl" to TranslateLanguage.DUTCH,
            "no" to TranslateLanguage.NORWEGIAN,
            "pl" to TranslateLanguage.POLISH,
            "pt" to TranslateLanguage.PORTUGUESE,
            "ro" to TranslateLanguage.ROMANIAN,
            "ru" to TranslateLanguage.RUSSIAN,
            "sk" to TranslateLanguage.SLOVAK,
            "sl" to TranslateLanguage.SLOVENIAN,
            "sq" to TranslateLanguage.ALBANIAN,
            "sv" to TranslateLanguage.SWEDISH,
            "sw" to TranslateLanguage.SWAHILI,
            "ta" to TranslateLanguage.TAMIL,
            "te" to TranslateLanguage.TELUGU,
            "th" to TranslateLanguage.THAI,
            "tl" to TranslateLanguage.TAGALOG,
            "tr" to TranslateLanguage.TURKISH,
            "uk" to TranslateLanguage.UKRAINIAN,
            "ur" to TranslateLanguage.URDU,
            "vi" to TranslateLanguage.VIETNAMESE,
            "zh" to TranslateLanguage.CHINESE
        )
    }

    private val modelManager = RemoteModelManager.getInstance()
    private val languageIdentifier = LanguageIdentification.getClient()

    // Cache translators to avoid recreating them
    private val translatorCache = mutableMapOf<Pair<String, String>, Translator>()

    private val _downloadedLanguages = MutableStateFlow<Set<String>>(emptySet())
    val downloadedLanguages: StateFlow<Set<String>> = _downloadedLanguages.asStateFlow()

    /**
     * Initialize by checking which languages are already downloaded.
     */
    suspend fun initialize() = withContext(Dispatchers.IO) {
        try {
            val downloaded = getDownloadedLanguages()
            _downloadedLanguages.value = downloaded
            Log.i(TAG, "Initialized with downloaded languages: $downloaded")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize", e)
        }
    }

    /**
     * Check which ML Kit translation models are downloaded.
     */
    suspend fun getDownloadedLanguages(): Set<String> = suspendCancellableCoroutine { cont ->
        modelManager.getDownloadedModels(TranslateRemoteModel::class.java)
            .addOnSuccessListener { models ->
                val languages = models.mapNotNull { model ->
                    LANGUAGE_MAP.entries.find { it.value == model.language }?.key
                }.toSet()
                cont.resume(languages)
            }
            .addOnFailureListener { e ->
                Log.e(TAG, "Failed to get downloaded models", e)
                cont.resume(emptySet())
            }
    }

    /**
     * Download a language model.
     *
     * @param languageCode Our language code (e.g., "it", "es")
     * @param onProgress Progress callback (0.0 to 1.0) - ML Kit doesn't provide granular progress
     */
    suspend fun downloadLanguage(
        languageCode: String,
        onProgress: (Float) -> Unit = {}
    ): Result<Unit> = withContext(Dispatchers.IO) {
        val mlKitCode = LANGUAGE_MAP[languageCode]
            ?: return@withContext Result.failure(IllegalArgumentException("Unknown language: $languageCode"))

        try {
            Log.i(TAG, "Downloading language: $languageCode")
            onProgress(0.1f) // Show some initial progress

            val model = TranslateRemoteModel.Builder(mlKitCode).build()
            val conditions = DownloadConditions.Builder()
                .requireWifi()
                .build()

            suspendCancellableCoroutine<Unit> { cont ->
                modelManager.download(model, conditions)
                    .addOnSuccessListener {
                        Log.i(TAG, "Downloaded language: $languageCode")
                        onProgress(1.0f)
                        cont.resume(Unit)
                    }
                    .addOnFailureListener { e ->
                        Log.e(TAG, "Failed to download language: $languageCode", e)
                        cont.resumeWithException(e)
                    }
            }

            // Update downloaded languages
            _downloadedLanguages.value = getDownloadedLanguages()
            Result.success(Unit)
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    /**
     * Delete a downloaded language model.
     */
    suspend fun deleteLanguage(languageCode: String): Result<Unit> = withContext(Dispatchers.IO) {
        val mlKitCode = LANGUAGE_MAP[languageCode]
            ?: return@withContext Result.failure(IllegalArgumentException("Unknown language: $languageCode"))

        try {
            val model = TranslateRemoteModel.Builder(mlKitCode).build()

            suspendCancellableCoroutine<Unit> { cont ->
                modelManager.deleteDownloadedModel(model)
                    .addOnSuccessListener {
                        Log.i(TAG, "Deleted language: $languageCode")
                        cont.resume(Unit)
                    }
                    .addOnFailureListener { e ->
                        Log.e(TAG, "Failed to delete language: $languageCode", e)
                        cont.resumeWithException(e)
                    }
            }

            // Clear cached translators for this language
            translatorCache.entries.removeIf { (pair, translator) ->
                if (pair.first == languageCode || pair.second == languageCode) {
                    translator.close()
                    true
                } else {
                    false
                }
            }

            // Update downloaded languages
            _downloadedLanguages.value = getDownloadedLanguages()
            Result.success(Unit)
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    /**
     * Translate text from source to target language.
     *
     * @param text Text to translate
     * @param sourceLang Source language code (or "auto" for detection)
     * @param targetLang Target language code
     * @return TranslationResult with translated text and detected source language
     */
    suspend fun translate(
        text: String,
        sourceLang: String,
        targetLang: String
    ): Result<TranslationResult> = withContext(Dispatchers.IO) {
        try {
            // Detect language if "auto"
            val actualSourceLang = if (sourceLang == "auto") {
                detectLanguage(text) ?: return@withContext Result.failure(
                    IllegalStateException("Could not detect source language")
                )
            } else {
                sourceLang
            }

            // Same language, no translation needed
            if (actualSourceLang == targetLang) {
                return@withContext Result.success(
                    TranslationResult(text, actualSourceLang)
                )
            }

            // Check if languages are downloaded
            val downloaded = _downloadedLanguages.value
            if (actualSourceLang !in downloaded) {
                return@withContext Result.failure(
                    IllegalStateException("Source language '$actualSourceLang' not downloaded")
                )
            }
            if (targetLang !in downloaded) {
                return@withContext Result.failure(
                    IllegalStateException("Target language '$targetLang' not downloaded")
                )
            }

            // Translate
            val translator = getOrCreateTranslator(actualSourceLang, targetLang)
            val translatedText = suspendCancellableCoroutine<String> { cont ->
                translator.translate(text)
                    .addOnSuccessListener { translated ->
                        cont.resume(translated)
                    }
                    .addOnFailureListener { e ->
                        cont.resumeWithException(e)
                    }
            }

            Result.success(TranslationResult(translatedText, actualSourceLang))
        } catch (e: Exception) {
            Log.e(TAG, "Translation failed", e)
            Result.failure(e)
        }
    }

    /**
     * Detect the language of the given text.
     */
    suspend fun detectLanguage(text: String): String? = suspendCancellableCoroutine { cont ->
        languageIdentifier.identifyLanguage(text)
            .addOnSuccessListener { langCode: String ->
                if (langCode == "und") {
                    cont.resume(null)
                } else {
                    // ML Kit returns BCP-47 codes, map to our codes
                    val ourCode = LANGUAGE_MAP.entries.find { it.value == langCode }?.key ?: langCode
                    cont.resume(ourCode)
                }
            }
            .addOnFailureListener { _: Exception ->
                cont.resume(null)
            }
    }

    private fun getOrCreateTranslator(sourceLang: String, targetLang: String): Translator {
        val key = sourceLang to targetLang
        return translatorCache.getOrPut(key) {
            val sourceCode = LANGUAGE_MAP[sourceLang]
                ?: throw IllegalArgumentException("Unknown source language: $sourceLang")
            val targetCode = LANGUAGE_MAP[targetLang]
                ?: throw IllegalArgumentException("Unknown target language: $targetLang")

            val options = TranslatorOptions.Builder()
                .setSourceLanguage(sourceCode)
                .setTargetLanguage(targetCode)
                .build()

            Translation.getClient(options)
        }
    }

    /**
     * Check if a language is supported by ML Kit.
     */
    fun isLanguageSupported(languageCode: String): Boolean {
        return languageCode in LANGUAGE_MAP
    }

    /**
     * Get all supported language codes.
     */
    fun getSupportedLanguages(): Set<String> = LANGUAGE_MAP.keys

    /**
     * Release resources.
     */
    fun release() {
        translatorCache.values.forEach { it.close() }
        translatorCache.clear()
        languageIdentifier.close()
    }
}

/**
 * Result of a translation.
 */
data class TranslationResult(
    val translatedText: String,
    val detectedSourceLang: String
)
