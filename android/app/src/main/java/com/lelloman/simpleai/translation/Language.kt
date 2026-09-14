package com.lelloman.simpleai.translation

import org.json.JSONArray
import org.json.JSONObject

enum class Language(val code: String, val mlKitCode: String) {
    AFRIKAANS("af", com.google.mlkit.nl.translate.TranslateLanguage.AFRIKAANS),
    ARABIC("ar", com.google.mlkit.nl.translate.TranslateLanguage.ARABIC),
    BELARUSIAN("be", com.google.mlkit.nl.translate.TranslateLanguage.BELARUSIAN),
    BULGARIAN("bg", com.google.mlkit.nl.translate.TranslateLanguage.BULGARIAN),
    BENGALI("bn", com.google.mlkit.nl.translate.TranslateLanguage.BENGALI),
    CATALAN("ca", com.google.mlkit.nl.translate.TranslateLanguage.CATALAN),
    CZECH("cs", com.google.mlkit.nl.translate.TranslateLanguage.CZECH),
    WELSH("cy", com.google.mlkit.nl.translate.TranslateLanguage.WELSH),
    DANISH("da", com.google.mlkit.nl.translate.TranslateLanguage.DANISH),
    GERMAN("de", com.google.mlkit.nl.translate.TranslateLanguage.GERMAN),
    GREEK("el", com.google.mlkit.nl.translate.TranslateLanguage.GREEK),
    ENGLISH("en", com.google.mlkit.nl.translate.TranslateLanguage.ENGLISH),
    ESPERANTO("eo", com.google.mlkit.nl.translate.TranslateLanguage.ESPERANTO),
    SPANISH("es", com.google.mlkit.nl.translate.TranslateLanguage.SPANISH),
    ESTONIAN("et", com.google.mlkit.nl.translate.TranslateLanguage.ESTONIAN),
    PERSIAN("fa", com.google.mlkit.nl.translate.TranslateLanguage.PERSIAN),
    FINNISH("fi", com.google.mlkit.nl.translate.TranslateLanguage.FINNISH),
    FRENCH("fr", com.google.mlkit.nl.translate.TranslateLanguage.FRENCH),
    IRISH("ga", com.google.mlkit.nl.translate.TranslateLanguage.IRISH),
    GALICIAN("gl", com.google.mlkit.nl.translate.TranslateLanguage.GALICIAN),
    GUJARATI("gu", com.google.mlkit.nl.translate.TranslateLanguage.GUJARATI),
    HEBREW("he", com.google.mlkit.nl.translate.TranslateLanguage.HEBREW),
    HINDI("hi", com.google.mlkit.nl.translate.TranslateLanguage.HINDI),
    CROATIAN("hr", com.google.mlkit.nl.translate.TranslateLanguage.CROATIAN),
    HAITIAN_CREOLE("ht", com.google.mlkit.nl.translate.TranslateLanguage.HAITIAN_CREOLE),
    HUNGARIAN("hu", com.google.mlkit.nl.translate.TranslateLanguage.HUNGARIAN),
    INDONESIAN("id", com.google.mlkit.nl.translate.TranslateLanguage.INDONESIAN),
    ICELANDIC("is", com.google.mlkit.nl.translate.TranslateLanguage.ICELANDIC),
    ITALIAN("it", com.google.mlkit.nl.translate.TranslateLanguage.ITALIAN),
    JAPANESE("ja", com.google.mlkit.nl.translate.TranslateLanguage.JAPANESE),
    GEORGIAN("ka", com.google.mlkit.nl.translate.TranslateLanguage.GEORGIAN),
    KANNADA("kn", com.google.mlkit.nl.translate.TranslateLanguage.KANNADA),
    KOREAN("ko", com.google.mlkit.nl.translate.TranslateLanguage.KOREAN),
    LITHUANIAN("lt", com.google.mlkit.nl.translate.TranslateLanguage.LITHUANIAN),
    LATVIAN("lv", com.google.mlkit.nl.translate.TranslateLanguage.LATVIAN),
    MACEDONIAN("mk", com.google.mlkit.nl.translate.TranslateLanguage.MACEDONIAN),
    MARATHI("mr", com.google.mlkit.nl.translate.TranslateLanguage.MARATHI),
    MALAY("ms", com.google.mlkit.nl.translate.TranslateLanguage.MALAY),
    MALTESE("mt", com.google.mlkit.nl.translate.TranslateLanguage.MALTESE),
    DUTCH("nl", com.google.mlkit.nl.translate.TranslateLanguage.DUTCH),
    NORWEGIAN("no", com.google.mlkit.nl.translate.TranslateLanguage.NORWEGIAN),
    POLISH("pl", com.google.mlkit.nl.translate.TranslateLanguage.POLISH),
    PORTUGUESE("pt", com.google.mlkit.nl.translate.TranslateLanguage.PORTUGUESE),
    ROMANIAN("ro", com.google.mlkit.nl.translate.TranslateLanguage.ROMANIAN),
    RUSSIAN("ru", com.google.mlkit.nl.translate.TranslateLanguage.RUSSIAN),
    SLOVAK("sk", com.google.mlkit.nl.translate.TranslateLanguage.SLOVAK),
    SLOVENIAN("sl", com.google.mlkit.nl.translate.TranslateLanguage.SLOVENIAN),
    ALBANIAN("sq", com.google.mlkit.nl.translate.TranslateLanguage.ALBANIAN),
    SWEDISH("sv", com.google.mlkit.nl.translate.TranslateLanguage.SWEDISH),
    SWAHILI("sw", com.google.mlkit.nl.translate.TranslateLanguage.SWAHILI),
    TAMIL("ta", com.google.mlkit.nl.translate.TranslateLanguage.TAMIL),
    TELUGU("te", com.google.mlkit.nl.translate.TranslateLanguage.TELUGU),
    THAI("th", com.google.mlkit.nl.translate.TranslateLanguage.THAI),
    TAGALOG("tl", com.google.mlkit.nl.translate.TranslateLanguage.TAGALOG),
    TURKISH("tr", com.google.mlkit.nl.translate.TranslateLanguage.TURKISH),
    UKRAINIAN("uk", com.google.mlkit.nl.translate.TranslateLanguage.UKRAINIAN),
    URDU("ur", com.google.mlkit.nl.translate.TranslateLanguage.URDU),
    VIETNAMESE("vi", com.google.mlkit.nl.translate.TranslateLanguage.VIETNAMESE),
    CHINESE("zh", com.google.mlkit.nl.translate.TranslateLanguage.CHINESE);
    val displayName: String get() = java.util.Locale.forLanguageTag(code).getDisplayLanguage(java.util.Locale.getDefault())
    val nativeName: String get() = java.util.Locale.forLanguageTag(code).let { it.getDisplayLanguage(it) }

    companion object {
        const val AUTO_DETECT = "auto"

        fun fromCode(code: String): Language? {
            return entries.find { it.code == code.lowercase() }
        }

        fun isValidCode(code: String): Boolean {
            return code == AUTO_DETECT || fromCode(code) != null
        }

        fun toJsonArray(): JSONArray {
            val array = JSONArray()
            // Add auto-detect first
            array.put(JSONObject().apply {
                put("code", AUTO_DETECT)
                put("name", "Auto-detect")
                put("nativeName", "Auto-detect")
            })
            // Add all languages
            entries.forEach { lang ->
                array.put(JSONObject().apply {
                    put("code", lang.code)
                    put("name", lang.displayName)
                    put("nativeName", lang.nativeName)
                })
            }
            return array
        }
    }
}
