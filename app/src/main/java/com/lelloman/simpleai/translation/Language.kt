package com.lelloman.simpleai.translation

import org.json.JSONArray
import org.json.JSONObject

enum class Language(val code: String, val displayName: String, val nativeName: String) {
    ENGLISH("en", "English", "English"),
    SPANISH("es", "Spanish", "Español"),
    FRENCH("fr", "French", "Français"),
    GERMAN("de", "German", "Deutsch"),
    ITALIAN("it", "Italian", "Italiano"),
    PORTUGUESE("pt", "Portuguese", "Português"),
    DUTCH("nl", "Dutch", "Nederlands"),
    POLISH("pl", "Polish", "Polski"),
    RUSSIAN("ru", "Russian", "Русский"),
    UKRAINIAN("uk", "Ukrainian", "Українська"),
    CHINESE("zh", "Chinese", "中文"),
    JAPANESE("ja", "Japanese", "日本語"),
    KOREAN("ko", "Korean", "한국어"),
    ARABIC("ar", "Arabic", "العربية"),
    HINDI("hi", "Hindi", "हिन्दी"),
    TURKISH("tr", "Turkish", "Türkçe"),
    VIETNAMESE("vi", "Vietnamese", "Tiếng Việt"),
    THAI("th", "Thai", "ไทย"),
    INDONESIAN("id", "Indonesian", "Bahasa Indonesia");

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
