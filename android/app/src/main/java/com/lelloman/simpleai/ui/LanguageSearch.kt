package com.lelloman.simpleai.ui

import java.text.Normalizer
import java.util.Locale

internal fun nativeLanguageName(code: String): String = Locale.forLanguageTag(code).let { it.getDisplayLanguage(it) }
private fun searchKey(value: String): String = Normalizer.normalize(value.trim(), Normalizer.Form.NFD)
    .replace(Regex("\\p{M}+"), "").lowercase(Locale.ROOT)
internal fun languageMatches(code: String, name: String, query: String): Boolean {
    val key = searchKey(query)
    return listOf(code, name, nativeLanguageName(code)).any { searchKey(it).contains(key) }
}
