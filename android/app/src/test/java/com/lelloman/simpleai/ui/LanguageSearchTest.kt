package com.lelloman.simpleai.ui

import org.junit.Assert.*
import org.junit.Test

class LanguageSearchTest {
    @Test fun searchesCodeDisplayAndNativeNamesIgnoringCaseAndAccents() {
        assertTrue(languageMatches("de", "German", "deutsch"))
        assertTrue(languageMatches("fr", "French", "FRANCAIS"))
        assertTrue(languageMatches("ja", "Japanese", "日本語"))
        assertTrue(languageMatches("it", "Italian", " IT "))
        assertFalse(languageMatches("it", "Italian", "German"))
    }
}
