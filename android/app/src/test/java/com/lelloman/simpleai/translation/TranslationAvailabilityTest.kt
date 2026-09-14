package com.lelloman.simpleai.translation

import org.junit.Assert.*
import org.junit.Test

class TranslationAvailabilityTest {
    @Test fun builtInEnglishIsAvailableWithoutADownload() {
        assertEquals(setOf("en"), TranslationAvailability.available(emptySet()))
        assertTrue(TranslationAvailability.downloaded(setOf("en")).isEmpty())
    }

    @Test fun otherPacksDoNotCreateAnEnglishDownload() {
        assertEquals(setOf("it"), TranslationAvailability.downloaded(setOf("it")))
        assertEquals(setOf("it", "en"), TranslationAvailability.available(setOf("it")))
    }
}
