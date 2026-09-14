package com.lelloman.simpleai.ui

import com.lelloman.simpleai.translation.Language
import com.lelloman.simpleai.translation.TranslationManager
import com.lelloman.simpleai.capability.CapabilityManager
import com.lelloman.simpleai.model.NluModel
import com.lelloman.simpleai.model.LocalAIModel
import org.junit.Assert.*
import org.junit.Test
import java.util.Locale

class MetadataTest {
    @Test fun everySupportedLanguageHasOneDisplayAndNativeName() {
        assertEquals(TranslationManager.SUPPORTED_LANGUAGES, Language.entries.map { it.code }.toSet())
        assertEquals(59, Language.entries.size)
        Language.entries.forEach { assertTrue(it.displayName.isNotBlank()); assertTrue(it.nativeName.isNotBlank()) }
    }
    @Test fun displayedSizesComeFromArtifactBytesAndUseDecimalUnits() {
        assertEquals(NluModel.SIZE_BYTES, CapabilityManager.VOICE_COMMANDS_MODEL_SIZE)
        val previous = Locale.getDefault()
        try {
            Locale.setDefault(Locale.US)
            assertEquals("533.6 MB", formatSize(NluModel.SIZE_BYTES))
            assertEquals("1.28 GB", formatSize(LocalAIModel.SIZE_BYTES))
        } finally { Locale.setDefault(previous) }
    }
}
