package com.lelloman.simpleai.capability

import org.junit.Assert.*
import org.junit.Test

class CapabilityStatusTest {

    // =========================================================================
    // CapabilityStatus Tests
    // =========================================================================

    @Test
    fun `NotDownloaded contains total bytes`() {
        val status = CapabilityStatus.NotDownloaded(totalBytes = 1024L)
        assertEquals(1024L, status.totalBytes)
    }

    @Test
    fun `Downloading contains downloaded and total bytes`() {
        val status = CapabilityStatus.Downloading(
            downloadedBytes = 512L,
            totalBytes = 1024L
        )
        assertEquals(512L, status.downloadedBytes)
        assertEquals(1024L, status.totalBytes)
    }

    @Test
    fun `Downloading progress is calculated correctly`() {
        val status = CapabilityStatus.Downloading(
            downloadedBytes = 500L,
            totalBytes = 1000L
        )
        assertEquals(0.5f, status.progress, 0.001f)
    }

    @Test
    fun `Downloading progress is 0 when totalBytes is 0`() {
        val status = CapabilityStatus.Downloading(
            downloadedBytes = 100L,
            totalBytes = 0L
        )
        assertEquals(0f, status.progress, 0.001f)
    }

    @Test
    fun `Downloading progress is 1 when fully downloaded`() {
        val status = CapabilityStatus.Downloading(
            downloadedBytes = 1000L,
            totalBytes = 1000L
        )
        assertEquals(1.0f, status.progress, 0.001f)
    }

    @Test
    fun `Ready is a singleton object`() {
        val ready1 = CapabilityStatus.Ready
        val ready2 = CapabilityStatus.Ready
        assertSame(ready1, ready2)
    }

    @Test
    fun `Error contains message`() {
        val status = CapabilityStatus.Error(message = "Something went wrong")
        assertEquals("Something went wrong", status.message)
    }

    @Test
    fun `Error canRetry defaults to true`() {
        val status = CapabilityStatus.Error(message = "Error")
        assertTrue(status.canRetry)
    }

    @Test
    fun `Error canRetry can be set to false`() {
        val status = CapabilityStatus.Error(message = "Fatal error", canRetry = false)
        assertFalse(status.canRetry)
    }

    // =========================================================================
    // Capability Tests
    // =========================================================================

    @Test
    fun `Capability isReady returns true when status is Ready`() {
        val capability = Capability(
            id = CapabilityId.LOCAL_AI,
            name = "Local AI",
            description = "On-device LLM",
            status = CapabilityStatus.Ready
        )
        assertTrue(capability.isReady)
    }

    @Test
    fun `Capability isReady returns false when status is not Ready`() {
        val capability = Capability(
            id = CapabilityId.LOCAL_AI,
            name = "Local AI",
            description = "On-device LLM",
            status = CapabilityStatus.NotDownloaded(1024L)
        )
        assertFalse(capability.isReady)
    }

    @Test
    fun `Capability isDownloading returns true when status is Downloading`() {
        val capability = Capability(
            id = CapabilityId.VOICE_COMMANDS,
            name = "Voice Commands",
            description = "NLU",
            status = CapabilityStatus.Downloading(500L, 1000L)
        )
        assertTrue(capability.isDownloading)
    }

    @Test
    fun `Capability isDownloading returns false when status is not Downloading`() {
        val capability = Capability(
            id = CapabilityId.VOICE_COMMANDS,
            name = "Voice Commands",
            description = "NLU",
            status = CapabilityStatus.Ready
        )
        assertFalse(capability.isDownloading)
    }

    @Test
    fun `Capability canDownload returns true when NotDownloaded`() {
        val capability = Capability(
            id = CapabilityId.LOCAL_AI,
            name = "Local AI",
            description = "LLM",
            status = CapabilityStatus.NotDownloaded(1024L)
        )
        assertTrue(capability.canDownload)
    }

    @Test
    fun `Capability canDownload returns true when Error`() {
        val capability = Capability(
            id = CapabilityId.LOCAL_AI,
            name = "Local AI",
            description = "LLM",
            status = CapabilityStatus.Error("Failed")
        )
        assertTrue(capability.canDownload)
    }

    @Test
    fun `Capability canDownload returns false when Ready`() {
        val capability = Capability(
            id = CapabilityId.CLOUD_AI,
            name = "Cloud AI",
            description = "Cloud",
            status = CapabilityStatus.Ready
        )
        assertFalse(capability.canDownload)
    }

    @Test
    fun `Capability canDownload returns false when Downloading`() {
        val capability = Capability(
            id = CapabilityId.TRANSLATION,
            name = "Translation",
            description = "ML Kit",
            status = CapabilityStatus.Downloading(100L, 200L)
        )
        assertFalse(capability.canDownload)
    }

    // =========================================================================
    // TranslationCapability Tests
    // =========================================================================

    @Test
    fun `TranslationCapability hasLanguages returns true when languages downloaded`() {
        val capability = TranslationCapability(
            baseCapability = Capability(
                id = CapabilityId.TRANSLATION,
                name = "Translation",
                description = "ML Kit",
                status = CapabilityStatus.Ready
            ),
            downloadedLanguages = setOf("en", "it"),
            availableLanguages = setOf("en", "it", "fr", "de")
        )
        assertTrue(capability.hasLanguages)
    }

    @Test
    fun `TranslationCapability hasLanguages returns false when no languages`() {
        val capability = TranslationCapability(
            baseCapability = Capability(
                id = CapabilityId.TRANSLATION,
                name = "Translation",
                description = "ML Kit",
                status = CapabilityStatus.NotDownloaded(0)
            ),
            downloadedLanguages = emptySet(),
            availableLanguages = setOf("en", "it", "fr", "de")
        )
        assertFalse(capability.hasLanguages)
    }

    // =========================================================================
    // CapabilityId Tests
    // =========================================================================

    @Test
    fun `all CapabilityId values exist`() {
        val ids = CapabilityId.entries
        assertEquals(4, ids.size)
        assertTrue(ids.contains(CapabilityId.VOICE_COMMANDS))
        assertTrue(ids.contains(CapabilityId.TRANSLATION))
        assertTrue(ids.contains(CapabilityId.CLOUD_AI))
        assertTrue(ids.contains(CapabilityId.LOCAL_AI))
    }
}
