package com.lelloman.simpleai.capability

import android.content.Context
import android.content.SharedPreferences
import android.util.Log
import io.mockk.every
import io.mockk.mockk
import io.mockk.mockkStatic
import io.mockk.unmockkStatic
import io.mockk.verify
import org.junit.After
import org.junit.Assert.*
import org.junit.Before
import org.junit.Test

class CapabilityManagerTest {
    @Test
    fun `shared translation inventory updates readiness for service clients`() {
        val manager = CapabilityManager(mockContext, "https://cloud.example")
        manager.syncTranslationLanguages(setOf("en", "it"))
        assertEquals(setOf("it"), manager.downloadedLanguages.value)
        assertEquals(CapabilityStatus.Ready, manager.translationStatus.value)
        manager.syncTranslationLanguages(emptySet())
        assertTrue(manager.downloadedLanguages.value.isEmpty())
        assertTrue(manager.translationStatus.value is CapabilityStatus.NotDownloaded)
    }

    private lateinit var mockContext: Context
    private lateinit var mockPrefs: SharedPreferences
    private lateinit var mockEditor: SharedPreferences.Editor
    private lateinit var manager: CapabilityManager

    @Before
    fun setup() {
        mockkStatic(Log::class)
        every { Log.e(any(), any()) } returns 0
        every { Log.e(any(), any(), any()) } returns 0

        mockEditor = mockk(relaxed = true) {
            every { putString(any(), any()) } returns this@mockk
            every { apply() } returns Unit
        }

        mockPrefs = mockk {
            every { getString(any(), any()) } returns null
            every { edit() } returns mockEditor
        }

        mockContext = mockk {
            every { getSharedPreferences(any(), any()) } returns mockPrefs
        }

        manager = CapabilityManager(mockContext, "https://cloud.example")
    }

    @After
    fun teardown() {
        unmockkStatic(Log::class)
    }

    // =========================================================================
    // Initial State Tests
    // =========================================================================

    @Test
    fun `voiceCommandsStatus is Checking initially`() {
        val status = manager.voiceCommandsStatus.value
        assertEquals(CapabilityStatus.Checking, status)
    }

    @Test
    fun `translationStatus is Checking initially`() {
        val status = manager.translationStatus.value
        assertEquals(CapabilityStatus.Checking, status)
    }

    @Test
    fun `cloudAiStatus is Ready initially`() {
        val status = manager.cloudAiStatus.value
        assertTrue(status is CapabilityStatus.Ready)
    }

    @Test
    fun `localAiStatus is Checking initially`() {
        val status = manager.localAiStatus.value
        assertEquals(CapabilityStatus.Checking, status)
    }

    @Test
    fun `downloadedLanguages is empty initially`() {
        assertTrue(manager.downloadedLanguages.value.isEmpty())
    }

    // =========================================================================
    // Voice Commands Status Updates
    // =========================================================================

    @Test
    fun `updateVoiceCommandsStatus changes status`() {
        manager.updateVoiceCommandsStatus(CapabilityStatus.Ready)
        assertTrue(manager.voiceCommandsStatus.value is CapabilityStatus.Ready)
    }

    @Test
    fun `updateVoiceCommandsStatus to Downloading`() {
        manager.updateVoiceCommandsStatus(CapabilityStatus.Downloading(50L, 100L))
        val status = manager.voiceCommandsStatus.value
        assertTrue(status is CapabilityStatus.Downloading)
        assertEquals(0.5f, (status as CapabilityStatus.Downloading).progress, 0.01f)
    }

    @Test
    fun `updateVoiceCommandsStatus to Error`() {
        manager.updateVoiceCommandsStatus(CapabilityStatus.Error("Failed"))
        val status = manager.voiceCommandsStatus.value
        assertTrue(status is CapabilityStatus.Error)
        assertEquals("Failed", (status as CapabilityStatus.Error).message)
    }

    // =========================================================================
    // Translation Language Management
    // =========================================================================

    @Test
    fun `addTranslationLanguage adds language`() {
        manager.addTranslationLanguage("it")
        assertTrue(manager.downloadedLanguages.value.contains("it"))
    }

    @Test
    fun `addTranslationLanguage does not invent an English download`() {
        manager.addTranslationLanguage("it")
        assertFalse(manager.downloadedLanguages.value.contains("en"))
        assertTrue(manager.downloadedLanguages.value.contains("it"))
    }

    @Test
    fun `addTranslationLanguage sets status to Ready`() {
        manager.addTranslationLanguage("fr")
        assertTrue(manager.translationStatus.value is CapabilityStatus.Ready)
    }

    @Test
    fun `addTranslationLanguage persists to SharedPreferences`() {
        manager.addTranslationLanguage("de")
        verify { mockEditor.putString("translation_languages", any()) }
        verify { mockEditor.apply() }
    }

    @Test
    fun `removeTranslationLanguage removes language`() {
        manager.addTranslationLanguage("it")
        manager.addTranslationLanguage("fr")
        manager.removeTranslationLanguage("it")

        assertFalse(manager.downloadedLanguages.value.contains("it"))
        assertTrue(manager.downloadedLanguages.value.contains("fr"))
        assertFalse(manager.downloadedLanguages.value.contains("en"))
    }

    @Test
    fun `removeTranslationLanguage does not include built-in English in downloads`() {
        manager.addTranslationLanguage("it")
        manager.removeTranslationLanguage("en")

        // English is built in and is never a downloaded pack.
        assertFalse(manager.downloadedLanguages.value.contains("en"))
    }

    @Test
    fun `removeTranslationLanguage removes English when last language removed`() {
        manager.addTranslationLanguage("it")
        manager.removeTranslationLanguage("it")

        // Both should be gone
        assertTrue(manager.downloadedLanguages.value.isEmpty())
    }

    @Test
    fun `removeTranslationLanguage sets status to NotDownloaded when empty`() {
        manager.addTranslationLanguage("it")
        manager.removeTranslationLanguage("it")

        assertTrue(manager.translationStatus.value is CapabilityStatus.NotDownloaded)
    }

    @Test
    fun `syncTranslationLanguages replaces all languages`() {
        manager.addTranslationLanguage("it")
        manager.addTranslationLanguage("fr")

        manager.syncTranslationLanguages(setOf("en", "de", "es"))

        assertEquals(setOf("de", "es"), manager.downloadedLanguages.value)
    }

    // =========================================================================
    // Local AI Status Updates
    // =========================================================================

    @Test
    fun `updateLocalAiStatus changes status`() {
        manager.updateLocalAiStatus(CapabilityStatus.Ready)
        assertTrue(manager.localAiStatus.value is CapabilityStatus.Ready)
    }

    @Test
    fun `updateLocalAiStatus to Downloading with progress`() {
        manager.updateLocalAiStatus(CapabilityStatus.Downloading(500_000_000L, 1_000_000_000L))
        val status = manager.localAiStatus.value
        assertTrue(status is CapabilityStatus.Downloading)
        assertEquals(0.5f, (status as CapabilityStatus.Downloading).progress, 0.01f)
    }

    // =========================================================================
    // Capability Queries
    // =========================================================================

    @Test
    fun `getCapability returns correct capability for VOICE_COMMANDS`() {
        val capability = manager.getCapability(CapabilityId.VOICE_COMMANDS)
        assertEquals(CapabilityId.VOICE_COMMANDS, capability.id)
        assertEquals("Voice Commands", capability.name)
    }

    @Test
    fun `getCapability returns correct capability for TRANSLATION`() {
        val capability = manager.getCapability(CapabilityId.TRANSLATION)
        assertEquals(CapabilityId.TRANSLATION, capability.id)
        assertEquals("Translation", capability.name)
    }

    @Test
    fun `getCapability returns correct capability for CLOUD_AI`() {
        val capability = manager.getCapability(CapabilityId.CLOUD_AI)
        assertEquals(CapabilityId.CLOUD_AI, capability.id)
        assertEquals("Cloud AI", capability.name)
        assertTrue(capability.isReady) // A configured Cloud AI endpoint is available
    }

    @Test
    fun `getCapability returns correct capability for LOCAL_AI`() {
        val capability = manager.getCapability(CapabilityId.LOCAL_AI)
        assertEquals(CapabilityId.LOCAL_AI, capability.id)
        assertEquals("Local AI", capability.name)
    }

    @Test
    fun `getAllCapabilities returns all 4 capabilities`() {
        val capabilities = manager.getAllCapabilities()
        assertEquals(4, capabilities.size)
        assertTrue(capabilities.any { it.id == CapabilityId.VOICE_COMMANDS })
        assertTrue(capabilities.any { it.id == CapabilityId.TRANSLATION })
        assertTrue(capabilities.any { it.id == CapabilityId.CLOUD_AI })
        assertTrue(capabilities.any { it.id == CapabilityId.LOCAL_AI })
    }

    @Test
    fun `getTranslationCapability includes downloaded languages`() {
        manager.addTranslationLanguage("it")
        manager.addTranslationLanguage("fr")

        val capability = manager.getTranslationCapability()
        assertFalse(capability.downloadedLanguages.contains("en"))
        assertTrue(capability.downloadedLanguages.contains("it"))
        assertTrue(capability.downloadedLanguages.contains("fr"))
    }

    @Test
    fun `getTranslationCapability includes available languages`() {
        val capability = manager.getTranslationCapability()
        assertFalse(capability.availableLanguages.isEmpty())
    }

    // =========================================================================
    // State Persistence
    // =========================================================================

    @Test
    fun `loads persisted translation languages on init`() {
        // Setup mock to return persisted languages
        every { mockPrefs.getString("translation_languages", null) } returns """["en","it","fr"]"""

        // Create new manager to trigger load
        val newManager = CapabilityManager(mockContext, "https://cloud.example")

        assertFalse(newManager.downloadedLanguages.value.contains("en"))
        assertTrue(newManager.downloadedLanguages.value.contains("it"))
        assertTrue(newManager.downloadedLanguages.value.contains("fr"))
    }

    @Test
    fun `keeps translation Checking until persisted inventory is verified`() {
        every { mockPrefs.getString("translation_languages", null) } returns """["en","it"]"""

        val newManager = CapabilityManager(mockContext, "https://cloud.example")

        assertEquals(CapabilityStatus.Checking, newManager.translationStatus.value)
    }

    @Test
    fun `handles corrupted persisted data gracefully`() {
        every { mockPrefs.getString("translation_languages", null) } returns "invalid json"

        // Should not throw
        val newManager = CapabilityManager(mockContext, "https://cloud.example")

        // Should fall back to empty
        assertTrue(newManager.downloadedLanguages.value.isEmpty())
    }
}
