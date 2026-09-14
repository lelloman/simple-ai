package com.lelloman.simpleai.cloud

import android.content.SharedPreferences
import io.mockk.*
import org.junit.Assert.*
import org.junit.Test

class CloudSettingsTest {
    @Test fun overridePersistsAndEmptyDisablesBuildDefault() {
        val values = mutableMapOf<String, String>()
        val prefs = mockk<SharedPreferences>()
        val editor = mockk<SharedPreferences.Editor>()
        every { prefs.getString(any(), any()) } answers { values[firstArg()] ?: secondArg() }
        every { prefs.edit() } returns editor
        every { editor.putString(any(), any()) } answers { values[firstArg()] = secondArg(); editor }
        every { editor.commit() } returns true
        val settings = CloudSettings(prefs, "https://default.example")
        assertTrue(settings.save("  https://custom.example/api/  "))
        assertEquals("https://custom.example/api/", settings.endpoint.value)
        assertEquals(settings.endpoint.value, CloudSettings(prefs, "https://default.example").endpoint.value)
        assertFalse(settings.save("http://insecure.example"))
        assertEquals("https://custom.example/api/", settings.endpoint.value)
        assertTrue(settings.save(""))
        assertEquals("", CloudSettings(prefs, "https://default.example").endpoint.value)
    }

    @Test fun failedPersistenceDoesNotChangeActiveEndpoint() {
        val prefs = mockk<SharedPreferences>()
        val editor = mockk<SharedPreferences.Editor>()
        every { prefs.getString(any(), any()) } returns "https://original.example"
        every { prefs.edit() } returns editor
        every { editor.putString(any(), any()) } returns editor
        every { editor.commit() } returns false
        val settings = CloudSettings(prefs)
        assertFalse(settings.save("https://replacement.example"))
        assertEquals("https://original.example", settings.endpoint.value)
    }
}
