package com.lelloman.simpleai.translation

import org.junit.Assert.*
import org.junit.Test

class LanguageTest {

    // Note: toJsonArray() tests are commented out because org.json.JSONArray/JSONObject
    // are Android classes not available in pure JVM unit tests. These would require
    // Robolectric or instrumented tests to run properly.

    @Test
    fun `fromCode returns correct language for valid codes`() {
        assertEquals(Language.ENGLISH, Language.fromCode("en"))
        assertEquals(Language.SPANISH, Language.fromCode("es"))
        assertEquals(Language.FRENCH, Language.fromCode("fr"))
        assertEquals(Language.GERMAN, Language.fromCode("de"))
        assertEquals(Language.ITALIAN, Language.fromCode("it"))
        assertEquals(Language.PORTUGUESE, Language.fromCode("pt"))
        assertEquals(Language.DUTCH, Language.fromCode("nl"))
        assertEquals(Language.POLISH, Language.fromCode("pl"))
        assertEquals(Language.RUSSIAN, Language.fromCode("ru"))
        assertEquals(Language.UKRAINIAN, Language.fromCode("uk"))
        assertEquals(Language.CHINESE, Language.fromCode("zh"))
        assertEquals(Language.JAPANESE, Language.fromCode("ja"))
        assertEquals(Language.KOREAN, Language.fromCode("ko"))
        assertEquals(Language.ARABIC, Language.fromCode("ar"))
        assertEquals(Language.HINDI, Language.fromCode("hi"))
        assertEquals(Language.TURKISH, Language.fromCode("tr"))
        assertEquals(Language.VIETNAMESE, Language.fromCode("vi"))
        assertEquals(Language.THAI, Language.fromCode("th"))
        assertEquals(Language.INDONESIAN, Language.fromCode("id"))
    }

    @Test
    fun `fromCode is case insensitive`() {
        assertEquals(Language.ENGLISH, Language.fromCode("EN"))
        assertEquals(Language.ENGLISH, Language.fromCode("En"))
        assertEquals(Language.JAPANESE, Language.fromCode("JA"))
    }

    @Test
    fun `fromCode returns null for invalid codes`() {
        assertNull(Language.fromCode("invalid"))
        assertNull(Language.fromCode(""))
        assertNull(Language.fromCode("xyz"))
        assertNull(Language.fromCode("auto")) // auto is not a Language enum value
    }

    @Test
    fun `isValidCode returns true for all language codes`() {
        Language.entries.forEach { language ->
            assertTrue("${language.code} should be valid", Language.isValidCode(language.code))
        }
    }

    @Test
    fun `isValidCode returns true for auto`() {
        assertTrue(Language.isValidCode("auto"))
        assertTrue(Language.isValidCode(Language.AUTO_DETECT))
    }

    @Test
    fun `isValidCode returns false for invalid codes`() {
        assertFalse(Language.isValidCode("invalid"))
        assertFalse(Language.isValidCode(""))
        assertFalse(Language.isValidCode("xyz"))
    }

    @Test
    fun `AUTO_DETECT constant has correct value`() {
        assertEquals("auto", Language.AUTO_DETECT)
    }

    // The following tests are skipped in JVM unit tests because they require Android's org.json classes.
    // Uncomment and move to instrumented tests if needed.
    /*
    @Test
    fun `toJsonArray contains all languages plus auto-detect`() {
        val jsonArray = Language.toJsonArray()
        assertEquals(Language.entries.size + 1, jsonArray.length())
    }

    @Test
    fun `toJsonArray first entry is auto-detect`() {
        val jsonArray = Language.toJsonArray()
        val firstEntry = jsonArray.getJSONObject(0)
        assertEquals("auto", firstEntry.getString("code"))
        assertEquals("Auto-detect", firstEntry.getString("name"))
        assertEquals("Auto-detect", firstEntry.getString("nativeName"))
    }

    @Test
    fun `toJsonArray contains correct structure for each language`() {
        val jsonArray = Language.toJsonArray()
        val englishEntry = jsonArray.getJSONObject(1)
        assertEquals("en", englishEntry.getString("code"))
        assertEquals("English", englishEntry.getString("name"))
        assertEquals("English", englishEntry.getString("nativeName"))
    }
    */

    @Test
    fun `all languages have non-empty properties`() {
        Language.entries.forEach { language ->
            assertTrue("${language.name} code should not be empty", language.code.isNotEmpty())
            assertTrue("${language.name} displayName should not be empty", language.displayName.isNotEmpty())
            assertTrue("${language.name} nativeName should not be empty", language.nativeName.isNotEmpty())
        }
    }

    @Test
    fun `all language codes are unique`() {
        val codes = Language.entries.map { it.code }
        assertEquals("All language codes should be unique", codes.size, codes.toSet().size)
    }

    @Test
    fun `all language codes are lowercase and two characters`() {
        Language.entries.forEach { language ->
            assertEquals("${language.name} code should be 2 characters", 2, language.code.length)
            assertEquals("${language.name} code should be lowercase", language.code.lowercase(), language.code)
        }
    }
}
