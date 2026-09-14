package com.lelloman.simpleai.api

import com.lelloman.simpleai.BuildConfig
import org.junit.Assert.*
import org.junit.Test

class ServiceTranslationClientTest {
    @Test fun sendsCurrentProtocolAndParsesServiceResult() {
        val result = ServiceTranslationClient.request("hello", "en", "it") { protocol, text, source, target ->
            assertEquals(BuildConfig.MAX_PROTOCOL_VERSION, protocol)
            assertEquals(listOf("hello", "en", "it"), listOf(text, source, target))
            """{"status":"success","data":{"translatedText":"ciao","detectedSourceLang":"en"}}"""
        }.getOrThrow()
        assertEquals("ciao", result.translatedText)
    }
    @Test fun preservesServiceFailureForVisibleRecovery() {
        val failure = ServiceTranslationClient.request("hi", "en", "it") { _, _, _, _ ->
            """{"status":"error","error":{"message":"Download Italian first"}}"""
        }.exceptionOrNull()
        assertEquals("Download Italian first", failure?.message)
    }
}
