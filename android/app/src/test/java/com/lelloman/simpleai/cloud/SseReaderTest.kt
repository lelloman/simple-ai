package com.lelloman.simpleai.cloud

import okio.Buffer
import org.junit.Assert.*
import org.junit.Test

class SseReaderTest {
    @Test fun handlesCommentsCrLfMultilineAndDone() {
        val events = mutableListOf<String>()
        SseReader.read(Buffer().writeUtf8(
            ": keepalive\r\nevent: message\r\ndata: {\"text\":\r\ndata: \"Ciao 🌍\"}\r\n\r\ndata: [DONE]\r\n\r\n"
        ), events::add)
        assertEquals(listOf("{\"text\":\n\"Ciao 🌍\"}", "[DONE]"), events)
    }

    @Test fun unexpectedEofIsAnErrorEvenAfterText() {
        val events = mutableListOf<String>()
        assertThrows(CloudException::class.java) {
            SseReader.read(Buffer().writeUtf8("data: hello\n\n"), events::add)
        }
        assertEquals(listOf("hello"), events)
    }

    @Test fun oversizedFrameIsRejected() {
        assertThrows(Exception::class.java) {
            SseReader.read(Buffer().writeUtf8("data: " + "x".repeat(129 * 1024) + "\n\n")) {}
        }
    }
}
