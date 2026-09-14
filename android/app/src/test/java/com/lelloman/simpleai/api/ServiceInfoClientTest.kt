package com.lelloman.simpleai.api

import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.put
import org.junit.Assert.*
import org.junit.Test

class ServiceInfoClientTest {
    @Test fun `UI protocol is accepted by the actual service handler`() {
        val capabilities = buildJsonObject { put("voiceCommands", buildJsonObject { put("status", "ready") }) }
        val actual = ServiceInfoClient.request { version ->
            ProtocolHandler.validateProtocol(version)
                ?: ProtocolHandler.success(version, buildJsonObject { put("capabilities", capabilities) })
        }
        assertEquals(capabilities, actual)
    }

    @Test fun `protocol failures preserve the actionable server message`() {
        val error = assertThrows(IllegalStateException::class.java) {
            ServiceInfoClient.request { ProtocolHandler.validateProtocol(0)!! }
        }
        assertTrue(error.message!!.contains("Please update your app"))
    }

    @Test fun `incomplete success is an error rather than empty state`() {
        assertThrows(IllegalArgumentException::class.java) {
            ServiceInfoClient.request { ProtocolHandler.success(it, buildJsonObject {}) }
        }
    }
}
