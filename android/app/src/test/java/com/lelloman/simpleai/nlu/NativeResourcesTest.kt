package com.lelloman.simpleai.nlu

import org.junit.Assert.*
import org.junit.Test

class NativeResourcesTest {
    @Test fun headsOnlySessionReplacementClosesOldSessionWithoutPatchMetadata() {
        var oldClosed = 0
        var newClosed = 0
        val slot = NativeResourceSlot<AutoCloseable>()
        slot.value = AutoCloseable { oldClosed++ }
        slot.value = AutoCloseable { newClosed++ }
        assertEquals(1, oldClosed)
        assertEquals(0, newClosed)
        slot.close()
        slot.close()
        assertEquals(1, newClosed)
    }

    @Test fun inferenceFailureClosesResultAndInputsEvenWhenACloserThrows() {
        val closed = mutableListOf<String>()
        val error = runCatching {
            NativeResources().use { resources ->
                resources.own(AutoCloseable { closed.add("ids") })
                resources.own(AutoCloseable { closed.add("mask"); error("close failure") })
                resources.own(AutoCloseable { closed.add("result") })
                error("inference failure")
            }
        }.exceptionOrNull()!!
        assertEquals("inference failure", error.message)
        assertEquals(listOf("result", "mask", "ids"), closed)
        assertEquals("close failure", error.suppressed.single().message)
    }

    @Test fun allocationFailureStillClosesAlreadyCreatedTensor() {
        var closed = false
        runCatching { NativeResources().use { it.own(AutoCloseable { closed = true }); error("second tensor allocation") } }
        assertTrue(closed)
    }
}
