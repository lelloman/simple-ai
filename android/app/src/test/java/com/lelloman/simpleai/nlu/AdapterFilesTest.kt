package com.lelloman.simpleai.nlu

import org.junit.Assert.*
import org.junit.Test
import java.io.*
import java.nio.ByteBuffer
import java.nio.ByteOrder

class AdapterFilesTest {
    private fun heads(rows: Int = 1, columns: Int = 768): ByteArray = ByteArrayOutputStream().also { bytes ->
        DataOutputStream(bytes).use { output ->
            fun int(value: Int) = output.writeInt(Integer.reverseBytes(value))
            repeat(2) { int(rows); int(columns); repeat(if (rows == 1 && columns == 768) 768 else 0) { int(0) } }
            repeat(2) { int(1); int(0) }
        }
    }.toByteArray()

    @Test fun completeReadsHandleFragmentedInput() {
        val stream = object : FilterInputStream(heads().inputStream()) {
            override fun read(bytes: ByteArray, offset: Int, length: Int) = super.read(bytes, offset, minOf(1, length))
        }
        assertEquals(768, AdapterFiles.heads(stream).intentHead.size)
    }

    @Test fun truncatedAndUnboundedHeadsAreRejected() {
        for (bytes in listOf(heads().copyOf(100), heads(Int.MAX_VALUE), heads(columns = 1024))) {
            assertTrue(runCatching { AdapterFiles.heads(bytes.inputStream()) }.isFailure)
        }
    }

    @Test fun metadataLimitsAndLabelCompatibilityAreEnforced() {
        val heads = AdapterFiles.heads(heads().inputStream())
        for (json in listOf(
            """{"intents":["a"],"slot_labels":["O"],"max_length":999999}""",
            """{"intents":["a","b"],"slot_labels":["O"]}""",
            """{"intents":["a"],"slot_labels":["bad"]}"""
        )) assertTrue(runCatching { AdapterFiles.config(json.byteInputStream(), heads) }.isFailure)
        assertTrue(runCatching { AdapterFiles.text("12345".byteInputStream(), 4) }.isFailure)
    }

    private fun patch(count: Int, offset: Long, length: Int, data: ByteArray = byteArrayOf()): ByteArray =
        ByteBuffer.allocate(24 + data.size).order(ByteOrder.LITTLE_ENDIAN)
            .put("LORA".toByteArray()).putInt(1).putInt(count).putLong(offset).putInt(length).put(data).array()

    @Test fun invalidPatchesCannotMutateLiveWeights() {
        val initial = ByteArray(16) { it.toByte() }
        for (bytes in listOf(patch(5000, 0, 1), patch(1, Long.MAX_VALUE, 4), patch(1, 0, Int.MAX_VALUE), patch(1, 0, 4, byteArrayOf(1)))) {
            val buffer = ByteBuffer.wrap(initial.copyOf())
            assertTrue(runCatching { LoraPatcher().applyPatch(buffer, bytes.inputStream(), "bad", "1") }.isFailure)
            assertArrayEquals(initial, buffer.array())
        }
    }
}
