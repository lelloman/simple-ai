package com.lelloman.simpleai.nlu

import java.io.File
import java.security.MessageDigest
import org.junit.Assert.*
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder

class WorkingModelFileTest {
    @get:Rule val folder = TemporaryFolder()
    private fun hash(bytes: ByteArray) = MessageDigest.getInstance("SHA-256").digest(bytes).joinToString("") { "%02x".format(it) }

    @Test fun `patching and abandoned working files never change the base and restart restores it`() {
        val original = byteArrayOf(1, 2, 3, 4)
        val base = folder.newFile("base").apply { writeBytes(original) }
        val work = File(folder.root, "work")
        val first = WorkingModelFile(base, work, hash(original))
        first.prepare().writeBytes(byteArrayOf(9, 9, 9, 9))
        assertArrayEquals(original, base.readBytes())
        // Simulate a previous process disappearing without cleanup.
        val restarted = WorkingModelFile(base, work, hash(original))
        assertArrayEquals(original, restarted.prepare().readBytes())
        restarted.discard()
        assertFalse(work.exists())
        assertArrayEquals(original, base.readBytes())
    }

    @Test fun `legacy corrupted base is rejected before use`() {
        val base = folder.newFile("base").apply { writeBytes(byteArrayOf(9)) }
        val work = File(folder.root, "work")
        assertThrows(IllegalStateException::class.java) {
            WorkingModelFile(base, work, hash(byteArrayOf(1))).prepare()
        }
        assertFalse(work.exists())
        assertFalse(File(folder.root, "work.tmp").exists())
        assertArrayEquals(byteArrayOf(9), base.readBytes())
    }
}
