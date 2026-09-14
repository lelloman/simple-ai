package com.lelloman.simpleai.download

import org.junit.Assert.*
import org.junit.Test
import java.nio.file.Files
import java.io.File

class DownloadPolicyTest {
    @Test fun resumeBudgetIncludesWorkingCopyAndReserve() {
        assertEquals(700L + 1000L + 64 * 1024 * 1024, DownloadPolicy.requiredBytes(1000, 300, 1000))
        assertEquals(64L * 1024 * 1024, DownloadPolicy.requiredBytes(1000, 2000))
    }

    @Test fun usageIncludesAllModelAndPartialDirectories() {
        val root = Files.createTempDirectory("storage-test").toFile()
        try {
            for (name in listOf("files/models/model.gguf", "files/models/model.gguf.tmp", "files/nlu_models/base.onnx", "no_backup/mlkit/language", "files/nlu_models/working.onnx")) {
                File(root, name).apply { parentFile.mkdirs(); writeBytes(ByteArray(10)) }
            }
            assertEquals(50L, DownloadPolicy.usedBytes(root))
        } finally { root.deleteRecursively() }
    }
}
