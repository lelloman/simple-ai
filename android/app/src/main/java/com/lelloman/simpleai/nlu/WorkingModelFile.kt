package com.lelloman.simpleai.nlu

import java.io.File
import java.io.FileOutputStream
import java.security.MessageDigest

/** Never map the downloaded artifact writable. A previous process's working file is disposable. */
class WorkingModelFile(private val base: File, val file: File, private val expectedSha256: String) {
    fun prepare(): File {
        val temporary = File(file.parentFile, "${file.name}.tmp")
        try {
            val digest = MessageDigest.getInstance("SHA-256")
            base.inputStream().use { input ->
                FileOutputStream(temporary).use { output ->
                    val buffer = ByteArray(65536)
                    while (true) {
                        val count = input.read(buffer)
                        if (count < 0) break
                        digest.update(buffer, 0, count)
                        output.write(buffer, 0, count)
                    }
                    output.fd.sync()
                }
            }
            val actual = digest.digest().joinToString("") { "%02x".format(it) }
            check(actual == expectedSha256) {
                "Voice Commands model integrity check failed. Download it again to restore the original model."
            }
            check(temporary.renameTo(file)) { "Could not prepare the Voice Commands working model" }
            return file
        } finally { temporary.delete() }
    }

    fun discard() { file.delete() }
}
