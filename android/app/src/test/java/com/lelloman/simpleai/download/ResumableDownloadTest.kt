package com.lelloman.simpleai.download

import io.mockk.*
import kotlinx.coroutines.flow.toList
import kotlinx.coroutines.runBlocking
import okhttp3.*
import okhttp3.ResponseBody.Companion.toResponseBody
import org.junit.Assert.*
import org.junit.Test
import java.nio.file.Files
import java.io.File
import java.security.MessageDigest

class ResumableDownloadTest {
    private val url = "https://model.example/model"
    private fun response(code: Int, text: String, range: String? = null, etag: String = "\"v1\""): Response =
        Response.Builder().request(Request.Builder().url(url).build()).protocol(Protocol.HTTP_1_1)
            .code(code).message("test").header("ETag", etag).apply { range?.let { header("Content-Range", it) } }
            .body(text.toResponseBody()).build()

    private fun scenario(responses: List<Response>, hash: String? = null, check: (List<DownloadState>, File, List<Request>) -> Unit) = runBlocking {
        val directory = Files.createTempDirectory("resume-test").toFile()
        try {
            val target = File(directory, "model")
            File(target.path + ".tmp").writeText("abc")
            File(target.path + ".tmp.identity").writeText(url + "\n\"v1\"")
            val requests = mutableListOf<Request>()
            val client = mockk<OkHttpClient>()
            var index = 0
            every { client.newCall(any()) } answers {
                requests.add(firstArg())
                val result = responses[index++]
                mockk<okhttp3.Call>(relaxed = true).also { every { it.execute() } returns result }
            }
            val states = ResumableDownload(client).download(ModelConfig("test", url, "model", 1, sha256 = hash), target).toList()
            check(states, target, requests)
        } finally { directory.deleteRecursively() }
    }

    @Test fun fullResponseRestartsCounterAndFile() = scenario(listOf(response(200, "abcdef"))) { states, file, _ ->
        assertEquals(0L, states.filterIsInstance<DownloadState.Downloading>().first().downloadedBytes)
        assertEquals("abcdef", file.readText())
        assertEquals(DownloadState.Completed, states.last())
    }

    @Test fun matchingRangeAppends() = scenario(listOf(response(206, "def", "bytes 3-5/6"))) { states, file, requests ->
        assertEquals("abcdef", file.readText())
        assertEquals("bytes=3-", requests.single().header("Range"))
        assertEquals("\"v1\"", requests.single().header("If-Range"))
        assertEquals(DownloadState.Completed, states.last())
    }

    @Test fun unsatisfiableRangeRetriesWithoutOffset() = scenario(listOf(response(416, ""), response(200, "fresh"))) { states, file, requests ->
        assertNull(requests.last().header("Range"))
        assertEquals("fresh", file.readText())
        assertEquals(DownloadState.Completed, states.last())
    }

    @Test fun malformedRangeIsNotPromoted() = scenario(listOf(response(206, "def", "bytes 2-4/5"))) { states, file, _ ->
        assertFalse(file.exists())
        assertTrue(states.last() is DownloadState.Error)
    }

    @Test fun changedArtifactIsNotAppended() = scenario(listOf(response(206, "def", "bytes 3-5/6", "\"v2\""))) { states, file, _ ->
        assertFalse(file.exists())
        assertTrue(states.last() is DownloadState.Error)
    }

    @Test fun checksumMismatchCannotBecomeAnInstalledModel() = scenario(listOf(response(200, "bad")), "0".repeat(64)) { states, file, _ ->
        assertFalse(file.exists())
        assertFalse(File(file.path + ".tmp").exists())
        assertTrue(states.last() is DownloadState.Error)
    }

    @Test fun matchingChecksumAllowsPromotion() {
        val hash = MessageDigest.getInstance("SHA-256").digest("abcdef".toByteArray()).joinToString("") { "%02x".format(it) }
        scenario(listOf(response(206, "def", "bytes 3-5/6")), hash) { states, file, _ ->
            assertEquals("abcdef", file.readText())
            assertEquals(DownloadState.Completed, states.last())
        }
    }
}
