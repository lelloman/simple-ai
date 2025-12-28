package com.lelloman.simpleai.download

import android.content.Context
import app.cash.turbine.test
import io.mockk.every
import io.mockk.mockk
import io.mockk.verify
import kotlinx.coroutines.test.runTest
import okhttp3.Call
import okhttp3.OkHttpClient
import okhttp3.Protocol
import okhttp3.Request
import okhttp3.Response
import okhttp3.ResponseBody.Companion.toResponseBody
import okio.Buffer
import org.junit.After
import org.junit.Assert.*
import org.junit.Before
import org.junit.Test
import java.io.File

class ModelDownloadManagerTest {

    private lateinit var tempDir: File
    private lateinit var mockContext: Context
    private lateinit var mockClient: OkHttpClient

    @Before
    fun setup() {
        tempDir = File(System.getProperty("java.io.tmpdir"), "test_models_${System.currentTimeMillis()}")
        tempDir.mkdirs()

        mockContext = mockk(relaxed = true)
        mockClient = mockk(relaxed = true)
    }

    @After
    fun teardown() {
        tempDir.deleteRecursively()
    }

    private fun createManager(): ModelDownloadManager {
        return ModelDownloadManager(
            context = mockContext,
            client = mockClient,
            modelsDirProvider = { tempDir }
        )
    }

    @Test
    fun `getModelFile returns correct file path`() {
        val manager = createManager()
        val config = ModelConfig("Test", "http://test.com/model.gguf", "model.gguf", 100)

        val file = manager.getModelFile(config)

        assertEquals("model.gguf", file.name)
        assertEquals(tempDir, file.parentFile)
    }

    @Test
    fun `getModelFile uses default config when none provided`() {
        val manager = createManager()

        val file = manager.getModelFile()

        assertEquals(DefaultModel.CONFIG.fileName, file.name)
    }

    @Test
    fun `isModelDownloaded returns false when file does not exist`() {
        val manager = createManager()
        val config = ModelConfig("Test", "http://test.com/model.gguf", "nonexistent.gguf", 100)

        assertFalse(manager.isModelDownloaded(config))
    }

    @Test
    fun `isModelDownloaded returns false when file is empty`() {
        val manager = createManager()
        val config = ModelConfig("Test", "http://test.com/model.gguf", "empty.gguf", 100)
        File(tempDir, "empty.gguf").createNewFile()

        assertFalse(manager.isModelDownloaded(config))
    }

    @Test
    fun `isModelDownloaded returns true when file exists with content`() {
        val manager = createManager()
        val config = ModelConfig("Test", "http://test.com/model.gguf", "model.gguf", 100)
        File(tempDir, "model.gguf").writeText("model content")

        assertTrue(manager.isModelDownloaded(config))
    }

    @Test
    fun `deleteModel deletes the model file`() {
        val manager = createManager()
        val config = ModelConfig("Test", "http://test.com/model.gguf", "model.gguf", 100)
        val modelFile = File(tempDir, "model.gguf")
        modelFile.writeText("model content")

        assertTrue(modelFile.exists())

        val result = manager.deleteModel(config)

        assertTrue(result)
        assertFalse(modelFile.exists())
    }

    @Test
    fun `deleteModel also deletes temp file`() {
        val manager = createManager()
        val config = ModelConfig("Test", "http://test.com/model.gguf", "model.gguf", 100)
        val tempFile = File(tempDir, "model.gguf.tmp")
        tempFile.writeText("partial download")

        assertTrue(tempFile.exists())

        manager.deleteModel(config)

        assertFalse(tempFile.exists())
    }

    @Test
    fun `deleteModel returns false when file does not exist`() {
        val manager = createManager()
        val config = ModelConfig("Test", "http://test.com/model.gguf", "nonexistent.gguf", 100)

        val result = manager.deleteModel(config)

        assertFalse(result)
    }

    @Test
    fun `downloadModel emits Idle then Downloading then Completed on success`() = runTest {
        val manager = createManager()
        val config = ModelConfig("Test", "http://test.com/model.gguf", "test.gguf", 1)
        val content = "test model content"

        val mockCall = mockk<Call>()
        val response = Response.Builder()
            .request(Request.Builder().url("http://test.com/model.gguf").build())
            .protocol(Protocol.HTTP_1_1)
            .code(200)
            .message("OK")
            .body(content.toResponseBody())
            .build()

        every { mockClient.newCall(any()) } returns mockCall
        every { mockCall.execute() } returns response

        manager.downloadModel(config).test {
            assertEquals(DownloadState.Idle, awaitItem())

            // May get Downloading states depending on timing
            var item = awaitItem()
            while (item is DownloadState.Downloading) {
                assertTrue(item.progress >= 0f && item.progress <= 1f)
                item = awaitItem()
            }

            assertEquals(DownloadState.Completed, item)
            awaitComplete()
        }

        // Verify file was created
        assertTrue(manager.isModelDownloaded(config))
    }

    @Test
    fun `downloadModel emits Error on HTTP failure`() = runTest {
        val manager = createManager()
        val config = ModelConfig("Test", "http://test.com/model.gguf", "test.gguf", 1)

        val mockCall = mockk<Call>()
        val response = Response.Builder()
            .request(Request.Builder().url("http://test.com/model.gguf").build())
            .protocol(Protocol.HTTP_1_1)
            .code(404)
            .message("Not Found")
            .body("".toResponseBody())
            .build()

        every { mockClient.newCall(any()) } returns mockCall
        every { mockCall.execute() } returns response

        manager.downloadModel(config).test {
            assertEquals(DownloadState.Idle, awaitItem())

            val errorState = awaitItem()
            assertTrue(errorState is DownloadState.Error)
            assertTrue((errorState as DownloadState.Error).message.contains("404"))

            awaitComplete()
        }
    }

    @Test
    fun `downloadModel emits Error on network exception`() = runTest {
        val manager = createManager()
        val config = ModelConfig("Test", "http://test.com/model.gguf", "test.gguf", 1)

        val mockCall = mockk<Call>()
        every { mockClient.newCall(any()) } returns mockCall
        every { mockCall.execute() } throws java.io.IOException("Network error")

        manager.downloadModel(config).test {
            assertEquals(DownloadState.Idle, awaitItem())

            val errorState = awaitItem()
            assertTrue(errorState is DownloadState.Error)
            assertTrue((errorState as DownloadState.Error).message.contains("Network error"))

            awaitComplete()
        }
    }

    @Test
    fun `downloadModel sends Range header when resuming`() = runTest {
        val manager = createManager()
        val config = ModelConfig("Test", "http://test.com/model.gguf", "resume.gguf", 1)

        // Create a partial temp file
        val tempFile = File(tempDir, "resume.gguf.tmp")
        tempFile.writeBytes(ByteArray(1000))

        val mockCall = mockk<Call>()
        val response = Response.Builder()
            .request(Request.Builder().url("http://test.com/model.gguf").build())
            .protocol(Protocol.HTTP_1_1)
            .code(206)
            .message("Partial Content")
            .body("remaining content".toResponseBody())
            .build()

        every { mockClient.newCall(any()) } returns mockCall
        every { mockCall.execute() } returns response

        manager.downloadModel(config).test {
            cancelAndConsumeRemainingEvents()
        }

        // Verify Range header was set
        verify {
            mockClient.newCall(match { request ->
                request.header("Range") == "bytes=1000-"
            })
        }
    }

    @Test
    fun `DefaultModel CONFIG has expected values`() {
        assertEquals("Qwen3-1.7B", DefaultModel.CONFIG.name)
        assertTrue(DefaultModel.CONFIG.url.startsWith("https://huggingface.co/"))
        assertTrue(DefaultModel.CONFIG.fileName.endsWith(".gguf"))
        assertTrue(DefaultModel.CONFIG.expectedSizeMb > 0)
    }

    @Test
    fun `ModelConfig data class works correctly`() {
        val config = ModelConfig("name", "url", "file.gguf", 500)

        assertEquals("name", config.name)
        assertEquals("url", config.url)
        assertEquals("file.gguf", config.fileName)
        assertEquals(500, config.expectedSizeMb)
    }

    @Test
    fun `DownloadState sealed class variants`() {
        val idle: DownloadState = DownloadState.Idle
        val downloading: DownloadState = DownloadState.Downloading(0.5f, 500, 1000)
        val completed: DownloadState = DownloadState.Completed
        val error: DownloadState = DownloadState.Error("test error")

        assertTrue(idle is DownloadState.Idle)
        assertTrue(downloading is DownloadState.Downloading)
        assertEquals(0.5f, (downloading as DownloadState.Downloading).progress)
        assertEquals(500L, downloading.downloadedBytes)
        assertEquals(1000L, downloading.totalBytes)
        assertTrue(completed is DownloadState.Completed)
        assertTrue(error is DownloadState.Error)
        assertEquals("test error", (error as DownloadState.Error).message)
    }
}
