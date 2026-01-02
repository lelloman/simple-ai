package com.lelloman.simpleai.llm

import android.content.ContentResolver
import android.content.Context
import com.lelloman.simpleai.util.NoOpLogger
import io.mockk.every
import io.mockk.mockk
import org.junit.After
import org.junit.Assert.*
import org.junit.Before
import org.junit.Test
import java.io.File

/**
 * Simple test double for LlamaHelperWrapper that doesn't require MockK reflection.
 */
private class FakeLlamaHelperWrapper : LlamaHelperWrapper {
    var loadCalled = false
    var loadPath: String? = null
    var loadContextLength: Int? = null
    var predictCalled = false
    var predictPrompt: String? = null
    var stopPredictionCalled = false
    var abortCalled = false
    var releaseCalled = false
    var abortCount = 0
    var releaseCount = 0

    // If set, load() will invoke callback immediately
    var autoCompleteLoad = true

    override fun load(path: String, contextLength: Int, onLoaded: (Long) -> Unit) {
        loadCalled = true
        loadPath = path
        loadContextLength = contextLength
        if (autoCompleteLoad) {
            onLoaded(0L)
        }
    }

    override fun predict(prompt: String) {
        predictCalled = true
        predictPrompt = prompt
    }

    override fun stopPrediction() {
        stopPredictionCalled = true
    }

    override fun abort() {
        abortCalled = true
        abortCount++
    }

    override fun release() {
        releaseCalled = true
        releaseCount++
    }

    fun reset() {
        loadCalled = false
        loadPath = null
        loadContextLength = null
        predictCalled = false
        predictPrompt = null
        stopPredictionCalled = false
        abortCalled = false
        releaseCalled = false
        abortCount = 0
        releaseCount = 0
    }
}

class LlamaEngineTest {

    private lateinit var tempDir: File
    private lateinit var mockContext: Context
    private lateinit var mockContentResolver: ContentResolver
    private lateinit var fakeWrapper: FakeLlamaHelperWrapper
    private lateinit var engine: LlamaEngine

    @Before
    fun setup() {
        tempDir = File(System.getProperty("java.io.tmpdir"), "test_llm_${System.currentTimeMillis()}")
        tempDir.mkdirs()

        mockContentResolver = mockk(relaxed = true)
        mockContext = mockk(relaxed = true) {
            every { contentResolver } returns mockContentResolver
        }
        fakeWrapper = FakeLlamaHelperWrapper()
    }

    @After
    fun teardown() {
        if (::engine.isInitialized) {
            engine.release()
        }
        tempDir.deleteRecursively()
    }

    private fun createEngine(timeoutMs: Long = 100L): LlamaEngine {
        engine = LlamaEngine(
            context = mockContext,
            helperFactory = { _, _, _ -> fakeWrapper },
            logger = NoOpLogger,
            generationTimeoutMs = timeoutMs,
            uriResolver = { file -> "content://test/${file.name}" }
        )
        return engine
    }

    @Test
    fun `isLoaded returns false initially`() {
        val engine = createEngine()
        assertFalse(engine.isLoaded)
    }

    @Test
    fun `modelInfo is null initially`() {
        val engine = createEngine()
        assertNull(engine.modelInfo)
    }

    @Test
    fun `loadModel fails when file does not exist`() {
        val engine = createEngine()
        val nonExistentFile = File(tempDir, "nonexistent.gguf")

        val result = engine.loadModel(nonExistentFile)

        assertTrue(result.isFailure)
        assertTrue(result.exceptionOrNull() is IllegalArgumentException)
        assertFalse(engine.isLoaded)
    }

    @Test
    fun `loadModel succeeds when file exists and load callback fires`() {
        val engine = createEngine()
        val modelFile = File(tempDir, "model.gguf")
        modelFile.writeText("dummy model content")

        val result = engine.loadModel(modelFile)

        assertTrue(result.isSuccess)
        assertTrue(engine.isLoaded)
        assertNotNull(engine.modelInfo)
        assertEquals("model.gguf", engine.modelInfo?.name)
        assertTrue(fakeWrapper.loadCalled)
    }

    @Test
    fun `loadModel sets correct modelInfo properties`() {
        val engine = createEngine()
        val modelFile = File(tempDir, "test_model.gguf")
        modelFile.writeBytes(ByteArray(1024)) // 1KB file

        engine.loadModel(modelFile)

        val info = engine.modelInfo
        assertNotNull(info)
        assertEquals("test_model.gguf", info?.name)
        assertEquals(modelFile.absolutePath, info?.path)
        assertEquals(1024L, info?.sizeBytes)
        assertEquals(4096, info?.contextSize)
    }

    @Test
    fun `loadModel unloads previous model first`() {
        val engine = createEngine()
        val modelFile1 = File(tempDir, "model1.gguf")
        val modelFile2 = File(tempDir, "model2.gguf")
        modelFile1.writeText("model1")
        modelFile2.writeText("model2")

        engine.loadModel(modelFile1)
        engine.loadModel(modelFile2)

        // Should have called abort and release on previous model
        assertTrue(fakeWrapper.abortCount >= 1)
        assertTrue(fakeWrapper.releaseCount >= 1)

        assertEquals("model2.gguf", engine.modelInfo?.name)
    }

    @Test
    fun `unloadModel clears state`() {
        val engine = createEngine()
        val modelFile = File(tempDir, "model.gguf")
        modelFile.writeText("model")

        engine.loadModel(modelFile)
        assertTrue(engine.isLoaded)

        engine.unloadModel()

        assertFalse(engine.isLoaded)
        assertNull(engine.modelInfo)
    }

    @Test
    fun `unloadModel calls abort and release on wrapper`() {
        val engine = createEngine()
        val modelFile = File(tempDir, "model.gguf")
        modelFile.writeText("model")

        engine.loadModel(modelFile)
        fakeWrapper.reset() // Reset to check only the unload calls
        engine.unloadModel()

        assertTrue(fakeWrapper.abortCalled)
        assertTrue(fakeWrapper.releaseCalled)
    }

    @Test
    fun `generate fails when model not loaded`() {
        val engine = createEngine()

        val result = engine.generate("test prompt")

        assertTrue(result.isFailure)
        assertTrue(result.exceptionOrNull() is IllegalStateException)
    }

    @Test
    fun `generate calls predict on wrapper`() {
        val engine = createEngine()
        val modelFile = File(tempDir, "model.gguf")
        modelFile.writeText("model")

        engine.loadModel(modelFile)

        // generate() will timeout internally since we're not emitting events,
        // but it should still call predict on the wrapper
        engine.generate("test prompt")

        // Verify predict was called with the correct prompt
        assertTrue(fakeWrapper.predictCalled)
        assertEquals("test prompt", fakeWrapper.predictPrompt)
    }

    @Test
    fun `GenerationParams has correct defaults`() {
        val params = GenerationParams()

        assertEquals(512, params.maxTokens)
        assertEquals(0.7f, params.temperature)
        assertEquals(0.9f, params.topP)
        assertEquals(40, params.topK)
    }

    @Test
    fun `GenerationParams can be customized`() {
        val params = GenerationParams(
            maxTokens = 1024,
            temperature = 0.5f,
            topP = 0.8f,
            topK = 50
        )

        assertEquals(1024, params.maxTokens)
        assertEquals(0.5f, params.temperature)
        assertEquals(0.8f, params.topP)
        assertEquals(50, params.topK)
    }

    @Test
    fun `ModelInfo data class works correctly`() {
        val info = ModelInfo(
            name = "test.gguf",
            path = "/path/to/model",
            sizeBytes = 1024,
            contextSize = 2048
        )

        assertEquals("test.gguf", info.name)
        assertEquals("/path/to/model", info.path)
        assertEquals(1024L, info.sizeBytes)
        assertEquals(2048, info.contextSize)
    }

    @Test
    fun `LLMEngine interface defines required methods`() {
        // Verify the interface contract
        val engine: LLMEngine = createEngine()

        // These should compile and be accessible
        val isLoaded: Boolean = engine.isLoaded
        val modelInfo: ModelInfo? = engine.modelInfo
        val loadResult: Result<Unit> = engine.loadModel(File("test"))
        engine.unloadModel()
        val genResult1: Result<String> = engine.generate("prompt")
        val genResult2: Result<String> = engine.generate("prompt", GenerationParams())

        // Use values to avoid unused variable warnings
        assertFalse(isLoaded)
        assertNull(modelInfo)
        assertTrue(loadResult.isFailure)
        assertTrue(genResult1.isFailure)
        assertTrue(genResult2.isFailure)
    }

    @Test
    fun `release cleans up resources`() {
        val engine = createEngine()
        val modelFile = File(tempDir, "model.gguf")
        modelFile.writeText("model")

        engine.loadModel(modelFile)
        fakeWrapper.reset()
        engine.release()

        assertFalse(engine.isLoaded)
        assertTrue(fakeWrapper.abortCalled)
        assertTrue(fakeWrapper.releaseCalled)
    }
}
