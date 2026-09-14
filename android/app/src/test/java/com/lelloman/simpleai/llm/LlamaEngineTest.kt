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
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.first
import org.nehuatl.llamacpp.LlamaHelper

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
    var loadResult: Result<Unit> = Result.success(Unit)
    var onPredict: (() -> Unit)? = null

    override fun load(path: String, contextLength: Int, onLoaded: (Result<Unit>) -> Unit) {
        loadCalled = true
        loadPath = path
        loadContextLength = contextLength
        if (autoCompleteLoad) {
            onLoaded(loadResult)
        }
    }

    override fun predict(prompt: String, params: GenerationParams) {
        predictCalled = true
        predictPrompt = prompt
        onPredict?.invoke()
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
    @Test fun `synchronous prediction events cannot arrive before collection`() {
        val engine = createEngine()
        engine.loadModel(File(tempDir, "early.gguf").apply { writeText("model") })
        fakeWrapper.onPredict = {
            assertTrue(engine.llmFlow.subscriptionCount.value > 0)
            engine.llmFlow.tryEmit(LlamaHelper.LLMEvent.Ongoing("early", 1))
            engine.llmFlow.tryEmit(LlamaHelper.LLMEvent.Done("early", 1, 1))
        }
        assertEquals("early", engine.generate("prompt").getOrThrow())
    }

    @Test fun `failed model load releases helper and preserves reason`() {
        val engine = createEngine()
        fakeWrapper.loadResult = Result.failure(IllegalStateException("unsupported model architecture"))
        val result = engine.loadModel(File(tempDir, "invalid.gguf").apply { writeText("bad") })
        assertEquals("unsupported model architecture", result.exceptionOrNull()?.message)
        assertTrue(fakeWrapper.abortCalled)
        assertTrue(fakeWrapper.releaseCalled)
        assertFalse(engine.isLoaded)
    }

    @Test fun `model load timeout releases unpublished helper`() {
        val engine = createEngine(loadTimeoutMs = 20)
        fakeWrapper.autoCompleteLoad = false
        val result = engine.loadModel(File(tempDir, "slow.gguf").apply { writeText("model") })
        assertTrue(result.exceptionOrNull()?.message.orEmpty().contains("timed out"))
        assertTrue(fakeWrapper.abortCalled)
        assertTrue(fakeWrapper.releaseCalled)
        assertFalse(engine.isLoaded)
    }

    @Test fun `concurrent callers receive only their own generation`() = runBlocking {
        val engine = createEngine(2000)
        engine.loadModel(File(tempDir, "concurrent.gguf").apply { writeText("model") })
        val first = async(Dispatchers.IO) { engine.generate("first") }
        withTimeout(1000) { engine.llmFlow.subscriptionCount.first { it > 0 } }
        val second = async(Dispatchers.IO) { engine.generate("second") }
        delay(50)
        assertEquals("first", fakeWrapper.predictPrompt)
        engine.llmFlow.emit(LlamaHelper.LLMEvent.Ongoing("one", 1))
        engine.llmFlow.emit(LlamaHelper.LLMEvent.Done("one", 1, 1))
        assertEquals("one", first.await().getOrThrow())
        withTimeout(1000) {
            while (fakeWrapper.predictPrompt != "second") delay(1)
            engine.llmFlow.subscriptionCount.first { it > 0 }
        }
        engine.llmFlow.emit(LlamaHelper.LLMEvent.Ongoing("two", 1))
        engine.llmFlow.emit(LlamaHelper.LLMEvent.Done("two", 1, 1))
        assertEquals("two", second.await().getOrThrow())
    }

    @Test fun `late events from timed out helper cannot reach the next caller`() = runBlocking {
        val engine = createEngine(200)
        engine.loadModel(File(tempDir, "late.gguf").apply { writeText("model") })
        val oldFlow = engine.llmFlow
        assertTrue(engine.generate("timeout").isFailure)
        val next = async(Dispatchers.IO) { engine.generate("next") }
        withTimeout(1000) {
            while (engine.llmFlow === oldFlow) delay(1)
            engine.llmFlow.subscriptionCount.first { it > 0 }
        }
        oldFlow.emit(LlamaHelper.LLMEvent.Ongoing("stale", 1))
        oldFlow.emit(LlamaHelper.LLMEvent.Done("stale", 1, 1))
        engine.llmFlow.emit(LlamaHelper.LLMEvent.Ongoing("fresh", 1))
        engine.llmFlow.emit(LlamaHelper.LLMEvent.Done("fresh", 1, 1))
        assertEquals("fresh", next.await().getOrThrow())
        assertTrue(fakeWrapper.releaseCalled)
    }

    @Test fun `timeout without output is a failure and stops native prediction`() {
        val engine = createEngine()
        val file = File(tempDir, "timeout.gguf").apply { writeText("model") }
        engine.loadModel(file)
        val result = engine.generate("hello")
        assertTrue(result.exceptionOrNull() is GenerationTimeoutException)
        assertEquals("", (result.exceptionOrNull() as GenerationTimeoutException).partialText)
        assertTrue(fakeWrapper.stopPredictionCalled)
    }

    @Test fun `partial output remains a timeout rather than successful completion`() = runBlocking {
        val engine = createEngine(200)
        engine.loadModel(File(tempDir, "partial.gguf").apply { writeText("model") })
        val result = async(Dispatchers.IO) { engine.generate("hello") }
        engine.llmFlow.subscriptionCount.first { it > 0 }
        engine.llmFlow.emit(LlamaHelper.LLMEvent.Ongoing("partial", 1))
        val failure = result.await().exceptionOrNull() as GenerationTimeoutException
        assertEquals("partial", failure.partialText)
        assertTrue(fakeWrapper.stopPredictionCalled)
    }

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

    private fun createEngine(timeoutMs: Long = 100L, loadTimeoutMs: Long = 100L): LlamaEngine {
        engine = LlamaEngine(
            context = mockContext,
            helperFactory = { _, _, _ -> fakeWrapper },
            logger = NoOpLogger,
            generationTimeoutMs = timeoutMs,
            loadTimeoutMs = loadTimeoutMs,
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
