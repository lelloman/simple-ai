package com.lelloman.simpleai.llm

import org.junit.After
import org.junit.Assert.*
import org.junit.Before
import org.junit.Test
import java.io.File

class StubLLMEngineTest {

    private lateinit var tempDir: File
    private lateinit var engine: StubLLMEngine

    @Before
    fun setup() {
        tempDir = File(System.getProperty("java.io.tmpdir"), "test_stub_${System.currentTimeMillis()}")
        tempDir.mkdirs()
        engine = StubLLMEngine()
    }

    @After
    fun teardown() {
        tempDir.deleteRecursively()
    }

    @Test
    fun `isLoaded returns false initially`() {
        assertFalse(engine.isLoaded)
    }

    @Test
    fun `modelInfo is null initially`() {
        assertNull(engine.modelInfo)
    }

    @Test
    fun `loadModel fails when file does not exist`() {
        val nonExistentFile = File(tempDir, "nonexistent.gguf")

        val result = engine.loadModel(nonExistentFile)

        assertTrue(result.isFailure)
        assertTrue(result.exceptionOrNull() is IllegalArgumentException)
        assertFalse(engine.isLoaded)
    }

    @Test
    fun `loadModel succeeds when file exists`() {
        val modelFile = File(tempDir, "model.gguf")
        modelFile.writeText("model content")

        val result = engine.loadModel(modelFile)

        assertTrue(result.isSuccess)
        assertTrue(engine.isLoaded)
    }

    @Test
    fun `loadModel sets correct modelInfo`() {
        val modelFile = File(tempDir, "test_model.gguf")
        modelFile.writeBytes(ByteArray(2048))

        engine.loadModel(modelFile)

        val info = engine.modelInfo
        assertNotNull(info)
        assertEquals("test_model.gguf", info?.name)
        assertEquals(modelFile.absolutePath, info?.path)
        assertEquals(2048L, info?.sizeBytes)
        assertEquals(4096, info?.contextSize)
    }

    @Test
    fun `unloadModel clears state`() {
        val modelFile = File(tempDir, "model.gguf")
        modelFile.writeText("model")
        engine.loadModel(modelFile)
        assertTrue(engine.isLoaded)

        engine.unloadModel()

        assertFalse(engine.isLoaded)
        assertNull(engine.modelInfo)
    }

    @Test
    fun `generate fails when model not loaded`() {
        val result = engine.generate("test prompt")

        assertTrue(result.isFailure)
        assertTrue(result.exceptionOrNull() is IllegalStateException)
    }

    @Test
    fun `generate returns stub response when loaded`() {
        val modelFile = File(tempDir, "model.gguf")
        modelFile.writeText("model")
        engine.loadModel(modelFile)

        val result = engine.generate("Hello world")

        assertTrue(result.isSuccess)
        val response = result.getOrNull()
        assertNotNull(response)
        assertTrue(response!!.contains("[STUB]"))
        assertTrue(response.contains("Hello world"))
    }

    @Test
    fun `generate includes params in response`() {
        val modelFile = File(tempDir, "model.gguf")
        modelFile.writeText("model")
        engine.loadModel(modelFile)

        val params = GenerationParams(maxTokens = 256, temperature = 0.5f)
        val result = engine.generate("test", params)

        val response = result.getOrNull()!!
        assertTrue(response.contains("maxTokens=256"))
        assertTrue(response.contains("temp=0.5"))
    }

    @Test
    fun `generate uses default params when not specified`() {
        val modelFile = File(tempDir, "model.gguf")
        modelFile.writeText("model")
        engine.loadModel(modelFile)

        val result = engine.generate("test")

        val response = result.getOrNull()!!
        assertTrue(response.contains("maxTokens=512"))
        assertTrue(response.contains("temp=0.7"))
    }

    @Test
    fun `multiple load and unload cycles work correctly`() {
        val modelFile = File(tempDir, "model.gguf")
        modelFile.writeText("model")

        // First cycle
        engine.loadModel(modelFile)
        assertTrue(engine.isLoaded)
        engine.unloadModel()
        assertFalse(engine.isLoaded)

        // Second cycle
        engine.loadModel(modelFile)
        assertTrue(engine.isLoaded)
        engine.unloadModel()
        assertFalse(engine.isLoaded)
    }

    @Test
    fun `loading different models updates modelInfo`() {
        val model1 = File(tempDir, "model1.gguf")
        val model2 = File(tempDir, "model2.gguf")
        val content1 = "model1"
        val content2 = "model2 with more content"
        model1.writeText(content1)
        model2.writeText(content2)

        engine.loadModel(model1)
        assertEquals("model1.gguf", engine.modelInfo?.name)
        assertEquals(content1.length.toLong(), engine.modelInfo?.sizeBytes)

        engine.loadModel(model2)
        assertEquals("model2.gguf", engine.modelInfo?.name)
        assertEquals(content2.length.toLong(), engine.modelInfo?.sizeBytes)
    }

    @Test
    fun `StubLLMEngine implements LLMEngine interface`() {
        val llmEngine: LLMEngine = StubLLMEngine()

        // Verify interface methods are accessible
        assertFalse(llmEngine.isLoaded)
        assertNull(llmEngine.modelInfo)
    }
}
