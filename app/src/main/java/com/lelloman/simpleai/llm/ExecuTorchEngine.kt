package com.lelloman.simpleai.llm

import com.lelloman.simpleai.util.AndroidLogger
import com.lelloman.simpleai.util.Logger
import org.pytorch.executorch.extension.llm.LlmCallback
import org.pytorch.executorch.extension.llm.LlmModule
import java.io.File
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicReference

/**
 * ExecuTorch-based LLM engine for optimized on-device inference.
 * Uses Meta's ExecuTorch runtime with .pte model files.
 */
class ExecuTorchEngine(
    private val logger: Logger = AndroidLogger,
    private val generationTimeoutMs: Long = DEFAULT_GENERATION_TIMEOUT_MS
) : LLMEngine {

    companion object {
        private const val TAG = "ExecuTorchEngine"
        private const val DEFAULT_GENERATION_TIMEOUT_MS = 120_000L
        // This model was exported with seq_len=1024 (runtime limit, not architectural)
        // ~4 chars per token for Llama tokenizer
        private const val MAX_SEQ_LEN = 1024
        private const val CHARS_PER_TOKEN = 4
        private const val MAX_PROMPT_CHARS = (MAX_SEQ_LEN - 100) * CHARS_PER_TOKEN  // Leave room for output
    }

    private var llmModule: LlmModule? = null
    private var _modelInfo: ModelInfo? = null
    private var tokenizerPath: String? = null

    override val isLoaded: Boolean
        get() = _modelInfo != null && llmModule != null

    override val modelInfo: ModelInfo?
        get() = _modelInfo

    override fun loadModel(modelPath: File): Result<Unit> {
        return try {
            if (!modelPath.exists()) {
                return Result.failure(IllegalArgumentException("Model file does not exist: ${modelPath.absolutePath}"))
            }

            // Unload previous model if any
            unloadModel()

            val loadStartTime = System.currentTimeMillis()
            logger.i(TAG, "Loading ExecuTorch model from: ${modelPath.absolutePath}")

            // Find tokenizer - should be in same directory
            val modelDir = modelPath.parentFile
            // Try tokenizer.model first (Llama 3 format), then tokenizer.bin, then tokenizer.json
            val tokenizerCandidates = listOf("tokenizer.model", "tokenizer.bin", "tokenizer.json")
            val foundTokenizer = tokenizerCandidates
                .map { File(modelDir, it) }
                .firstOrNull { it.exists() }

            if (foundTokenizer == null) {
                return Result.failure(IllegalArgumentException("Tokenizer not found in ${modelDir?.absolutePath}"))
            }
            tokenizerPath = foundTokenizer.absolutePath

            logger.i(TAG, "Using tokenizer: $tokenizerPath")

            // Create LlmModule with model path, tokenizer path, and temperature
            val module = LlmModule(
                modelPath.absolutePath,
                tokenizerPath,
                0.7f // default temperature
            )

            // Load the model
            val loadResult = module.load()
            val loadElapsed = System.currentTimeMillis() - loadStartTime

            if (loadResult != 0) {
                logger.w(TAG, "Model load failed with code $loadResult after ${loadElapsed}ms")
                return Result.failure(RuntimeException("Failed to load model: error code $loadResult"))
            }

            llmModule = module
            _modelInfo = ModelInfo(
                name = modelPath.name,
                path = modelPath.absolutePath,
                sizeBytes = modelPath.length(),
                contextSize = MAX_SEQ_LEN // This model has a small context window
            )

            logger.i(TAG, "ExecuTorch model loaded successfully: ${modelPath.name} in ${loadElapsed}ms")
            Result.success(Unit)

        } catch (e: Exception) {
            logger.e(TAG, "Error loading ExecuTorch model", e)
            Result.failure(e)
        }
    }

    override fun unloadModel() {
        try {
            llmModule?.stop()
            llmModule?.resetNative()
        } catch (e: Exception) {
            logger.w(TAG, "Error during model unload", e)
        }
        llmModule = null
        _modelInfo = null
        tokenizerPath = null
    }

    override fun generate(prompt: String, params: GenerationParams): Result<String> {
        val module = llmModule
            ?: return Result.failure(IllegalStateException("Model not loaded"))

        return try {
            val startTime = System.currentTimeMillis()
            logger.i(TAG, "Generating response for prompt (${prompt.length} chars)")

            val responseBuilder = StringBuilder()
            val completionLatch = CountDownLatch(1)
            val errorRef = AtomicReference<String?>(null)
            var firstTokenTime: Long? = null
            var tokensPerSecond = 0f

            val callback = object : LlmCallback {
                override fun onResult(token: String) {
                    if (firstTokenTime == null) {
                        firstTokenTime = System.currentTimeMillis()
                        val ttft = firstTokenTime!! - startTime
                        logger.i(TAG, "First token after ${ttft}ms (prompt processing time)")
                    }
                    responseBuilder.append(token)
                }

                override fun onStats(tps: Float) {
                    tokensPerSecond = tps
                    logger.i(TAG, "Tokens per second: $tps")
                    completionLatch.countDown()
                }
            }

            // Start generation (non-blocking, uses callback)
            val genResult = module.generate(prompt, params.maxTokens, callback)

            if (genResult != 0) {
                return Result.failure(RuntimeException("Generation failed with code $genResult"))
            }

            // Wait for completion with timeout
            val completed = completionLatch.await(generationTimeoutMs, TimeUnit.MILLISECONDS)

            val elapsed = System.currentTimeMillis() - startTime
            val response = responseBuilder.toString()

            if (!completed) {
                module.stop()
                logger.w(TAG, "Generation timed out after ${elapsed}ms")
                return Result.failure(RuntimeException("Generation timed out"))
            }

            val genTime = if (firstTokenTime != null) System.currentTimeMillis() - firstTokenTime!! else 0
            logger.i(TAG, "Generation complete: ${response.length} chars in ${elapsed}ms (gen: ${genTime}ms, ${tokensPerSecond} tok/s)")

            Result.success(response)

        } catch (e: Exception) {
            logger.e(TAG, "Error during generation", e)
            Result.failure(e)
        }
    }

    fun release() {
        unloadModel()
    }
}
