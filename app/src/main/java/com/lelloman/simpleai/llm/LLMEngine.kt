package com.lelloman.simpleai.llm

import android.content.ContentResolver
import android.content.Context
import androidx.core.content.FileProvider
import com.lelloman.simpleai.util.AndroidLogger
import com.lelloman.simpleai.util.Logger
import org.nehuatl.llamacpp.LlamaHelper
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.channels.BufferOverflow
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withTimeoutOrNull
import java.io.File

typealias LlamaHelperFactory = (
    ContentResolver,
    CoroutineScope,
    MutableSharedFlow<LlamaHelper.LLMEvent>
) -> LlamaHelperWrapper

data class GenerationParams(
    val maxTokens: Int = 512,
    val temperature: Float = 0.7f,
    val topP: Float = 0.9f,
    val topK: Int = 40
)

data class ModelInfo(
    val name: String,
    val path: String,
    val sizeBytes: Long,
    val contextSize: Int
)

interface LLMEngine {
    val isLoaded: Boolean
    val modelInfo: ModelInfo?

    fun loadModel(modelPath: File): Result<Unit>
    fun unloadModel()
    fun generate(prompt: String, params: GenerationParams = GenerationParams()): Result<String>
}

/**
 * Real llama.cpp implementation using kotlinllamacpp library.
 */
class LlamaEngine(
    private val context: Context,
    private val helperFactory: LlamaHelperFactory = ::RealLlamaHelperWrapper,
    private val logger: Logger = AndroidLogger,
    private val generationTimeoutMs: Long = DEFAULT_GENERATION_TIMEOUT_MS,
    private val uriResolver: (File) -> String = { file ->
        FileProvider.getUriForFile(context, FILE_PROVIDER_AUTHORITY, file).toString()
    }
) : LLMEngine {

    companion object {
        private const val TAG = "LlamaEngine"
        private const val DEFAULT_CONTEXT_LENGTH = 4096
        private const val DEFAULT_GENERATION_TIMEOUT_MS = 120_000L // 2 minutes
        private const val FILE_PROVIDER_AUTHORITY = "com.lelloman.simpleai.fileprovider"
    }

    private val contentResolver: ContentResolver = context.contentResolver

    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)

    internal val llmFlow = MutableSharedFlow<LlamaHelper.LLMEvent>(
        replay = 0,
        extraBufferCapacity = 256,
        onBufferOverflow = BufferOverflow.DROP_OLDEST
    )

    private var llamaHelper: LlamaHelperWrapper? = null
    private var _modelInfo: ModelInfo? = null
    private var currentModelPath: String? = null

    override val isLoaded: Boolean
        get() = _modelInfo != null && llamaHelper != null

    override val modelInfo: ModelInfo?
        get() = _modelInfo

    override fun loadModel(modelPath: File): Result<Unit> {
        return try {
            if (!modelPath.exists()) {
                return Result.failure(IllegalArgumentException("Model file does not exist: ${modelPath.absolutePath}"))
            }

            // Unload previous model if any
            unloadModel()

            logger.i(TAG, "Loading model from: ${modelPath.absolutePath}")

            // Convert to content:// URI - required by the library
            val contentUriString = uriResolver(modelPath)
            logger.i(TAG, "Content URI: $contentUriString")

            val helper = helperFactory(contentResolver, scope, llmFlow)

            // Load model using content URI
            var loadSuccess = false
            var loadError: String? = null

            helper.load(
                path = contentUriString,
                contextLength = DEFAULT_CONTEXT_LENGTH
            ) { _ ->
                loadSuccess = true
            }

            // Wait for load to complete - the library loads asynchronously
            // Model loading can take several seconds for warmup
            val maxWaitMs = 30_000L
            val loadStartTime = System.currentTimeMillis()
            while (!loadSuccess && (System.currentTimeMillis() - loadStartTime) < maxWaitMs) {
                Thread.sleep(100)
            }
            val loadElapsed = System.currentTimeMillis() - loadStartTime

            if (!loadSuccess) {
                logger.w(TAG, "Model load failed after ${loadElapsed}ms")
                return Result.failure(RuntimeException("Failed to load model: ${loadError ?: "timeout"}"))
            }

            llamaHelper = helper
            currentModelPath = modelPath.absolutePath
            _modelInfo = ModelInfo(
                name = modelPath.name,
                path = modelPath.absolutePath,
                sizeBytes = modelPath.length(),
                contextSize = DEFAULT_CONTEXT_LENGTH
            )

            logger.i(TAG, "Model loaded successfully: ${modelPath.name} in ${loadElapsed}ms")
            Result.success(Unit)

        } catch (e: Exception) {
            logger.e(TAG, "Error loading model", e)
            Result.failure(e)
        }
    }

    override fun unloadModel() {
        try {
            llamaHelper?.let { helper ->
                helper.abort()
                helper.release()
            }
        } catch (e: Exception) {
            logger.w(TAG, "Error during model unload", e)
        }
        llamaHelper = null
        _modelInfo = null
        currentModelPath = null
    }

    override fun generate(prompt: String, params: GenerationParams): Result<String> {
        val helper = llamaHelper
            ?: return Result.failure(IllegalStateException("Model not loaded"))

        return try {
            val startTime = System.currentTimeMillis()
            logger.i(TAG, "Generating response for prompt (${prompt.length} chars)")

            val response = runBlocking {
                generateAsync(helper, prompt, params)
            }

            val elapsed = System.currentTimeMillis() - startTime
            if (response != null) {
                logger.i(TAG, "Generation complete: ${response.length} chars in ${elapsed}ms")
                Result.success(response)
            } else {
                logger.w(TAG, "Generation failed after ${elapsed}ms")
                Result.failure(RuntimeException("Generation timed out or failed"))
            }

        } catch (e: Exception) {
            logger.e(TAG, "Error during generation", e)
            Result.failure(e)
        }
    }

    private suspend fun generateAsync(
        helper: LlamaHelperWrapper,
        prompt: String,
        params: GenerationParams
    ): String? {
        val responseBuilder = StringBuilder()
        var hasError = false
        val startTime = System.currentTimeMillis()
        var firstTokenTime: Long? = null

        // Start prediction
        helper.predict(prompt)

        // Collect events with timeout - use first{} to exit on terminal events
        withTimeoutOrNull(generationTimeoutMs) {
            llmFlow.first { event ->
                when (event) {
                    is LlamaHelper.LLMEvent.Started -> {
                        logger.i(TAG, "Generation started after ${System.currentTimeMillis() - startTime}ms")
                        false // continue collecting
                    }
                    is LlamaHelper.LLMEvent.Loaded -> {
                        logger.i(TAG, "Model loaded in generation context after ${System.currentTimeMillis() - startTime}ms")
                        false // continue collecting
                    }
                    is LlamaHelper.LLMEvent.Ongoing -> {
                        if (firstTokenTime == null) {
                            firstTokenTime = System.currentTimeMillis()
                            val ttft = firstTokenTime!! - startTime
                            logger.i(TAG, "First token after ${ttft}ms (prompt processing time)")
                        }
                        responseBuilder.append(event.word)
                        // Check if we've reached max tokens (approximate by char count)
                        if (responseBuilder.length > params.maxTokens * 4) {
                            helper.stopPrediction()
                            true // stop collecting
                        } else {
                            false // continue collecting
                        }
                    }
                    is LlamaHelper.LLMEvent.Done -> {
                        val total = System.currentTimeMillis() - startTime
                        val genTime = if (firstTokenTime != null) System.currentTimeMillis() - firstTokenTime!! else 0
                        logger.i(TAG, "Generation done: total=${total}ms, generation=${genTime}ms, ${responseBuilder.length} chars")
                        true // stop collecting
                    }
                    is LlamaHelper.LLMEvent.Error -> {
                        logger.e(TAG, "Generation error after ${System.currentTimeMillis() - startTime}ms: ${event.message}")
                        hasError = true
                        true // stop collecting
                    }
                }
            }
        }

        return if (hasError) null else responseBuilder.toString()
    }

    fun release() {
        unloadModel()
        scope.cancel()
    }
}

/**
 * Stub implementation for testing without native library.
 */
class StubLLMEngine : LLMEngine {
    private var _modelInfo: ModelInfo? = null

    override val isLoaded: Boolean
        get() = _modelInfo != null

    override val modelInfo: ModelInfo?
        get() = _modelInfo

    override fun loadModel(modelPath: File): Result<Unit> {
        return if (modelPath.exists()) {
            _modelInfo = ModelInfo(
                name = modelPath.name,
                path = modelPath.absolutePath,
                sizeBytes = modelPath.length(),
                contextSize = 4096
            )
            Result.success(Unit)
        } else {
            Result.failure(IllegalArgumentException("Model file does not exist: ${modelPath.absolutePath}"))
        }
    }

    override fun unloadModel() {
        _modelInfo = null
    }

    override fun generate(prompt: String, params: GenerationParams): Result<String> {
        if (!isLoaded) {
            return Result.failure(IllegalStateException("Model not loaded"))
        }
        return Result.success(
            "[STUB] This is a test response.\n\nPrompt: \"$prompt\"\n" +
            "Params: maxTokens=${params.maxTokens}, temp=${params.temperature}"
        )
    }
}
