package com.lelloman.simpleai.llm

import android.content.ContentResolver
import android.content.Context
import androidx.core.content.FileProvider
import com.lelloman.simpleai.util.AndroidLogger
import com.lelloman.simpleai.util.Logger
import org.nehuatl.llamacpp.LlamaHelper
import kotlinx.coroutines.async
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.CoroutineStart
import kotlinx.coroutines.CompletableDeferred
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

class GenerationTimeoutException(val partialText: String) : RuntimeException("Generation timed out")

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
    private val loadTimeoutMs: Long = 30_000L,
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

    private fun newEventFlow() = MutableSharedFlow<LlamaHelper.LLMEvent>(
        replay = 0,
        extraBufferCapacity = 256,
        onBufferOverflow = BufferOverflow.DROP_OLDEST
    )

    @Volatile internal var llmFlow = newEventFlow()
        private set
    private var resetBeforeGeneration = false

    @Volatile private var llamaHelper: LlamaHelperWrapper? = null
    @Volatile private var _modelInfo: ModelInfo? = null
    private var currentModelPath: String? = null

    override val isLoaded: Boolean
        get() = _modelInfo != null && llamaHelper != null

    override val modelInfo: ModelInfo?
        get() = _modelInfo

    @Synchronized override fun loadModel(modelPath: File): Result<Unit> {
        var pendingHelper: LlamaHelperWrapper? = null
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

            llmFlow = newEventFlow()
            val helper = helperFactory(contentResolver, scope, llmFlow).also { pendingHelper = it }

            val loadStartTime = System.currentTimeMillis()
            val loaded = CompletableDeferred<Result<Unit>>()
            helper.load(contentUriString, DEFAULT_CONTEXT_LENGTH) { loaded.complete(it) }
            val outcome = runBlocking { withTimeoutOrNull(loadTimeoutMs) { loaded.await() } }
                ?: throw IllegalStateException("Model loading timed out after ${loadTimeoutMs}ms")
            outcome.getOrThrow()
            val loadElapsed = System.currentTimeMillis() - loadStartTime

            llamaHelper = helper
            pendingHelper = null
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
        } catch (e: LinkageError) {
            Result.failure(IllegalStateException("Native inference is unavailable on this device", e))
        } finally {
            pendingHelper?.let { helper ->
                try { helper.abort() } finally { helper.release() }
            }
        }
    }

    @Synchronized override fun unloadModel() {
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
        resetBeforeGeneration = false
    }

    @Synchronized override fun generate(prompt: String, params: GenerationParams): Result<String> {
        // A stopped request has no tagged terminal acknowledgement. Recreate the
        // helper with a separate stream before another caller can start.
        if (resetBeforeGeneration) {
            val path = currentModelPath ?: return Result.failure(IllegalStateException("Model not loaded"))
            loadModel(File(path)).exceptionOrNull()?.let { return Result.failure(it) }
        }
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
            resetBeforeGeneration = true
            Result.failure(e)
        }
    }

    private suspend fun generateAsync(
        helper: LlamaHelperWrapper,
        prompt: String,
        params: GenerationParams
    ): String? = coroutineScope {
        val responseBuilder = StringBuilder()
        var hasError = false
        val startTime = System.currentTimeMillis()
        var firstTokenTime: Long? = null

        // UNDISTPATCHED enters first() and registers before predict can emit.
        val collector = async(start = CoroutineStart.UNDISPATCHED) { withTimeoutOrNull(generationTimeoutMs) {
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
                        false // Native n_predict enforces the token limit.
                    }
                    is LlamaHelper.LLMEvent.Done -> {
                        responseBuilder.clear().append(event.fullText)
                        val total = System.currentTimeMillis() - startTime
                        val genTime = if (firstTokenTime != null) System.currentTimeMillis() - firstTokenTime!! else 0
                        logger.i(TAG, "Generation done: total=${total}ms, generation=${genTime}ms, ${responseBuilder.length} chars")
                        true // stop collecting
                    }
                    is LlamaHelper.LLMEvent.Error -> {
                        logger.e(TAG, "Generation error after ${System.currentTimeMillis() - startTime}ms: ${event.message}")
                        hasError = true
                        resetBeforeGeneration = true
                        true // stop collecting
                    }
                }
            }
        } }
        val terminal = try {
            helper.predict(prompt, params)
            collector.await()
        } finally {
            collector.cancel()
            // Stop native work on timeout, errors and coroutine cancellation too.
            helper.stopPrediction()
        }

        if (terminal == null) throw GenerationTimeoutException(responseBuilder.toString())

        if (hasError) null else responseBuilder.toString()
    }

    @Synchronized fun release() {
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
