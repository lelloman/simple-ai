package com.lelloman.simpleai.llm

import android.content.ContentResolver
import android.content.Context
import android.net.Uri
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
    private val generationTimeoutMs: Long = DEFAULT_GENERATION_TIMEOUT_MS
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

            // Convert file path to content:// URI using FileProvider
            val contentUri: Uri = FileProvider.getUriForFile(context, FILE_PROVIDER_AUTHORITY, modelPath)
            logger.i(TAG, "Content URI: $contentUri")

            // Verify the file is accessible via ContentResolver before passing to native
            try {
                contentResolver.openFileDescriptor(contentUri, "r")?.use { fd ->
                    logger.i(TAG, "File descriptor obtained successfully, fd=${fd.fd}")
                } ?: throw IllegalStateException("Failed to open file descriptor")
            } catch (e: Exception) {
                logger.e(TAG, "Failed to access file via ContentResolver", e)
                return Result.failure(e)
            }

            val helper = helperFactory(contentResolver, scope, llmFlow)

            // Load model using content URI
            var loadSuccess = false
            var loadError: String? = null

            helper.load(
                path = contentUri.toString(),
                contextLength = DEFAULT_CONTEXT_LENGTH
            ) { _ ->
                loadSuccess = true
            }

            // Wait for load to complete - the library loads asynchronously
            // Model loading can take several seconds for warmup
            val maxWaitMs = 30_000L
            val startTime = System.currentTimeMillis()
            while (!loadSuccess && (System.currentTimeMillis() - startTime) < maxWaitMs) {
                Thread.sleep(100)
            }

            if (!loadSuccess) {
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

            logger.i(TAG, "Model loaded successfully: ${modelPath.name}")
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
            logger.d(TAG, "Generating response for prompt (${prompt.length} chars)")

            val response = runBlocking {
                generateAsync(helper, prompt, params)
            }

            if (response != null) {
                logger.d(TAG, "Generation complete: ${response.length} chars")
                Result.success(response)
            } else {
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
        var isComplete = false
        var hasError = false

        // Start prediction
        helper.predict(prompt)

        // Collect events with timeout
        withTimeoutOrNull(generationTimeoutMs) {
            llmFlow.collect { event ->
                when (event) {
                    is LlamaHelper.LLMEvent.Started -> {
                        logger.d(TAG, "Generation started")
                    }
                    is LlamaHelper.LLMEvent.Loaded -> {
                        logger.d(TAG, "Model loaded in generation context")
                    }
                    is LlamaHelper.LLMEvent.Ongoing -> {
                        responseBuilder.append(event.word)
                        // Check if we've reached max tokens (approximate by char count)
                        if (responseBuilder.length > params.maxTokens * 4) {
                            helper.stopPrediction()
                            isComplete = true
                            return@collect
                        }
                    }
                    is LlamaHelper.LLMEvent.Done -> {
                        logger.d(TAG, "Generation done")
                        isComplete = true
                        return@collect
                    }
                    is LlamaHelper.LLMEvent.Error -> {
                        logger.e(TAG, "Generation error: ${event.message}")
                        hasError = true
                        return@collect
                    }
                }
            }
        }

        // Stop prediction if still running
        try {
            helper.stopPrediction()
        } catch (_: Exception) {}

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
