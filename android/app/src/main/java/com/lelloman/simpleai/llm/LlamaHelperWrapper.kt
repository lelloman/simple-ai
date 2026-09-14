package com.lelloman.simpleai.llm

import android.content.ContentResolver
import android.net.Uri
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.MutableSharedFlow
import org.nehuatl.llamacpp.LlamaHelper
import org.nehuatl.llamacpp.LlamaContext
import org.nehuatl.llamacpp.LlamaAndroid

interface LlamaHelperWrapper {
    fun load(path: String, contextLength: Int, onLoaded: (Result<Unit>) -> Unit)
    fun predict(prompt: String, params: GenerationParams)
    fun stopPrediction()
    fun abort()
    fun release()
}

internal fun GenerationParams.nativeOptions(prompt: String): Map<String, Any> = mapOf(
    "prompt" to prompt, "emit_partial_completion" to true,
    "temperature" to temperature.toDouble(), "top_p" to topP.toDouble(),
    "top_k" to topK, "n_predict" to maxTokens
)

/** Uses the lower-level API: LlamaHelper 0.2.0 ignores sampling and has a broken stop guard. */
class RealLlamaHelperWrapper(
    private val contentResolver: ContentResolver,
    private val scope: CoroutineScope,
    private val sharedFlow: MutableSharedFlow<LlamaHelper.LLMEvent>
) : LlamaHelperWrapper {
    // Initialize the dependency's ABI loader before handing any descriptor to JNI.
    private val loader = LlamaAndroid(contentResolver)
    private val lifecycle = Any()
    @Volatile private var native: LlamaContext? = null
    private var closed = false
    private var loadJob: Job? = null
    private var predictionJob: Job? = null
    private var text = StringBuilder()
    private var tokens = 0

    override fun load(path: String, contextLength: Int, onLoaded: (Result<Unit>) -> Unit) {
        loadJob = scope.launch {
            try {
                val descriptor = contentResolver.openFileDescriptor(Uri.parse(path), "r")
                    ?: error("Cannot open model")
                // initContextWithFd takes ownership and closes the passed descriptor.
                val fd = descriptor.detachFd()
                descriptor.close()
                val context = LlamaContext(System.identityHashCode(this@RealLlamaHelperWrapper), mapOf(
                    "model" to path, "model_fd" to fd, "n_ctx" to contextLength,
                    "use_mmap" to false, "use_mlock" to false
                ))
                check(context.context != 0L) { "Native model initialization failed" }
                synchronized(lifecycle) {
                    if (closed || !isActive) context.release()
                    else {
                        native = context
                        context.setTokenCallback { word ->
                            text.append(word)
                            sharedFlow.tryEmit(LlamaHelper.LLMEvent.Ongoing(word, ++tokens))
                        }
                        onLoaded(Result.success(Unit))
                    }
                }
            } catch (e: Exception) {
                onLoaded(Result.failure(e))
            } catch (e: LinkageError) {
                onLoaded(Result.failure(e))
            }
        }
    }

    override fun predict(prompt: String, params: GenerationParams) {
        val context = checkNotNull(native) { "Model not loaded" }
        text = StringBuilder()
        tokens = 0
        predictionJob = scope.launch {
            val started = System.currentTimeMillis()
            try {
                val result = context.completion(params.nativeOptions(prompt))
                sharedFlow.tryEmit(LlamaHelper.LLMEvent.Done(result["text"] as? String ?: text.toString(), tokens, System.currentTimeMillis() - started))
            } catch (e: Exception) {
                sharedFlow.tryEmit(LlamaHelper.LLMEvent.Error(e.message ?: "Generation failed"))
            }
        }
    }

    override fun stopPrediction() {
        predictionJob?.cancel()
        native?.stopCompletion()
        runBlocking { predictionJob?.join() }
    }

    override fun abort() {
        synchronized(lifecycle) { closed = true }
        loadJob?.cancel()
        stopPrediction()
    }

    override fun release() {
        abort()
        runBlocking { loadJob?.join() }
        synchronized(lifecycle) { native?.release(); native = null }
    }
}
