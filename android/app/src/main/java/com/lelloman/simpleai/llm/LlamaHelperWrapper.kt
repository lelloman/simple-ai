package com.lelloman.simpleai.llm

import android.content.ContentResolver
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.flow.MutableSharedFlow
import org.nehuatl.llamacpp.LlamaHelper

interface LlamaHelperWrapper {
    fun load(path: String, contextLength: Int, onLoaded: (Long) -> Unit)
    fun predict(prompt: String)
    fun stopPrediction()
    fun abort()
    fun release()
}

class RealLlamaHelperWrapper(
    contentResolver: ContentResolver,
    scope: CoroutineScope,
    sharedFlow: MutableSharedFlow<LlamaHelper.LLMEvent>
) : LlamaHelperWrapper {

    private val helper = LlamaHelper(
        contentResolver = contentResolver,
        scope = scope,
        sharedFlow = sharedFlow
    )

    override fun load(path: String, contextLength: Int, onLoaded: (Long) -> Unit) {
        helper.load(path, contextLength, onLoaded)
    }

    override fun predict(prompt: String) {
        helper.predict(prompt)
    }

    override fun stopPrediction() {
        helper.stopPrediction()
    }

    override fun abort() {
        helper.abort()
    }

    override fun release() {
        helper.release()
    }
}
