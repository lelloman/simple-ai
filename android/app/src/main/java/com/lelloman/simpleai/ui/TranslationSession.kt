package com.lelloman.simpleai.ui

import androidx.lifecycle.SavedStateHandle
import com.lelloman.simpleai.translation.TranslationResult
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.*

data class TranslationDraft(val text: String = "", val source: String = "auto", val target: String = "en")

class TranslationSession(
    private val scope: CoroutineScope,
    private val saved: SavedStateHandle,
    private val translate: suspend (TranslationDraft) -> Result<TranslationResult>
) {
    private val _state = MutableStateFlow(TranslationState(draft = TranslationDraft(
        saved["translation.text"] ?: "", saved["translation.source"] ?: "auto", saved["translation.target"] ?: "en"
    )))
    val state = _state.asStateFlow()
    private var revision = 0L
    private var job: Job? = null

    fun edit(draft: TranslationDraft) {
        if (draft == _state.value.draft) return
        revision++
        job?.cancel()
        saved["translation.text"] = draft.text
        saved["translation.source"] = draft.source
        saved["translation.target"] = draft.target
        _state.value = TranslationState(draft = draft)
    }

    fun submit() {
        val draft = _state.value.draft
        if (draft.text.isBlank()) return
        job?.cancel()
        val requestRevision = ++revision
        _state.value = TranslationState(draft = draft, isTranslating = true)
        job = scope.launch {
            val result = try { translate(draft) }
            catch (e: CancellationException) { throw e }
            catch (e: Exception) { Result.failure(e) }
            if (requestRevision != revision || draft != _state.value.draft) return@launch
            _state.value = result.fold(
                { TranslationState(draft = draft, translatedText = it.translatedText, detectedLanguage = it.detectedSourceLang) },
                { TranslationState(draft = draft, error = it.message ?: "Translation failed") }
            )
        }
    }
}
