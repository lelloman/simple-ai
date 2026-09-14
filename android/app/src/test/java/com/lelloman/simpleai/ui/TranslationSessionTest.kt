package com.lelloman.simpleai.ui

import androidx.lifecycle.SavedStateHandle
import com.lelloman.simpleai.translation.TranslationResult
import kotlinx.coroutines.*
import kotlinx.coroutines.test.*
import org.junit.Assert.*
import org.junit.Test

@OptIn(ExperimentalCoroutinesApi::class)
class TranslationSessionTest {
    @Test fun draftRestoresWithoutAnUnrelatedResult() = runTest {
        val saved = SavedStateHandle()
        val draft = TranslationDraft("hello", "en", "it")
        val session = TranslationSession(backgroundScope, saved) { Result.success(TranslationResult("ciao", "en")) }
        session.edit(draft)
        session.submit()
        runCurrent()
        assertEquals("ciao", session.state.value.translatedText)
        val restored = TranslationSession(backgroundScope, saved) { error("not requested") }
        assertEquals(draft, restored.state.value.draft)
        assertNull(restored.state.value.translatedText)
        assertFalse(restored.state.value.isTranslating)
    }

    @Test fun editingAnyRequestFieldInvalidatesResult() = runTest {
        val session = TranslationSession(backgroundScope, SavedStateHandle()) { Result.success(TranslationResult("result", "en")) }
        val draft = TranslationDraft("hello", "en", "it")
        for (edited in listOf(draft.copy(text = "other"), draft.copy(source = "fr"), draft.copy(target = "de"))) {
            session.edit(draft)
            session.submit()
            runCurrent()
            assertNotNull(session.state.value.translatedText)
            session.edit(edited)
            assertNull(session.state.value.translatedText)
        }
    }

    @Test fun responseFromUncancellableOldRequestCannotReplaceNewDraft() = runTest {
        val gate = CompletableDeferred<Unit>()
        val session = TranslationSession(backgroundScope, SavedStateHandle()) {
            withContext(NonCancellable) { gate.await() }
            Result.success(TranslationResult("stale", "en"))
        }
        session.edit(TranslationDraft("old", "en", "it"))
        session.submit()
        runCurrent()
        session.edit(TranslationDraft("new", "en", "fr"))
        gate.complete(Unit)
        runCurrent()
        assertEquals("new", session.state.value.draft.text)
        assertNull(session.state.value.translatedText)
        assertFalse(session.state.value.isTranslating)
    }
}
