package com.lelloman.simpleai.download

import kotlinx.coroutines.*
import kotlinx.coroutines.flow.*

class KeyedDownloads(private val scope: CoroutineScope, private val download: suspend (String) -> Result<Unit>) {
    private val _active = MutableStateFlow<Set<String>>(emptySet())
    val active = _active.asStateFlow()
    private val _errors = MutableStateFlow<Map<String, String>>(emptyMap())
    val errors = _errors.asStateFlow()

    @Synchronized fun start(key: String) {
        if (key in _active.value) return
        _active.update { it + key }
        clearError(key)
        scope.launch {
            try {
                download(key).exceptionOrNull()?.let { error -> _errors.update { it + (key to (error.message ?: "Download failed")) } }
            } catch (e: CancellationException) { throw e }
            catch (e: Exception) { _errors.update { it + (key to (e.message ?: "Download failed")) } }
            finally { _active.update { it - key } }
        }
    }

    fun clearError(key: String) { _errors.update { it - key } }
}
