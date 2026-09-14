package com.lelloman.simpleai.access

/** Fail fast instead of occupying Binder threads waiting for other clients. */
internal class CallerBudget(private val now: () -> Long) {
    private data class Window(var start: Long, var count: Int = 0, var active: Boolean = false)
    private val callers = mutableMapOf<Int, Window>()
    @Synchronized fun acquire(uid: Int): AutoCloseable? {
        val time = now()
        callers.entries.removeAll { !it.value.active && time - it.value.start >= 60_000 }
        if (callers.size >= 256 && uid !in callers) return null
        val window = callers.getOrPut(uid) { Window(time) }
        if (window.active || window.count >= 30 || callers.values.count { it.active } >= 4) return null
        window.count++
        window.active = true
        return AutoCloseable { synchronized(this) { window.active = false } }
    }
}
