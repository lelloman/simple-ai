package com.lelloman.simpleai.nlu

/** Owns a replaceable native session independently of adapter/LoRA metadata. */
internal class NativeResourceSlot<T : AutoCloseable> : AutoCloseable {
    var value: T? = null
        set(next) {
            if (field === next) return
            val previous = field
            field = next
            previous?.close()
        }
    override fun close() { value = null }
}

/** Releases every acquired tensor/result even if a later allocation or close fails. */
internal class NativeResources : AutoCloseable {
    private val resources = mutableListOf<AutoCloseable>()
    fun <T : AutoCloseable> own(resource: T): T = resource.also { resources.add(it) }
    override fun close() {
        var failure: Exception? = null
        resources.asReversed().forEach {
            try { it.close() } catch (e: Exception) {
                if (failure == null) failure = e else failure!!.addSuppressed(e)
            }
        }
        resources.clear()
        failure?.let { throw it }
    }
}
