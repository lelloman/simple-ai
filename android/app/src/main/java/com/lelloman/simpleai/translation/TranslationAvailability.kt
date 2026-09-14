package com.lelloman.simpleai.translation

/** ML Kit bundles English; only non-English packs are downloadable/removable. */
object TranslationAvailability {
    const val BUILT_IN = "en"
    fun downloaded(models: Set<String>) = models - BUILT_IN
    fun available(models: Set<String>) = downloaded(models) + BUILT_IN
}
