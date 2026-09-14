package com.lelloman.simpleai.cloud

import okio.BufferedSource

/** Parse bounded SSE frames without buffering the whole response. */
internal object SseReader {
    private const val MAX_FRAME_BYTES = 128 * 1024L

    fun read(source: BufferedSource, onData: (String) -> Unit) {
        val data = StringBuilder()
        while (!source.exhausted()) {
            val line = source.readUtf8LineStrict(MAX_FRAME_BYTES)
            if (line.isEmpty()) {
                if (data.isNotEmpty()) {
                    val payload = data.toString().removeSuffix("\n")
                    onData(payload)
                    if (payload == "[DONE]") return
                    data.clear()
                }
            } else if (line.startsWith("data:")) {
                data.append(line.substring(5).removePrefix(" ")).append('\n')
                require(data.length <= MAX_FRAME_BYTES) { "Cloud stream frame too large" }
            }
        }
        throw CloudException("Cloud stream ended before [DONE]")
    }
}
