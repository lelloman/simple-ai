package com.lelloman.simpleai.nlu

import kotlinx.serialization.json.*

/** Uses the same Hugging Face tokenizer pipeline as training, including normalization and offsets. */
class NativeTokenizer(json: String, maxLength: Int) : AutoCloseable {
    companion object { init { System.loadLibrary("simpleai_tokenizer") } }
    private var handle = create(json, maxLength)
    private external fun create(json: String, maxLength: Int): Long
    private external fun encodeNative(handle: Long, text: String): String
    private external fun destroy(handle: Long)

    data class Encoding(val ids: LongArray, val mask: LongArray, val offsets: List<Pair<Int, Int>>)

    @Synchronized fun encode(text: String): Encoding {
        check(handle != 0L) { "Tokenizer is closed" }
        val data = Json.parseToJsonElement(encodeNative(handle, text)).jsonObject
        return Encoding(
            data.getValue("ids").jsonArray.map { it.jsonPrimitive.long }.toLongArray(),
            data.getValue("mask").jsonArray.map { it.jsonPrimitive.long }.toLongArray(),
            data.getValue("offsets").jsonArray.map { it.jsonArray.let { pair -> pair[0].jsonPrimitive.int to pair[1].jsonPrimitive.int } }
        )
    }

    @Synchronized override fun close() { if (handle != 0L) { destroy(handle); handle = 0L } }
}
