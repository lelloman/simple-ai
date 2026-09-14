package com.lelloman.simpleai.nlu

import java.io.DataInputStream
import java.io.InputStream
import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlinx.serialization.json.*

internal object AdapterFiles {
    data class Heads(val intentHead: FloatArray, val intentBias: FloatArray, val slotHead: FloatArray, val slotBias: FloatArray, val numIntents: Int, val numSlots: Int)
    data class Config(val intents: List<String>, val slotLabels: List<String>, val maxLength: Int)

    fun heads(stream: InputStream): Heads {
        val input = DataInputStream(stream)
        fun int() = Integer.reverseBytes(input.readInt())
        fun floats(count: Int): FloatArray {
            val bytes = ByteArray(Math.multiplyExact(count, 4))
            input.readFully(bytes)
            val values = FloatArray(count)
            ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(values)
            require(values.all { it.isFinite() }) { "Non-finite head weights" }
            return values
        }
        fun matrix(): Pair<Int, FloatArray> {
            val rows = int()
            val columns = int()
            require(rows in 1..1024 && columns == 768) { "Heads require 1..1024 labels and XLM-R's 768 hidden features" }
            return rows to floats(rows * columns)
        }
        val (intents, intentWeights) = matrix()
        val (slots, slotWeights) = matrix()
        require(int() == intents) { "Intent bias dimension mismatch" }
        val intentBias = floats(intents)
        require(int() == slots) { "Slot bias dimension mismatch" }
        val slotBias = floats(slots)
        require(input.read() == -1) { "Unexpected trailing heads data" }
        return Heads(intentWeights, intentBias, slotWeights, slotBias, intents, slots)
    }

    fun text(input: InputStream, limit: Int): String {
        val output = ByteArrayOutputStream()
        val buffer = ByteArray(8192)
        while (true) {
            val count = input.read(buffer)
            if (count < 0) break
            require(output.size().toLong() + count <= limit) { "Adapter metadata exceeds size limit" }
            output.write(buffer, 0, count)
        }
        return output.toString(Charsets.UTF_8.name())
    }

    fun config(input: InputStream, heads: Heads): Config {
        val json = Json.parseToJsonElement(text(input, 1024 * 1024)).jsonObject
        fun labels(key: String, count: Int): List<String> {
            val labels = json.getValue(key).jsonArray.map {
                val value = it.jsonPrimitive
                require(value.isString && value.content.length in 1..256) { "Invalid $key label" }
                value.content
            }
            require(labels.size == count && labels.distinct().size == count) { "$key count does not match heads" }
            return labels
        }
        val maxLength = json["max_length"]?.jsonPrimitive?.int ?: 64
        require(maxLength in 2..512) { "max_length must be between 2 and 512" }
        val slots = labels("slot_labels", heads.numSlots)
        require(slots.all { it == "O" || it.startsWith("B-") && it.length > 2 || it.startsWith("I-") && it.length > 2 }) { "Invalid BIO slot label" }
        return Config(labels("intents", heads.numIntents), slots, maxLength)
    }
}
