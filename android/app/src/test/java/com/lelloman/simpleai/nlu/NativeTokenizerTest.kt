package com.lelloman.simpleai.nlu

import kotlinx.serialization.json.*
import org.junit.Assert.*
import org.junit.Test

class NativeTokenizerTest {
    private fun resource(name: String) = javaClass.getResource("/tokenizers/$name")!!.readText()

    @Test fun `Unigram and BPE match Python token IDs masks and original UTF16 offsets`() {
        for (name in listOf("unigram", "bpe")) {
            NativeTokenizer(resource("$name.json"), 64).use { tokenizer ->
                for (element in Json.parseToJsonElement(resource("$name-expected.json")).jsonArray) {
                    val expected = element.jsonObject
                    val text = expected.getValue("text").jsonPrimitive.content
                    val actual = tokenizer.encode(text)
                    assertArrayEquals("$name: $text", expected.getValue("ids").jsonArray.map { it.jsonPrimitive.long }.toLongArray(), actual.ids)
                    assertArrayEquals(expected.getValue("mask").jsonArray.map { it.jsonPrimitive.long }.toLongArray(), actual.mask)
                    assertEquals(expected.getValue("offsets").jsonArray.map { it.jsonArray.let { p -> p[0].jsonPrimitive.int to p[1].jsonPrimitive.int } }, actual.offsets)
                }
            }
        }
    }

    @Test fun `closed tokenizer cannot be reused and close is idempotent`() {
        val tokenizer = NativeTokenizer(resource("unigram.json"), 64)
        tokenizer.close()
        tokenizer.close()
        assertThrows(IllegalStateException::class.java) { tokenizer.encode("Hello") }
    }

    @Test fun `unsupported tokenizer JSON is explicitly rejected`() {
        assertThrows(IllegalArgumentException::class.java) { NativeTokenizer("{}", 64) }
    }
}
