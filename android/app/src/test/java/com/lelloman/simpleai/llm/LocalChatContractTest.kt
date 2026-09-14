package com.lelloman.simpleai.llm

import kotlinx.serialization.json.*
import org.junit.Assert.*
import org.junit.Test

class LocalChatContractTest {
    @Test fun samplingAndTokenLimitReachNativeUsingExpectedTypes() {
        val params = GenerationParams(17, 0.25f, 0.75f, 9).nativeOptions("prompt")
        assertEquals(17, params["n_predict"])
        assertEquals(0.25, params["temperature"])
        assertEquals(0.75, params["top_p"])
        assertEquals(9, params["top_k"])
        assertEquals(true, params["emit_partial_completion"])
    }

    @Test fun plainTextChatUsesQwenNonThinkingTemplate() {
        val messages = Json.parseToJsonElement("""[{"role":"user","content":"Hello"},{"role":"assistant","content":"Hi"},{"role":"user","content":"Again"}]""").jsonArray
        assertEquals("<|im_start|>system\nBe brief<|im_end|>\n<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\nHi<|im_end|>\n<|im_start|>user\nAgain<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", QwenChat.format(messages, "Be brief", null))
    }

    @Test fun unsupportedToolsAndMessageShapesAreRejected() {
        val messages = Json.parseToJsonElement("""[{"role":"user","content":"Hello"}]""").jsonArray
        assertTrue(runCatching { QwenChat.format(messages, null, "[{}]") }.isFailure)
        for (invalid in listOf("[]", "[{}]", "[{\"role\":\"tool\",\"content\":\"x\"}]", "[{\"role\":\"user\",\"content\":[]}]")) {
            assertTrue(runCatching { QwenChat.format(Json.parseToJsonElement(invalid).jsonArray, null, null) }.isFailure)
        }
    }
}
