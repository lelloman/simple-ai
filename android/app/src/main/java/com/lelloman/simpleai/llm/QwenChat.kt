package com.lelloman.simpleai.llm

import kotlinx.serialization.json.*

/** Qwen3's plain-text, non-thinking ChatML template; tools/multimodal messages are unsupported. */
object QwenChat {
    fun format(messages: JsonArray, system: String?, tools: String?): String {
        require(tools == null || Json.parseToJsonElement(tools).jsonArray.isEmpty()) { "Local tool calling is not supported" }
        require(messages.isNotEmpty()) { "At least one message is required" }
        val turns = messages.map { item ->
            val message = item.jsonObject
            val role = message["role"]?.jsonPrimitive?.content
            require(role in setOf("system", "user", "assistant")) { "Unsupported local chat role" }
            require(message["tool_calls"] == null && message["reasoning_content"] == null) { "Tool and reasoning history are unsupported" }
            val content = message["content"] as? JsonPrimitive
            require(content != null && content.isString) { "Local chat requires text content" }
            role!! to if (role == "assistant") content.content.substringAfterLast("</think>").trimStart('\n') else content.content
        }
        require(turns.last().first == "user") { "Local chat must end with a user message" }
        return buildString {
            system?.let { append("<|im_start|>system\n$it<|im_end|>\n") }
            turns.forEach { (role, content) -> append("<|im_start|>$role\n$content<|im_end|>\n") }
            append("<|im_start|>assistant\n<think>\n\n</think>\n\n")
        }
    }
}
