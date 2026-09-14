package com.lelloman.simpleai.api

import kotlinx.serialization.json.*

internal object RequestValidation {
    fun text(value: String, name: String, max: Int = 32768) {
        require(value.isNotBlank() && value.length <= max) { "$name must contain 1–$max characters" }
    }
    fun generation(prompt: String, tokens: Int, temperature: Float) {
        text(prompt, "prompt", 8192)
        require(tokens in 1..2048) { "maxTokens must be between 1 and 2048" }
        require(temperature.isFinite() && temperature in 0f..2f) { "temperature must be finite and between 0 and 2" }
    }
    fun messages(source: String): JsonArray {
        text(source, "messagesJson", 65536)
        val messages = Json.parseToJsonElement(source) as? JsonArray ?: throw IllegalArgumentException("messages must be an array")
        require(messages.size in 1..128) { "messages must contain 1–128 entries" }
        messages.forEach { entry ->
            val message = entry as? JsonObject ?: throw IllegalArgumentException("Each message must be an object")
            val role = message["role"] as? JsonPrimitive
            require(role?.isString == true && role.content in setOf("system", "user", "assistant", "tool", "developer")) { "Invalid message role" }
            val content = message["content"]
            require(content == null || content is JsonNull || content is JsonArray || content is JsonPrimitive && content.isString) { "Invalid message content" }
        }
        return messages
    }
}
