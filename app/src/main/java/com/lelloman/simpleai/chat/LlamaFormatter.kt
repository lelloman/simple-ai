package com.lelloman.simpleai.chat

import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.buildJsonArray
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.put
import java.util.UUID

/**
 * Formatter for Llama 3.1/3.2/3.3 native tool calling format.
 *
 * Format differences from Hermes:
 * - Uses special header tokens: <|start_header_id|>role<|end_header_id|>
 * - Tool calls use pythonic format: [func_name(param="value")]
 * - Tool results use "ipython" role
 * - Uses <|eot_id|> as end token
 *
 * Reference: https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/text_prompt_format.md
 */
class LlamaFormatter : ChatFormatter {

    override val format = ModelFormat.LLAMA
    override val supportsTools = true

    private val json = Json {
        ignoreUnknownKeys = true
        prettyPrint = false
    }

    // Matches pythonic tool calls: [func(param="value", param2=123)]
    private val toolCallRegex = Regex(
        """\[(\w+)\((.*?)\)\]""",
        setOf(RegexOption.DOT_MATCHES_ALL)
    )

    // Matches individual key=value pairs
    private val paramRegex = Regex(
        """(\w+)\s*=\s*(?:"([^"]*)"|'([^']*)'|(\d+(?:\.\d+)?)|(\w+))"""
    )

    override fun formatPrompt(
        messages: List<ChatMessage>,
        tools: List<ToolDefinition>,
        systemPrompt: String?
    ): String {
        val sb = StringBuilder()

        sb.append("<|begin_of_text|>")

        // System message with tools
        sb.append("<|start_header_id|>system<|end_header_id|>\n\n")

        if (tools.isNotEmpty()) {
            sb.append("Environment: ipython\n")
            sb.append("Tools: ")
            sb.append(tools.joinToString(", ") { it.function.name })
            sb.append("\n\n")
            sb.append("You have access to the following functions. ")
            sb.append("To call a function, respond with: [function_name(param1=\"value1\", param2=\"value2\")]\n\n")
            sb.append(formatToolDefinitions(tools))
            sb.append("\n\n")
        }

        sb.append(systemPrompt ?: "You are a helpful assistant.")
        sb.append("<|eot_id|>")

        // Conversation messages
        for (message in messages) {
            when (message.role) {
                "user" -> {
                    sb.append("<|start_header_id|>user<|end_header_id|>\n\n")
                    sb.append(message.content)
                    sb.append("<|eot_id|>")
                }

                "assistant" -> {
                    sb.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
                    // Include any tool calls the assistant made in pythonic format
                    message.toolCalls?.forEach { toolCall ->
                        sb.append("[${toolCall.name}(")
                        sb.append(formatArguments(toolCall.arguments))
                        sb.append(")]")
                    }
                    if (message.content.isNotEmpty()) {
                        sb.append(message.content)
                    }
                    sb.append("<|eot_id|>")
                }

                "tool" -> {
                    // Tool results use ipython role
                    sb.append("<|start_header_id|>ipython<|end_header_id|>\n\n")
                    sb.append(message.content)
                    sb.append("<|eot_id|>")
                }

                "system" -> {
                    sb.append("<|start_header_id|>system<|end_header_id|>\n\n")
                    sb.append(message.content)
                    sb.append("<|eot_id|>")
                }
            }
        }

        // Start assistant turn
        sb.append("<|start_header_id|>assistant<|end_header_id|>\n\n")

        return sb.toString()
    }

    private fun formatToolDefinitions(tools: List<ToolDefinition>): String {
        val toolsArray = buildJsonArray {
            tools.forEach { tool ->
                add(buildJsonObject {
                    put("name", tool.function.name)
                    put("description", tool.function.description)
                    put("parameters", buildJsonObject {
                        put("type", "dict")
                        // Copy properties from the original schema
                        val params = tool.function.parameters
                        params["properties"]?.let { put("properties", it) }
                        params["required"]?.let { put("required", it) }
                    })
                })
            }
        }
        return json.encodeToString(toolsArray)
    }

    private fun formatArguments(arguments: JsonObject): String {
        return arguments.entries.joinToString(", ") { (key, value) ->
            val valueStr = when {
                value.jsonPrimitive.isString -> "\"${value.jsonPrimitive.content}\""
                else -> value.jsonPrimitive.content
            }
            "$key=$valueStr"
        }
    }

    override fun parseResponse(response: String): ChatResponse {
        val trimmed = response.trim()

        // Remove any trailing tokens
        val cleaned = trimmed
            .replace("<|eot_id|>", "")
            .replace("<|eom_id|>", "")
            .trim()

        // Extract tool calls in pythonic format
        val toolCalls = mutableListOf<ToolCall>()
        val matches = toolCallRegex.findAll(cleaned)

        for (match in matches) {
            try {
                val funcName = match.groupValues[1]
                val paramsStr = match.groupValues[2]
                val arguments = parseArguments(paramsStr)

                toolCalls.add(
                    ToolCall(
                        id = "call_${UUID.randomUUID().toString().take(8)}",
                        name = funcName,
                        arguments = arguments
                    )
                )
            } catch (e: Exception) {
                // Skip malformed tool calls
            }
        }

        // Extract text content (everything outside tool calls)
        val textContent = cleaned
            .replace(toolCallRegex, "")
            .trim()

        return when {
            toolCalls.isNotEmpty() && textContent.isNotEmpty() ->
                ChatResponse.Mixed(textContent, toolCalls)

            toolCalls.isNotEmpty() ->
                ChatResponse.ToolCalls(toolCalls)

            else ->
                ChatResponse.Text(textContent.ifEmpty { cleaned })
        }
    }

    private fun parseArguments(paramsStr: String): JsonObject {
        val params = mutableMapOf<String, Any>()

        paramRegex.findAll(paramsStr).forEach { match ->
            val key = match.groupValues[1]
            val value = when {
                match.groupValues[2].isNotEmpty() -> match.groupValues[2] // double-quoted string
                match.groupValues[3].isNotEmpty() -> match.groupValues[3] // single-quoted string
                match.groupValues[4].isNotEmpty() -> {
                    // number
                    val numStr = match.groupValues[4]
                    if (numStr.contains(".")) numStr.toDouble() else numStr.toLong()
                }
                match.groupValues[5].isNotEmpty() -> {
                    // bare word (boolean or identifier)
                    when (match.groupValues[5].lowercase()) {
                        "true" -> true
                        "false" -> false
                        else -> match.groupValues[5]
                    }
                }
                else -> ""
            }
            params[key] = value
        }

        return buildJsonObject {
            params.forEach { (key, value) ->
                when (value) {
                    is String -> put(key, value)
                    is Long -> put(key, value)
                    is Double -> put(key, value)
                    is Boolean -> put(key, value)
                }
            }
        }
    }
}
