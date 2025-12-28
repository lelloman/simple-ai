package com.lelloman.simpleai.chat

import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.buildJsonArray
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.put
import java.util.UUID

/**
 * Formatter for Hermes-style tool calling format.
 *
 * Used by: Qwen 3, Hermes 2/3, Functionary, and many other open-source models.
 *
 * Format:
 * - Uses ChatML tokens: <|im_start|>, <|im_end|>
 * - Tools defined in <tools>...</tools> XML block
 * - Tool calls wrapped in <tool_call>...</tool_call>
 * - Tool results wrapped in <tool_response>...</tool_response>
 * - Optional thinking wrapped in <think>...</think>
 */
class HermesFormatter : ChatFormatter {

    override val format = ModelFormat.HERMES
    override val supportsTools = true

    private val json = Json {
        ignoreUnknownKeys = true
        prettyPrint = false
    }

    private val toolCallRegex = Regex(
        """<tool_call>\s*(.*?)\s*</tool_call>""",
        RegexOption.DOT_MATCHES_ALL
    )

    private val thinkRegex = Regex(
        """<think>.*?</think>""",
        RegexOption.DOT_MATCHES_ALL
    )

    override fun formatPrompt(
        messages: List<ChatMessage>,
        tools: List<ToolDefinition>,
        systemPrompt: String?
    ): String {
        val sb = StringBuilder()

        // System message with tools
        sb.append("<|im_start|>system\n")
        sb.append(systemPrompt ?: "You are a helpful assistant.")

        if (tools.isNotEmpty()) {
            sb.append("\n\n")
            sb.append("You have access to the following tools. Use them when appropriate.\n\n")
            sb.append("<tools>\n")
            sb.append(formatToolDefinitions(tools))
            sb.append("\n</tools>\n\n")
            sb.append("To call a tool, respond with:\n")
            sb.append("<tool_call>\n")
            sb.append("""{"name": "tool_name", "arguments": {"arg1": "value1"}}""")
            sb.append("\n</tool_call>\n\n")
            sb.append("You can call multiple tools by using multiple <tool_call> blocks.")
        }
        sb.append("<|im_end|>\n")

        // Conversation messages
        for (message in messages) {
            when (message.role) {
                "user" -> {
                    sb.append("<|im_start|>user\n")
                    sb.append(message.content)
                    sb.append("<|im_end|>\n")
                }

                "assistant" -> {
                    sb.append("<|im_start|>assistant\n")
                    // Include any tool calls the assistant made
                    message.toolCalls?.forEach { toolCall ->
                        sb.append("<tool_call>\n")
                        sb.append("""{"name": "${toolCall.name}", "arguments": ${toolCall.arguments}}""")
                        sb.append("\n</tool_call>\n")
                    }
                    if (message.content.isNotEmpty()) {
                        sb.append(message.content)
                    }
                    sb.append("<|im_end|>\n")
                }

                "tool" -> {
                    // Tool results come as a user message with tool_response wrapper
                    sb.append("<|im_start|>user\n")
                    sb.append("<tool_response>\n")
                    sb.append(message.content)
                    sb.append("\n</tool_response>")
                    sb.append("<|im_end|>\n")
                }

                "system" -> {
                    // Additional system messages (rare)
                    sb.append("<|im_start|>system\n")
                    sb.append(message.content)
                    sb.append("<|im_end|>\n")
                }
            }
        }

        // Start assistant turn and disable thinking for faster responses
        sb.append("<|im_start|>assistant\n")
        sb.append("<think>\n\n</think>\n\n")

        return sb.toString()
    }

    private fun formatToolDefinitions(tools: List<ToolDefinition>): String {
        val toolsArray = buildJsonArray {
            tools.forEach { tool ->
                add(buildJsonObject {
                    put("type", tool.type)
                    put("function", buildJsonObject {
                        put("name", tool.function.name)
                        put("description", tool.function.description)
                        put("parameters", tool.function.parameters)
                    })
                })
            }
        }
        return json.encodeToString(toolsArray)
    }

    override fun parseResponse(response: String): ChatResponse {
        val trimmed = response.trim()

        // Remove any trailing tokens that might have been generated
        val cleaned = trimmed
            .replace("<|im_end|>", "")
            .replace("<|im_start|>", "")
            .trim()

        // Extract all tool calls
        val toolCalls = mutableListOf<ToolCall>()
        val matches = toolCallRegex.findAll(cleaned)

        for (match in matches) {
            try {
                val jsonStr = match.groupValues[1].trim()
                val obj = json.parseToJsonElement(jsonStr).jsonObject
                val name = obj["name"]?.jsonPrimitive?.content ?: continue
                val arguments = obj["arguments"]?.jsonObject ?: JsonObject(emptyMap())

                toolCalls.add(
                    ToolCall(
                        id = "call_${UUID.randomUUID().toString().take(8)}",
                        name = name,
                        arguments = arguments
                    )
                )
            } catch (e: Exception) {
                // Skip malformed tool calls
            }
        }

        // Extract text content (everything outside tool_call and think tags)
        val textContent = cleaned
            .replace(toolCallRegex, "")
            .replace(thinkRegex, "")
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
}
