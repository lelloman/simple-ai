package com.lelloman.simpleai.cloud

import android.util.Log
import com.lelloman.simpleai.BuildConfig
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlinx.serialization.Serializable
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import java.util.concurrent.TimeUnit

/**
 * Client for proxying chat requests to our cloud LLM endpoint.
 *
 * The endpoint is OpenAI-compatible, expecting:
 * - POST /v1/chat/completions
 * - Authorization: Bearer <token>
 * - Body: {"model": "...", "messages": [...], "tools": [...]}
 */
class CloudLLMClient {

    companion object {
        private const val TAG = "CloudLLMClient"
        private const val TIMEOUT_SECONDS = 60L
        private val JSON_MEDIA_TYPE = "application/json".toMediaType()
    }

    private val json = Json {
        ignoreUnknownKeys = true
        encodeDefaults = true
    }

    private val httpClient = OkHttpClient.Builder()
        .connectTimeout(TIMEOUT_SECONDS, TimeUnit.SECONDS)
        .readTimeout(TIMEOUT_SECONDS, TimeUnit.SECONDS)
        .writeTimeout(TIMEOUT_SECONDS, TimeUnit.SECONDS)
        .build()

    private val endpoint: String
        get() = BuildConfig.CLOUD_LLM_ENDPOINT

    /**
     * Send a chat completion request to the cloud endpoint.
     *
     * @param messages JSON array of chat messages
     * @param tools JSON array of tool definitions (optional)
     * @param systemPrompt System prompt to prepend (optional)
     * @param authToken Client's auth token for the cloud service
     * @return ChatResponse with assistant message and optional tool calls
     */
    suspend fun chat(
        messages: JsonArray,
        tools: JsonArray?,
        systemPrompt: String?,
        authToken: String
    ): Result<ChatResponse> = withContext(Dispatchers.IO) {
        try {
            // Build messages array with optional system prompt
            val fullMessages = buildMessages(messages, systemPrompt)

            // Build request body
            val requestBody = buildRequestBody(fullMessages, tools)
            val requestJson = json.encodeToString(requestBody)

            Log.d(TAG, "Sending request to $endpoint/v1/chat/completions")

            val request = Request.Builder()
                .url("$endpoint/v1/chat/completions")
                .header("Authorization", "Bearer $authToken")
                .header("Content-Type", "application/json")
                .post(requestJson.toRequestBody(JSON_MEDIA_TYPE))
                .build()

            val response = httpClient.newCall(request).execute()

            if (!response.isSuccessful) {
                val errorBody = response.body?.string() ?: "No error body"
                Log.e(TAG, "Cloud request failed: ${response.code} - $errorBody")

                return@withContext when (response.code) {
                    401, 403 -> Result.failure(CloudAuthException("Authentication failed: ${response.code}"))
                    429 -> Result.failure(CloudRateLimitException("Rate limited"))
                    500, 502, 503, 504 -> Result.failure(CloudUnavailableException("Server error: ${response.code}"))
                    else -> Result.failure(CloudException("Request failed: ${response.code} - $errorBody"))
                }
            }

            val responseBody = response.body?.string()
                ?: return@withContext Result.failure(CloudException("Empty response body"))

            Log.d(TAG, "Response body: $responseBody")
            val chatResponse = parseResponse(responseBody)
            Log.d(TAG, "Parsed response: role=${chatResponse.role}, content=${chatResponse.content}, toolCalls=${chatResponse.toolCalls?.size ?: 0}")
            Result.success(chatResponse)

        } catch (e: java.net.UnknownHostException) {
            Log.e(TAG, "Network error", e)
            Result.failure(CloudUnavailableException("Network unavailable: ${e.message}"))
        } catch (e: java.net.SocketTimeoutException) {
            Log.e(TAG, "Timeout", e)
            Result.failure(CloudUnavailableException("Request timed out"))
        } catch (e: Exception) {
            Log.e(TAG, "Unexpected error", e)
            Result.failure(CloudException("Unexpected error: ${e.message}"))
        }
    }

    private fun buildMessages(messages: JsonArray, systemPrompt: String?): JsonArray {
        if (systemPrompt == null) return messages

        val systemMessage = JsonObject(mapOf(
            "role" to kotlinx.serialization.json.JsonPrimitive("system"),
            "content" to kotlinx.serialization.json.JsonPrimitive(systemPrompt)
        ))

        return JsonArray(listOf(systemMessage) + messages)
    }

    private fun buildRequestBody(messages: JsonArray, tools: JsonArray?): JsonObject {
        val fields = mutableMapOf<String, JsonElement>(
            "messages" to messages
        )

        if (tools != null && tools.isNotEmpty()) {
            fields["tools"] = tools
        }

        return JsonObject(fields)
    }

    private fun parseResponse(responseBody: String): ChatResponse {
        val responseJson = json.parseToJsonElement(responseBody).jsonObject

        val choices = responseJson["choices"]?.jsonArray
            ?: throw CloudException("No choices in response")

        if (choices.isEmpty()) {
            throw CloudException("Empty choices array")
        }

        val firstChoice = choices[0].jsonObject
        val message = firstChoice["message"]?.jsonObject
            ?: throw CloudException("No message in choice")

        val roleElement = message["role"]
        val role = if (roleElement != null && roleElement !is kotlinx.serialization.json.JsonNull) {
            roleElement.jsonPrimitive.content
        } else {
            "assistant"
        }
        val contentElement = message["content"]
        val content = if (contentElement != null && contentElement !is kotlinx.serialization.json.JsonNull) {
            contentElement.jsonPrimitive.content
        } else {
            null
        }

        // Parse tool calls if present (handle both absent and explicit null)
        val toolCallsElement = message["tool_calls"]
        val toolCalls = if (toolCallsElement != null && toolCallsElement !is kotlinx.serialization.json.JsonNull) {
            toolCallsElement.jsonArray.map { toolCallElement ->
            val toolCall = toolCallElement.jsonObject
            val id = toolCall["id"]?.jsonPrimitive?.content ?: ""
            val type = toolCall["type"]?.jsonPrimitive?.content ?: "function"
            val function = toolCall["function"]?.jsonObject

            ToolCall(
                id = id,
                type = type,
                function = ToolFunction(
                    name = function?.get("name")?.jsonPrimitive?.content ?: "",
                    arguments = function?.get("arguments")?.jsonPrimitive?.content ?: "{}"
                )
            )
        }
        } else {
            null
        }

        // Parse usage if present
        val usage = responseJson["usage"]?.jsonObject?.let { usageObj ->
            Usage(
                promptTokens = usageObj["prompt_tokens"]?.jsonPrimitive?.content?.toIntOrNull() ?: 0,
                completionTokens = usageObj["completion_tokens"]?.jsonPrimitive?.content?.toIntOrNull() ?: 0,
                totalTokens = usageObj["total_tokens"]?.jsonPrimitive?.content?.toIntOrNull() ?: 0
            )
        }

        val finishReason = firstChoice["finish_reason"]?.jsonPrimitive?.content

        return ChatResponse(
            role = role,
            content = content,
            toolCalls = toolCalls,
            finishReason = finishReason,
            usage = usage
        )
    }
}

/**
 * Response from a chat completion request.
 */
@Serializable
data class ChatResponse(
    val role: String,
    val content: String?,
    val toolCalls: List<ToolCall>?,
    val finishReason: String?,
    val usage: Usage?
)

@Serializable
data class ToolCall(
    val id: String,
    val type: String,
    val function: ToolFunction
)

@Serializable
data class ToolFunction(
    val name: String,
    val arguments: String
)

@Serializable
data class Usage(
    val promptTokens: Int,
    val completionTokens: Int,
    val totalTokens: Int
)

// Exception hierarchy for cloud errors
open class CloudException(message: String) : Exception(message)
class CloudAuthException(message: String) : CloudException(message)
class CloudUnavailableException(message: String) : CloudException(message)
class CloudRateLimitException(message: String) : CloudException(message)
