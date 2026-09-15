package com.lelloman.simpleai.cloud

import com.lelloman.simpleai.download.withResponse
import kotlinx.coroutines.CancellationException
import android.util.Log
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
class CloudLLMClient(private val endpointProvider: () -> String) {

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
        .callTimeout(TIMEOUT_SECONDS, TimeUnit.SECONDS)
        .connectTimeout(TIMEOUT_SECONDS, TimeUnit.SECONDS)
        .readTimeout(TIMEOUT_SECONDS, TimeUnit.SECONDS)
        .writeTimeout(TIMEOUT_SECONDS, TimeUnit.SECONDS)
        .followRedirects(false)
        .followSslRedirects(false)
        .build()

    private val endpoint: String
        get() = endpointProvider()

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
        promptCacheKey: String?,
        authToken: String,
        server: String = endpoint,
        sourceApp: String = "com.lelloman.simpleai"
    ): Result<ChatResponse> = withContext(Dispatchers.IO) {
        val chatUrl = CloudEndpoint.chatUrl(server) ?: return@withContext Result.failure(
            CloudUnavailableException("Set a server URL in Settings → Cloud AI")
        )
        try {
            // Build messages array with optional system prompt
            val fullMessages = buildMessages(messages, systemPrompt)

            // Build request body
            val requestBody = buildRequestBody(fullMessages, tools, promptCacheKey)
            val requestJson = json.encodeToString(requestBody)

            Log.d(TAG, "Sending cloud request: messages=${fullMessages.size}, tools=${tools?.size ?: 0}")

            val request = Request.Builder()
                .url(chatUrl)
                .header("Authorization", "Bearer $authToken")
                .header("Content-Type", "application/json")
                .header("X-SimpleAI-Source-App", sourceApp)
                .post(requestJson.toRequestBody(JSON_MEDIA_TYPE))
                .build()

            httpClient.newCall(request).withResponse { response ->

            if (!response.isSuccessful) {
                Log.w(TAG, "Cloud request failed: HTTP ${response.code}")

                return@withResponse when (response.code) {
                    401, 403 -> Result.failure(CloudAuthException("Authentication failed: ${response.code}"))
                    429 -> Result.failure(CloudRateLimitException("Rate limited"))
                    500, 502, 503, 504 -> Result.failure(CloudUnavailableException("Server error: ${response.code}"))
                    else -> Result.failure(CloudException("Request failed: HTTP ${response.code}"))
                }
            }

            val responseBody = response.body?.string()
                ?: return@withResponse Result.failure(CloudException("Empty response body"))

            Log.d(TAG, "Cloud response: HTTP ${response.code}, characters=${responseBody.length}")
            val chatResponse = parseResponse(responseBody)
            Log.d(TAG, "Cloud response parsed: contentCharacters=${chatResponse.content?.length ?: 0}, toolCalls=${chatResponse.toolCalls?.size ?: 0}")
            Result.success(chatResponse)
            }

        } catch (e: CancellationException) {
            throw e
        } catch (e: java.net.UnknownHostException) {
            Log.w(TAG, "Cloud network unavailable")
            Result.failure(CloudUnavailableException("Network unavailable"))
        } catch (e: java.net.SocketTimeoutException) {
            Log.w(TAG, "Cloud request timed out")
            Result.failure(CloudUnavailableException("Request timed out"))
        } catch (e: Exception) {
            Log.w(TAG, "Cloud request or response failed: ${e.javaClass.simpleName}")
            Result.failure(CloudException("Cloud request or response failed"))
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

    internal fun buildRequestBody(
        messages: JsonArray,
        tools: JsonArray?,
        promptCacheKey: String?
    ): JsonObject {
        val fields = mutableMapOf<String, JsonElement>(
            // Select the server-managed class explicitly, independent of the user’s roles.
            "model" to kotlinx.serialization.json.JsonPrimitive("class:fast"),
            "messages" to messages
        )

        if (tools != null && tools.isNotEmpty()) {
            fields["tools"] = tools
        }
        if (promptCacheKey != null) {
            fields["prompt_cache_key"] = kotlinx.serialization.json.JsonPrimitive(promptCacheKey)
        }

        return JsonObject(fields)
    }

    internal fun parseResponse(responseBody: String): ChatResponse {
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
        val usage = (responseJson["usage"] as? JsonObject)?.let { usageObj ->
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
