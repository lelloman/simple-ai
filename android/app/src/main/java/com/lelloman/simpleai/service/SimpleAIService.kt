package com.lelloman.simpleai.service

import android.app.PendingIntent
import com.lelloman.simpleai.MainActivity
import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Intent
import com.lelloman.simpleai.api.ActiveRequests
import com.lelloman.simpleai.api.RequestValidation
import com.lelloman.simpleai.cloud.CloudRateLimitException
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.TimeoutCancellationException
import kotlinx.coroutines.withContext
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.Job
import android.os.Binder
import android.os.SystemClock
import com.lelloman.simpleai.access.CallerBudget
import com.lelloman.simpleai.access.ClientAccess
import android.os.Build
import android.os.IBinder
import android.os.ParcelFileDescriptor
import android.util.Log
import androidx.core.app.NotificationCompat
import com.lelloman.simpleai.BuildConfig
import com.lelloman.simpleai.ISimpleAI
import com.lelloman.simpleai.R
import com.lelloman.simpleai.api.ErrorCode
import com.lelloman.simpleai.api.ProtocolHandler
import com.lelloman.simpleai.capability.CapabilityId
import com.lelloman.simpleai.cloud.CloudAuthException
import com.lelloman.simpleai.cloud.CloudLLMClient
import com.lelloman.simpleai.cloud.CloudUnavailableException
import com.lelloman.simpleai.capability.CapabilityManager
import com.lelloman.simpleai.capability.CapabilityStatus
import com.lelloman.simpleai.llm.GenerationParams
import com.lelloman.simpleai.llm.GenerationTimeoutException
import com.lelloman.simpleai.llm.LlamaEngine
import com.lelloman.simpleai.model.ModelRepository
import com.lelloman.simpleai.model.LocalAIModel
import com.lelloman.simpleai.nlu.OnnxNLUEngine
import com.lelloman.simpleai.translation.TranslationManager
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.buildJsonArray
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.put

/**
 * SimpleAI foreground service implementing the new capability-based AIDL interface.
 */
class SimpleAIService : Service() {

    companion object {
        private const val TAG = "SimpleAIService"
        private const val NOTIFICATION_CHANNEL_ID = "simple_ai_service_channel"
        private const val NOTIFICATION_ID = 1

        const val ACTION_STATUS_UPDATE = "com.lelloman.simpleai.STATUS_UPDATE"
    }

    private val serviceScope = CoroutineScope(SupervisorJob() + Dispatchers.Main)

    private lateinit var capabilityManager: CapabilityManager

    // Engines
    private val models by lazy { ModelRepository.get(this) }
    private val nluEngine: OnnxNLUEngine? get() = models.voice.engine
    private val translationManager: TranslationManager get() = models.translation
    private val cloudClient = CloudLLMClient { models.cloudSettings.endpoint.value }
    private val llamaEngine: LlamaEngine? get() = models.local.engine

    private val json = Json {
        ignoreUnknownKeys = true
        encodeDefaults = true
    }

    private val activeCount = java.util.concurrent.atomic.AtomicInteger()
    @Volatile private var explicitlyStarted = false
    private val activeRequests = ActiveRequests()
    private val callerBudget = CallerBudget(SystemClock::elapsedRealtime)

    private val binder = object : ISimpleAI.Stub() {

        override fun getServiceInfo(protocolVersion: Int): String {
            // Validate protocol
            ProtocolHandler.validateProtocol(protocolVersion)?.let { return it }

            val proto = ProtocolHandler.clampProtocol(protocolVersion)
            return ProtocolHandler.success(proto, buildServiceInfoData())
        }

        override fun classify(
            protocolVersion: Int,
            text: String,
            adapterId: String,
            adapterVersion: String,
            patchFd: ParcelFileDescriptor?,
            headsFd: ParcelFileDescriptor?,
            tokenizerFd: ParcelFileDescriptor?,
            configFd: ParcelFileDescriptor?
        ): String {
            val callerUid = Binder.getCallingUid()
            val responseProtocol = ProtocolHandler.clampProtocol(protocolVersion)
            if (!ClientAccess.get(this@SimpleAIService).allowed(callerUid)) return ProtocolHandler.error(
                responseProtocol, ErrorCode.CLIENT_NOT_APPROVED, "Open SimpleAI > Apps to allow access")
            val lease = callerBudget.acquire(callerUid) ?: return ProtocolHandler.error(
                responseProtocol, ErrorCode.RATE_LIMITED, "Caller busy or request budget exceeded; retry later")
            try {
            return runRequest(callerUid, responseProtocol) {
            RequestValidation.text(text, "text")
            RequestValidation.text(adapterId, "adapterId", 256)
            RequestValidation.text(adapterVersion, "adapterVersion", 256)
            // Validate protocol
            ProtocolHandler.validateProtocol(protocolVersion)?.let { return@runRequest it }
            val proto = ProtocolHandler.clampProtocol(protocolVersion)

            // Wait for engine to be ready (blocks until model is loaded into memory)
            // Must wait BEFORE checking status, since status is NotDownloaded during init
            withContext(Dispatchers.IO) {
                models.voice.initialize()
            }

            // Check capability status AFTER waiting for initialization
            val status = capabilityManager.voiceCommandsStatus.value
            if (status is CapabilityStatus.NotDownloaded) {
                return@runRequest ProtocolHandler.error(
                    proto, ErrorCode.CAPABILITY_NOT_READY,
                    "Voice Commands capability not downloaded"
                )
            }
            if (status is CapabilityStatus.Downloading) {
                return@runRequest ProtocolHandler.error(
                    proto, ErrorCode.CAPABILITY_DOWNLOADING,
                    "Voice Commands downloading: ${(status.progress * 100).toInt()}%",
                    buildJsonObject { put("progress", status.progress) }
                )
            }
            if (status is CapabilityStatus.Error) {
                return@runRequest ProtocolHandler.error(
                    proto, ErrorCode.CAPABILITY_ERROR, status.message
                )
            }

            val engine = nluEngine ?: return@runRequest ProtocolHandler.error(
                proto, ErrorCode.CAPABILITY_ERROR, "NLU engine failed to initialize"
            )

            // Check if we need to switch adapters
            return@runRequest withContext(Dispatchers.IO) {
                try {
                    engine.classifyWithAdapter(
                        text, "$callerUid:$adapterId", adapterVersion, patchFd, headsFd, tokenizerFd, configFd
                    ).fold(
                        onSuccess = { result ->
                            ProtocolHandler.success(proto, buildJsonObject {
                                put("intent", result.intent)
                                put("intentConfidence", result.intentConfidence)
                                put("slots", buildJsonObject {
                                    result.slots.forEach { (slotType, values) ->
                                        put(slotType, buildJsonArray {
                                            values.forEach { add(JsonPrimitive(it)) }
                                        })
                                    }
                                })
                            })
                        },
                        onFailure = { e ->
                            ProtocolHandler.error(
                                proto, when (e) {
                                    is OnnxNLUEngine.MissingAdapterFiles -> ErrorCode.INVALID_REQUEST
                                    is OnnxNLUEngine.AdapterLoadFailed -> ErrorCode.ADAPTER_LOAD_FAILED
                                    else -> ErrorCode.INTERNAL_ERROR
                                },
                                "Classification failed: ${e.message}"
                            )
                        }
                    )
                } catch (e: Exception) {
                    Log.e(TAG, "Error in classify", e)
                    ProtocolHandler.error(proto, ErrorCode.INTERNAL_ERROR, "Error: ${e.message}")
                }
            }

            }
            } finally { lease.close() }
        }

        override fun clearAdapter(protocolVersion: Int): String {
            val callerUid = Binder.getCallingUid()
            val responseProtocol = ProtocolHandler.clampProtocol(protocolVersion)
            if (!ClientAccess.get(this@SimpleAIService).allowed(callerUid)) return ProtocolHandler.error(
                responseProtocol, ErrorCode.CLIENT_NOT_APPROVED, "Open SimpleAI > Apps to allow access")
            val lease = callerBudget.acquire(callerUid) ?: return ProtocolHandler.error(
                responseProtocol, ErrorCode.RATE_LIMITED, "Caller busy or request budget exceeded; retry later")
            try {
            return runRequest(callerUid, responseProtocol) {
            
            ProtocolHandler.validateProtocol(protocolVersion)?.let { return@runRequest it }
            val proto = ProtocolHandler.clampProtocol(protocolVersion)

            // Wait for engine to be ready
            withContext(Dispatchers.IO) {
                models.voice.initialize()
            }

            val engine = nluEngine ?: return@runRequest ProtocolHandler.error(
                proto, ErrorCode.CAPABILITY_ERROR, "NLU engine failed to initialize"
            )

            return@runRequest withContext(Dispatchers.IO) {
                engine.removeAdapter("$callerUid:").fold(
                    onSuccess = {
                        ProtocolHandler.success(proto, buildJsonObject {
                            put("message", "Adapter removed")
                        })
                    },
                    onFailure = { e ->
                        ProtocolHandler.error(proto, ErrorCode.INTERNAL_ERROR, "Failed to remove adapter: ${e.message}")
                    }
                )
            }

            }
            } finally { lease.close() }
        }

        override fun translate(
            protocolVersion: Int,
            text: String,
            sourceLang: String,
            targetLang: String
        ): String {
            val callerUid = Binder.getCallingUid()
            val responseProtocol = ProtocolHandler.clampProtocol(protocolVersion)
            if (!ClientAccess.get(this@SimpleAIService).allowed(callerUid)) return ProtocolHandler.error(
                responseProtocol, ErrorCode.CLIENT_NOT_APPROVED, "Open SimpleAI > Apps to allow access")
            val lease = callerBudget.acquire(callerUid) ?: return ProtocolHandler.error(
                responseProtocol, ErrorCode.RATE_LIMITED, "Caller busy or request budget exceeded; retry later")
            try {
            return runRequest(callerUid, responseProtocol) {
            RequestValidation.text(text, "text")
            ProtocolHandler.validateProtocol(protocolVersion)?.let { return@runRequest it }
            val proto = ProtocolHandler.clampProtocol(protocolVersion)

            // Check capability
            val status = capabilityManager.translationStatus.value
            if (status !is CapabilityStatus.Ready) {
                return@runRequest when (status) {
                    is CapabilityStatus.NotDownloaded -> ProtocolHandler.error(
                        proto, ErrorCode.CAPABILITY_NOT_READY,
                        "No translation languages downloaded"
                    )
                    is CapabilityStatus.Downloading -> ProtocolHandler.error(
                        proto, ErrorCode.CAPABILITY_DOWNLOADING,
                        "Translation model downloading: ${(status.progress * 100).toInt()}%"
                    )
                    is CapabilityStatus.Error -> ProtocolHandler.error(
                        proto, ErrorCode.CAPABILITY_ERROR, status.message
                    )
                    else -> ProtocolHandler.error(proto, ErrorCode.CAPABILITY_NOT_READY, "Translation not ready")
                }
            }

            val manager = translationManager

            // Validate languages
            if (!manager.isLanguageSupported(targetLang)) {
                return@runRequest ProtocolHandler.error(
                    proto, ErrorCode.INVALID_REQUEST, "Unsupported target language: $targetLang"
                )
            }
            if (sourceLang != "auto" && !manager.isLanguageSupported(sourceLang)) {
                return@runRequest ProtocolHandler.error(
                    proto, ErrorCode.INVALID_REQUEST, "Unsupported source language: $sourceLang"
                )
            }

            return@runRequest withContext(Dispatchers.IO) {
                manager.translate(text, sourceLang, targetLang).fold(
                    onSuccess = { result ->
                        ProtocolHandler.success(proto, buildJsonObject {
                            put("translatedText", result.translatedText)
                            put("detectedSourceLang", result.detectedSourceLang)
                        })
                    },
                    onFailure = { e ->
                        val errorCode = when {
                            e.message?.contains("not downloaded") == true -> ErrorCode.TRANSLATION_LANGUAGE_NOT_AVAILABLE
                            else -> ErrorCode.INTERNAL_ERROR
                        }
                        ProtocolHandler.error(proto, errorCode, "Translation failed: ${e.message}")
                    }
                )
            }

            }
            } finally { lease.close() }
        }

        override fun getTranslationLanguages(protocolVersion: Int): String {
            ProtocolHandler.validateProtocol(protocolVersion)?.let { return it }
            val proto = ProtocolHandler.clampProtocol(protocolVersion)

            val languages = com.lelloman.simpleai.translation.TranslationAvailability.available(capabilityManager.downloadedLanguages.value)
            return ProtocolHandler.success(proto, buildJsonObject {
                put("languages", buildJsonArray {
                    languages.forEach { add(JsonPrimitive(it)) }
                })
            })
        }

        override fun cloudChat(
            protocolVersion: Int,
            messagesJson: String,
            toolsJson: String?,
            systemPrompt: String?,
            promptCacheKey: String?,
            authToken: String
        ): String {
            val callerUid = Binder.getCallingUid()
            val responseProtocol = ProtocolHandler.clampProtocol(protocolVersion)
            if (!ClientAccess.get(this@SimpleAIService).allowed(callerUid)) return ProtocolHandler.error(
                responseProtocol, ErrorCode.CLIENT_NOT_APPROVED, "Open SimpleAI > Apps to allow access")
            val lease = callerBudget.acquire(callerUid) ?: return ProtocolHandler.error(
                responseProtocol, ErrorCode.RATE_LIMITED, "Caller busy or request budget exceeded; retry later")
            try {
            return runRequest(callerUid, responseProtocol) {
            RequestValidation.messages(messagesJson)
            require((systemPrompt?.length ?: 0) <= 32768 && (toolsJson?.length ?: 0) <= 32768 && (promptCacheKey?.length ?: 0) <= 256)
            // Legacy AIDL argument intentionally ignored: the gateway owns authentication.
            ProtocolHandler.validateProtocol(protocolVersion)?.let { return@runRequest it }
            val proto = ProtocolHandler.clampProtocol(protocolVersion)

            // Parse messages
            val messages = try {
                RequestValidation.messages(messagesJson)
            } catch (e: Exception) {
                return@runRequest ProtocolHandler.error(
                    proto, ErrorCode.INVALID_REQUEST,
                    "Invalid messages JSON: ${e.message}"
                )
            }

            // Parse tools if provided
            val tools = if (toolsJson != null) {
                try {
                    json.parseToJsonElement(toolsJson).jsonArray
                } catch (e: Exception) {
                    return@runRequest ProtocolHandler.error(
                        proto, ErrorCode.INVALID_REQUEST,
                        "Invalid tools JSON: ${e.message}"
                    )
                }
            } else null

            return@runRequest withContext(Dispatchers.IO) {
                val session = try { models.gatewayAuth.session() }
                catch (e: CloudAuthException) { return@withContext ProtocolHandler.error(proto, ErrorCode.CLOUD_AUTH_FAILED, e.message ?: "Sign in to SimpleAI") }
                val sourceApp = packageManager.getPackagesForUid(callerUid)?.sorted()?.joinToString(",")?.take(255) ?: "unknown"
                cloudClient.chat(messages, tools, systemPrompt, promptCacheKey, session.token, session.server, sourceApp).fold(
                    onSuccess = { response ->
                        ProtocolHandler.success(proto, buildJsonObject {
                            put("role", response.role)
                            response.content?.let { put("content", it) }
                            response.finishReason?.let { put("finishReason", it) }
                            response.toolCalls?.let { toolCalls ->
                                put("toolCalls", buildJsonArray {
                                    toolCalls.forEach { call ->
                                        add(buildJsonObject {
                                            put("id", call.id)
                                            put("type", call.type)
                                            put("function", buildJsonObject {
                                                put("name", call.function.name)
                                                put("arguments", call.function.arguments)
                                            })
                                        })
                                    }
                                })
                            }
                            response.usage?.let { usage ->
                                put("usage", buildJsonObject {
                                    put("promptTokens", usage.promptTokens)
                                    put("completionTokens", usage.completionTokens)
                                    put("totalTokens", usage.totalTokens)
                                })
                            }
                        })
                    },
                    onFailure = { e ->
                        val errorCode = when (e) {
                            is CloudRateLimitException -> ErrorCode.RATE_LIMITED
                            is CloudAuthException -> ErrorCode.CLOUD_AUTH_FAILED
                            is CloudUnavailableException -> ErrorCode.CLOUD_UNAVAILABLE
                            else -> ErrorCode.INTERNAL_ERROR
                        }
                        ProtocolHandler.error(proto, errorCode, e.message ?: "Cloud request failed")
                    }
                )
            }

            }
            } finally { lease.close() }
        }

        override fun localGenerate(
            protocolVersion: Int,
            prompt: String,
            maxTokens: Int,
            temperature: Float
        ): String {
            val callerUid = Binder.getCallingUid()
            val responseProtocol = ProtocolHandler.clampProtocol(protocolVersion)
            if (!ClientAccess.get(this@SimpleAIService).allowed(callerUid)) return ProtocolHandler.error(
                responseProtocol, ErrorCode.CLIENT_NOT_APPROVED, "Open SimpleAI > Apps to allow access")
            val lease = callerBudget.acquire(callerUid) ?: return ProtocolHandler.error(
                responseProtocol, ErrorCode.RATE_LIMITED, "Caller busy or request budget exceeded; retry later")
            try {
            return runRequest(callerUid, responseProtocol) {
            RequestValidation.generation(prompt, maxTokens, temperature)
            ProtocolHandler.validateProtocol(protocolVersion)?.let { return@runRequest it }
            val proto = ProtocolHandler.clampProtocol(protocolVersion)

            // Check capability
            models.local.initialize()
            val status = capabilityManager.localAiStatus.value
            if (status !is CapabilityStatus.Ready) {
                return@runRequest when (status) {
                    is CapabilityStatus.NotDownloaded -> ProtocolHandler.error(
                        proto, ErrorCode.CAPABILITY_NOT_READY,
                        "Local AI model not downloaded. Size: ${LocalAIModel.SIZE_MB} MB"
                    )
                    is CapabilityStatus.Downloading -> ProtocolHandler.error(
                        proto, ErrorCode.CAPABILITY_DOWNLOADING,
                        "Local AI downloading: ${(status.progress * 100).toInt()}%",
                        buildJsonObject { put("progress", status.progress) }
                    )
                    is CapabilityStatus.Error -> ProtocolHandler.error(
                        proto, ErrorCode.CAPABILITY_ERROR, status.message
                    )
                    else -> ProtocolHandler.error(proto, ErrorCode.CAPABILITY_NOT_READY, "Local AI not ready")
                }
            }

            val engine = llamaEngine ?: return@runRequest ProtocolHandler.error(
                proto, ErrorCode.CAPABILITY_ERROR, "LLM engine not initialized"
            )

            val params = GenerationParams(
                maxTokens = maxTokens,
                temperature = temperature
            )

            return@runRequest engine.generateForRequest(prompt, params, currentCoroutineContext()[Job]).fold(
                onSuccess = { text ->
                    ProtocolHandler.success(proto, buildJsonObject {
                        put("text", text)
                    })
                },
                onFailure = { e ->
                    generationError(proto, e)
                }
            )

            }
            } finally { lease.close() }
        }

        override fun localChat(
            protocolVersion: Int,
            messagesJson: String,
            toolsJson: String?,
            systemPrompt: String?
        ): String {
            val callerUid = Binder.getCallingUid()
            val responseProtocol = ProtocolHandler.clampProtocol(protocolVersion)
            if (!ClientAccess.get(this@SimpleAIService).allowed(callerUid)) return ProtocolHandler.error(
                responseProtocol, ErrorCode.CLIENT_NOT_APPROVED, "Open SimpleAI > Apps to allow access")
            val lease = callerBudget.acquire(callerUid) ?: return ProtocolHandler.error(
                responseProtocol, ErrorCode.RATE_LIMITED, "Caller busy or request budget exceeded; retry later")
            try {
            return runRequest(callerUid, responseProtocol) {
            RequestValidation.messages(messagesJson)
            require((systemPrompt?.length ?: 0) <= 8192 && (toolsJson?.length ?: 0) <= 32768)
            ProtocolHandler.validateProtocol(protocolVersion)?.let { return@runRequest it }
            val proto = ProtocolHandler.clampProtocol(protocolVersion)

            // Check capability
            models.local.initialize()
            val status = capabilityManager.localAiStatus.value
            if (status !is CapabilityStatus.Ready) {
                return@runRequest when (status) {
                    is CapabilityStatus.NotDownloaded -> ProtocolHandler.error(
                        proto, ErrorCode.CAPABILITY_NOT_READY,
                        "Local AI model not downloaded. Size: ${LocalAIModel.SIZE_MB} MB"
                    )
                    is CapabilityStatus.Downloading -> ProtocolHandler.error(
                        proto, ErrorCode.CAPABILITY_DOWNLOADING,
                        "Local AI downloading: ${(status.progress * 100).toInt()}%",
                        buildJsonObject { put("progress", status.progress) }
                    )
                    is CapabilityStatus.Error -> ProtocolHandler.error(
                        proto, ErrorCode.CAPABILITY_ERROR, status.message
                    )
                    else -> ProtocolHandler.error(proto, ErrorCode.CAPABILITY_NOT_READY, "Local AI not ready")
                }
            }

            val engine = llamaEngine ?: return@runRequest ProtocolHandler.error(
                proto, ErrorCode.CAPABILITY_ERROR, "LLM engine not initialized"
            )

            val messages = try {
                RequestValidation.messages(messagesJson)
            } catch (e: Exception) {
                return@runRequest ProtocolHandler.error(
                    proto, ErrorCode.INVALID_REQUEST,
                    "Invalid messages JSON: ${e.message}"
                )
            }

            val prompt = try {
                com.lelloman.simpleai.llm.QwenChat.format(messages, systemPrompt, toolsJson)
            } catch (e: Exception) {
                return@runRequest ProtocolHandler.error(proto, ErrorCode.INVALID_REQUEST, e.message ?: "Unsupported local chat request")
            }
            return@runRequest engine.generateForRequest(prompt, GenerationParams(), currentCoroutineContext()[Job]).fold(
                onSuccess = { text ->
                    ProtocolHandler.success(proto, buildJsonObject {
                        put("role", "assistant")
                        put("content", text.trim())
                    })
                },
                onFailure = { e ->
                    generationError(proto, e)
                }
            )

            }
            } finally { lease.close() }
        }
        override fun cancelCurrentRequest(protocolVersion: Int): String {
            ProtocolHandler.validateProtocol(protocolVersion)?.let { return it }
            return ProtocolHandler.success(ProtocolHandler.clampProtocol(protocolVersion), buildJsonObject {
                put("cancelled", activeRequests.cancel(Binder.getCallingUid()))
            })
        }

    }

    private fun runRequest(uid: Int, proto: Int, block: suspend () -> String): String = try {
        activeCount.incrementAndGet()
        if (explicitlyStarted) updateNotification("${activeCount.get()} active request(s)")
        activeRequests.run(uid) { models.withWork(block) }
    } catch (_: TimeoutCancellationException) {
        ProtocolHandler.error(proto, ErrorCode.REQUEST_TIMEOUT, "Request deadline exceeded")
    } catch (_: CancellationException) {
        ProtocolHandler.error(proto, ErrorCode.REQUEST_CANCELLED, "Request cancelled")
    } catch (_: IllegalArgumentException) {
        ProtocolHandler.error(proto, ErrorCode.INVALID_REQUEST, "Invalid request parameters or schema")
    } catch (_: Exception) {
        ProtocolHandler.error(proto, ErrorCode.INTERNAL_ERROR, "Request failed")
    } finally {
        activeCount.decrementAndGet()
        if (explicitlyStarted) updateNotification(if (activeCount.get() == 0) "Available to connected apps" else "${activeCount.get()} active request(s)")
    }

    private fun generationError(proto: Int, error: Throwable): String =
        if (error is GenerationTimeoutException) {
            ProtocolHandler.error(proto, ErrorCode.GENERATION_TIMEOUT, "Generation timed out", buildJsonObject {
                put("partialText", error.partialText)
            })
        } else ProtocolHandler.error(proto, ErrorCode.INTERNAL_ERROR, "Generation failed: ${error.message}")

    private fun buildServiceInfoData() = buildJsonObject {
        put("supportsCancellation", true)
            put("serviceVersion", BuildConfig.SERVICE_VERSION)
        put("minProtocol", BuildConfig.MIN_PROTOCOL_VERSION)
        put("maxProtocol", BuildConfig.MAX_PROTOCOL_VERSION)
        put("capabilities", buildJsonObject {
            put("voiceCommands", buildCapabilityStatus(CapabilityId.VOICE_COMMANDS))
            put("translation", buildTranslationCapabilityStatus())
            put("cloudAi", buildCapabilityStatus(CapabilityId.CLOUD_AI))
            put("localAi", buildCapabilityStatus(CapabilityId.LOCAL_AI))
        })
    }

    private fun buildCapabilityStatus(id: CapabilityId) = buildJsonObject {
        val capability = capabilityManager.getCapability(id)
        if (id == CapabilityId.LOCAL_AI) put("supportsTools", false)
        when (val status = capability.status) {
            CapabilityStatus.Checking -> put("status", "checking")
            CapabilityStatus.Loading -> put("status", "loading")
            is CapabilityStatus.NotDownloaded -> {
                put("status", "not_downloaded")
                put("modelSize", status.totalBytes)
            }
            is CapabilityStatus.Downloading -> {
                put("status", "downloading")
                put("progress", status.progress)
                put("downloadedBytes", status.downloadedBytes)
                put("totalBytes", status.totalBytes)
            }
            CapabilityStatus.Downloaded -> {
                put("status", "ready")
                put("loaded", false)
            }
            is CapabilityStatus.Ready -> {
                put("loaded", true)
                put("status", "ready")
            }
            is CapabilityStatus.Error -> {
                put("status", "error")
                put("message", status.message)
                put("canRetry", status.canRetry)
            }
        }
    }

    private fun buildTranslationCapabilityStatus() = buildJsonObject {
        val capability = capabilityManager.getCapability(CapabilityId.TRANSLATION)
        when (val status = capability.status) {
            CapabilityStatus.Checking -> put("status", "checking")
            CapabilityStatus.Loading -> put("status", "loading")
            is CapabilityStatus.NotDownloaded -> {
                put("status", "not_downloaded")
            }
            is CapabilityStatus.Downloading -> {
                put("status", "downloading")
                put("progress", status.progress)
            }
            CapabilityStatus.Downloaded -> {
                put("status", "ready")
                put("loaded", false)
            }
            is CapabilityStatus.Ready -> {
                put("loaded", true)
                put("status", "ready")
            }
            is CapabilityStatus.Error -> {
                put("status", "error")
                put("message", status.message)
            }
        }
        put("builtInLanguages", buildJsonArray { add(JsonPrimitive("en")) })
        put("languages", buildJsonArray {
            capabilityManager.downloadedLanguages.value.forEach { add(JsonPrimitive(it)) }
        })
    }

    // =========================================================================
    // Service Lifecycle
    // =========================================================================

    override fun onCreate() {
        super.onCreate()
        Log.i(TAG, "SimpleAIService onCreate")

        capabilityManager = models.capabilities

        createNotificationChannel()

        initializeEngines()
    }

    private fun initializeEngines() {
        models.initialize()
    }

    override fun onBind(intent: Intent?): IBinder {
        Log.i(TAG, "SimpleAIService onBind")
        return binder
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        explicitlyStarted = true
        // Compatibility for clients that explicitly start the service. Binding is preferred.
        startForeground(NOTIFICATION_ID, createNotification("Available to connected apps"))
        serviceScope.launch {
            kotlinx.coroutines.delay(60_000)
            explicitlyStarted = false
            stopForeground(STOP_FOREGROUND_REMOVE)
            stopSelf()
        }
        return START_NOT_STICKY
    }

    override fun onDestroy() {
        super.onDestroy()
        Log.i(TAG, "SimpleAIService onDestroy")
        serviceScope.cancel()
    }

    // =========================================================================
    // Notifications
    // =========================================================================

    private fun createNotificationChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                NOTIFICATION_CHANNEL_ID,
                "SimpleAI Service",
                NotificationManager.IMPORTANCE_LOW
            ).apply {
                description = "Shows the status of SimpleAI capabilities"
            }
            getSystemService(NotificationManager::class.java).createNotificationChannel(channel)
        }
    }

    private fun createNotification(text: String): Notification {
        return NotificationCompat.Builder(this, NOTIFICATION_CHANNEL_ID)
            .setContentTitle("SimpleAI")
            .setContentText(text)
            .setSmallIcon(R.drawable.ic_notification)
            .setContentIntent(PendingIntent.getActivity(this, 0, Intent(this, MainActivity::class.java), PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE))
            .setOngoing(true)
            .build()
    }

    private fun updateNotification(text: String) {
        getSystemService(NotificationManager::class.java)
            .notify(NOTIFICATION_ID, createNotification(text))
    }
}
