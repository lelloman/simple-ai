package com.lelloman.simpleai.service

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Intent
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
import com.lelloman.simpleai.capability.CapabilityManager
import com.lelloman.simpleai.capability.CapabilityStatus
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
    private var nluEngine: OnnxNLUEngine? = null
    private var translationManager: TranslationManager? = null
    // TODO: cloudClient, llamaEngine will be added in later phases

    private val json = Json {
        ignoreUnknownKeys = true
        encodeDefaults = true
    }

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
            // Validate protocol
            ProtocolHandler.validateProtocol(protocolVersion)?.let { return it }
            val proto = ProtocolHandler.clampProtocol(protocolVersion)

            // Check capability
            val status = capabilityManager.voiceCommandsStatus.value
            if (status !is CapabilityStatus.Ready) {
                return when (status) {
                    is CapabilityStatus.NotDownloaded -> ProtocolHandler.error(
                        proto, ErrorCode.CAPABILITY_NOT_READY,
                        "Voice Commands capability not downloaded"
                    )
                    is CapabilityStatus.Downloading -> ProtocolHandler.error(
                        proto, ErrorCode.CAPABILITY_DOWNLOADING,
                        "Voice Commands downloading: ${(status.progress * 100).toInt()}%",
                        buildJsonObject { put("progress", status.progress) }
                    )
                    is CapabilityStatus.Error -> ProtocolHandler.error(
                        proto, ErrorCode.CAPABILITY_ERROR, status.message
                    )
                    else -> ProtocolHandler.error(proto, ErrorCode.CAPABILITY_NOT_READY, "Voice Commands not ready")
                }
            }

            val engine = nluEngine ?: return ProtocolHandler.error(
                proto, ErrorCode.CAPABILITY_ERROR, "NLU engine not initialized"
            )

            // Check if we need to switch adapters
            return runBlocking(Dispatchers.IO) {
                try {
                    val currentAdapter = engine.adapters.firstOrNull()
                    val needsSwitch = currentAdapter?.id != adapterId || currentAdapter.version != adapterVersion

                    if (needsSwitch) {
                        // Need all FDs to apply new adapter
                        if (patchFd == null || headsFd == null || tokenizerFd == null || configFd == null) {
                            return@runBlocking ProtocolHandler.error(
                                proto, ErrorCode.INVALID_REQUEST,
                                "Adapter files required for first call or version change"
                            )
                        }

                        engine.applyAdapter(
                            adapterId, adapterVersion,
                            patchFd, headsFd, tokenizerFd, configFd
                        ).onFailure { e ->
                            return@runBlocking ProtocolHandler.error(
                                proto, ErrorCode.ADAPTER_LOAD_FAILED,
                                "Failed to apply adapter: ${e.message}"
                            )
                        }
                    }

                    // Run classification
                    engine.classify(text, adapterId).fold(
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
                                proto, ErrorCode.INTERNAL_ERROR,
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

        override fun clearAdapter(protocolVersion: Int): String {
            ProtocolHandler.validateProtocol(protocolVersion)?.let { return it }
            val proto = ProtocolHandler.clampProtocol(protocolVersion)

            val engine = nluEngine ?: return ProtocolHandler.error(
                proto, ErrorCode.CAPABILITY_ERROR, "NLU engine not initialized"
            )

            return runBlocking(Dispatchers.IO) {
                engine.removeAdapter().fold(
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

        override fun translate(
            protocolVersion: Int,
            text: String,
            sourceLang: String,
            targetLang: String
        ): String {
            ProtocolHandler.validateProtocol(protocolVersion)?.let { return it }
            val proto = ProtocolHandler.clampProtocol(protocolVersion)

            // Check capability
            val status = capabilityManager.translationStatus.value
            if (status !is CapabilityStatus.Ready) {
                return when (status) {
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

            val manager = translationManager ?: return ProtocolHandler.error(
                proto, ErrorCode.CAPABILITY_ERROR, "Translation manager not initialized"
            )

            // Validate languages
            if (!manager.isLanguageSupported(targetLang)) {
                return ProtocolHandler.error(
                    proto, ErrorCode.INVALID_REQUEST, "Unsupported target language: $targetLang"
                )
            }
            if (sourceLang != "auto" && !manager.isLanguageSupported(sourceLang)) {
                return ProtocolHandler.error(
                    proto, ErrorCode.INVALID_REQUEST, "Unsupported source language: $sourceLang"
                )
            }

            return runBlocking(Dispatchers.IO) {
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

        override fun getTranslationLanguages(protocolVersion: Int): String {
            ProtocolHandler.validateProtocol(protocolVersion)?.let { return it }
            val proto = ProtocolHandler.clampProtocol(protocolVersion)

            val languages = capabilityManager.downloadedLanguages.value
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
            authToken: String
        ): String {
            ProtocolHandler.validateProtocol(protocolVersion)?.let { return it }
            val proto = ProtocolHandler.clampProtocol(protocolVersion)

            // TODO: Implement cloud proxy in Phase 4
            return ProtocolHandler.error(proto, ErrorCode.CAPABILITY_NOT_READY, "Cloud AI not yet implemented")
        }

        override fun localGenerate(
            protocolVersion: Int,
            prompt: String,
            maxTokens: Int,
            temperature: Float
        ): String {
            ProtocolHandler.validateProtocol(protocolVersion)?.let { return it }
            val proto = ProtocolHandler.clampProtocol(protocolVersion)

            // TODO: Implement local LLM in Phase 5
            return ProtocolHandler.error(proto, ErrorCode.CAPABILITY_NOT_READY, "Local AI not yet implemented")
        }

        override fun localChat(
            protocolVersion: Int,
            messagesJson: String,
            toolsJson: String?,
            systemPrompt: String?
        ): String {
            ProtocolHandler.validateProtocol(protocolVersion)?.let { return it }
            val proto = ProtocolHandler.clampProtocol(protocolVersion)

            // TODO: Implement local LLM in Phase 5
            return ProtocolHandler.error(proto, ErrorCode.CAPABILITY_NOT_READY, "Local AI not yet implemented")
        }
    }

    private fun buildServiceInfoData() = buildJsonObject {
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
        when (val status = capability.status) {
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
            is CapabilityStatus.Ready -> {
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
            is CapabilityStatus.NotDownloaded -> {
                put("status", "not_downloaded")
            }
            is CapabilityStatus.Downloading -> {
                put("status", "downloading")
                put("progress", status.progress)
            }
            is CapabilityStatus.Ready -> {
                put("status", "ready")
            }
            is CapabilityStatus.Error -> {
                put("status", "error")
                put("message", status.message)
            }
        }
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

        capabilityManager = CapabilityManager(this)

        createNotificationChannel()
        startForeground(NOTIFICATION_ID, createNotification("Initializing..."))

        initializeEngines()
    }

    private fun initializeEngines() {
        // Initialize NLU engine (Voice Commands)
        serviceScope.launch(Dispatchers.IO) {
            try {
                Log.i(TAG, "Initializing NLU engine...")
                capabilityManager.updateVoiceCommandsStatus(
                    CapabilityStatus.Downloading(0, CapabilityManager.VOICE_COMMANDS_MODEL_SIZE)
                )

                val engine = OnnxNLUEngine(this@SimpleAIService)
                engine.initialize().fold(
                    onSuccess = {
                        nluEngine = engine
                        capabilityManager.updateVoiceCommandsStatus(CapabilityStatus.Ready)
                        Log.i(TAG, "NLU engine ready")
                        updateNotification("Ready")
                    },
                    onFailure = { e ->
                        Log.e(TAG, "Failed to initialize NLU engine", e)
                        capabilityManager.updateVoiceCommandsStatus(
                            CapabilityStatus.Error(e.message ?: "Failed to initialize")
                        )
                        updateNotification("Error: ${e.message}")
                    }
                )
            } catch (e: Exception) {
                Log.e(TAG, "Error initializing NLU", e)
                capabilityManager.updateVoiceCommandsStatus(
                    CapabilityStatus.Error(e.message ?: "Initialization error")
                )
            }
        }

        // Initialize Translation manager
        serviceScope.launch(Dispatchers.IO) {
            try {
                Log.i(TAG, "Initializing Translation manager...")
                val manager = TranslationManager(this@SimpleAIService)
                manager.initialize()
                translationManager = manager

                // Initial sync with capability manager
                val downloaded = manager.downloadedLanguages.value
                syncTranslationLanguages(downloaded)

                if (downloaded.isNotEmpty()) {
                    capabilityManager.updateTranslationStatus(CapabilityStatus.Ready)
                    Log.i(TAG, "Translation ready with languages: $downloaded")
                } else {
                    Log.i(TAG, "Translation manager initialized, no languages downloaded")
                }

                // Observe ongoing changes to downloaded languages
                manager.downloadedLanguages.collect { languages ->
                    syncTranslationLanguages(languages)
                    if (languages.isNotEmpty()) {
                        capabilityManager.updateTranslationStatus(CapabilityStatus.Ready)
                    } else {
                        capabilityManager.updateTranslationStatus(CapabilityStatus.NotDownloaded(0))
                    }
                }
            } catch (e: Exception) {
                Log.e(TAG, "Error initializing Translation", e)
                capabilityManager.updateTranslationStatus(
                    CapabilityStatus.Error(e.message ?: "Initialization error")
                )
            }
        }

        // TODO: Initialize cloud and local AI in later phases
    }

    /**
     * Sync downloaded languages from TranslationManager to CapabilityManager.
     * This ensures both components have the same view of downloaded languages.
     */
    private fun syncTranslationLanguages(languages: Set<String>) {
        capabilityManager.syncTranslationLanguages(languages)
        Log.d(TAG, "Synced translation languages: $languages")
    }

    override fun onBind(intent: Intent?): IBinder {
        Log.i(TAG, "SimpleAIService onBind")
        return binder
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        return START_STICKY
    }

    override fun onDestroy() {
        super.onDestroy()
        Log.i(TAG, "SimpleAIService onDestroy")
        nluEngine?.release()
        nluEngine = null
        translationManager?.release()
        translationManager = null
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
            .setOngoing(true)
            .build()
    }

    private fun updateNotification(text: String) {
        getSystemService(NotificationManager::class.java)
            .notify(NOTIFICATION_ID, createNotification(text))
    }
}
