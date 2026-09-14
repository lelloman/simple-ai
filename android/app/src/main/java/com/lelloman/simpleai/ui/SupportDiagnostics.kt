package com.lelloman.simpleai.ui

import android.os.Build
import com.lelloman.simpleai.BuildConfig
import com.lelloman.simpleai.model.LocalAIModel
import com.lelloman.simpleai.model.NluModel

/** Fixed technical metadata only; never includes prompts, tokens, client identities or endpoints. */
internal fun supportDiagnostics(): String = """
SimpleAI ${BuildConfig.VERSION_NAME} (${BuildConfig.VERSION_CODE}), ${BuildConfig.BUILD_TYPE}
Service ${BuildConfig.SERVICE_VERSION}, protocol ${BuildConfig.MIN_PROTOCOL_VERSION}–${BuildConfig.MAX_PROTOCOL_VERSION}
Android API ${Build.VERSION.SDK_INT}; ABI ${Build.SUPPORTED_ABIS.joinToString()}
Voice Commands: ${NluModel.FILE_NAME}
Revision: ${NluModel.REVISION}; SHA-256: ${NluModel.SHA256}; ${NluModel.SIZE_BYTES} bytes
Local AI: ${LocalAIModel.NAME}; ${LocalAIModel.FILE_NAME}
Revision: ${LocalAIModel.REVISION}; SHA-256: ${LocalAIModel.SHA256}; ${LocalAIModel.SIZE_BYTES} bytes
ONNX Runtime 1.29.0; llama.cpp Kotlin wrapper 0.2.0; ML Kit Translation 17.0.3; tokenizers 0.22.2
""".trimIndent()
