package com.lelloman.simpleai.api

import com.lelloman.simpleai.BuildConfig
import com.lelloman.simpleai.translation.TranslationResult
import kotlinx.serialization.json.*

internal object ServiceTranslationClient {
    fun request(text: String, source: String, target: String, call: (Int, String, String, String) -> String): Result<TranslationResult> = runCatching {
        val response = Json.parseToJsonElement(call(BuildConfig.MAX_PROTOCOL_VERSION, text, source, target)).jsonObject
        check(response["status"]?.jsonPrimitive?.content == "success") {
            response["error"]?.jsonObject?.get("message")?.jsonPrimitive?.content ?: "Service translation failed"
        }
        val data = response.getValue("data").jsonObject
        TranslationResult(data.getValue("translatedText").jsonPrimitive.content, data.getValue("detectedSourceLang").jsonPrimitive.content)
    }
}
