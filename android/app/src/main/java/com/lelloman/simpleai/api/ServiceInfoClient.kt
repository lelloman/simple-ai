package com.lelloman.simpleai.api

import com.lelloman.simpleai.BuildConfig
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive

/** Shares the service's protocol and preserves failures for the UI to display. */
object ServiceInfoClient {
    fun request(getServiceInfo: (Int) -> String): JsonObject {
        val response = Json.parseToJsonElement(getServiceInfo(BuildConfig.MAX_PROTOCOL_VERSION)).jsonObject
        check(response["status"]?.jsonPrimitive?.content == "success") {
            response["error"]?.jsonObject?.get("message")?.jsonPrimitive?.content
                ?: "Could not read service status"
        }
        return requireNotNull(response["data"]?.jsonObject?.get("capabilities")?.jsonObject) {
            "Service response is missing capabilities"
        }
    }
}
