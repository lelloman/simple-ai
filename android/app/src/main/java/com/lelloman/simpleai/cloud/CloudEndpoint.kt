package com.lelloman.simpleai.cloud

import com.lelloman.simpleai.capability.CapabilityStatus
import okhttp3.HttpUrl
import okhttp3.HttpUrl.Companion.toHttpUrlOrNull

object CloudEndpoint {
    fun chatUrl(endpoint: String): HttpUrl? {
        val url = endpoint.trim().toHttpUrlOrNull() ?: return null
        if (url.scheme != "https" || url.username.isNotEmpty() || url.password.isNotEmpty() ||
            url.query != null || url.fragment != null) return null
        return url.newBuilder().encodedPath(url.encodedPath.trimEnd('/') + "/v1/chat/completions").build()
    }

    fun status(endpoint: String): CapabilityStatus = if (chatUrl(endpoint) != null) CapabilityStatus.Ready
        else CapabilityStatus.Error("Set a server URL in Settings → Cloud AI.", canRetry = false)
}
