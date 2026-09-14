package com.lelloman.simpleai.cloud

import android.content.SharedPreferences
import com.lelloman.simpleai.BuildConfig
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow

/** Device-local endpoint. An explicitly empty override disables cloud access. */
class CloudSettings(private val preferences: SharedPreferences, defaultEndpoint: String = BuildConfig.CLOUD_LLM_ENDPOINT) {
    private val current = MutableStateFlow(preferences.getString("endpoint", defaultEndpoint).orEmpty())
    val endpoint = current.asStateFlow()

    fun save(endpoint: String): Boolean {
        val value = endpoint.trim()
        if (value.isNotEmpty() && CloudEndpoint.chatUrl(value) == null) return false
        if (!preferences.edit().putString("endpoint", value).commit()) return false
        current.value = value
        return true
    }
}
