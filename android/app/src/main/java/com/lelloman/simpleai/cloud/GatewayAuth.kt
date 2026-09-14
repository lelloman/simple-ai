package com.lelloman.simpleai.cloud

import android.content.Context
import android.content.Intent
import android.net.Uri
import com.lelloman.simpleai.download.withResponse
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.NonCancellable
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlinx.coroutines.withContext
import net.openid.appauth.*
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.HttpUrl.Companion.toHttpUrl
import org.json.JSONObject
import java.util.concurrent.TimeUnit
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException

/** Only this gateway owns credentials. Caller-provided tokens are never used. */
class GatewayAuth(context: Context, private val settings: CloudSettings, private val store: GatewayCredentialStore = GatewayCredentialStore(context),
    private val service: AuthorizationService = AuthorizationService(context.applicationContext)) {
    private val mutex = Mutex()
    private val http = OkHttpClient.Builder().followRedirects(false).followSslRedirects(false).callTimeout(20, TimeUnit.SECONDS).build()
    private val _signedIn = MutableStateFlow(false)
    val signedIn = _signedIn.asStateFlow()
    private val _error = MutableStateFlow<String?>(null)
    val error = _error.asStateFlow()
    init { _signedIn.value = runCatching { read()?.optJSONObject("auth")?.let { AuthState.jsonDeserialize(it).isAuthorized } == true }.getOrDefault(false) }

    private fun read(): JSONObject? = store.read()?.let(::JSONObject)?.takeIf { it.getString("server") == settings.endpoint.value }
    private fun persist(server: String, state: AuthState) {
        check(server == settings.endpoint.value) { "Server changed. Sign in again." }
        store.write(JSONObject().put("server", server).put("auth", state.jsonSerialize()).toString())
        _signedIn.value = state.isAuthorized
    }

    suspend fun changeServer(value: String): Boolean = withContext(Dispatchers.IO) { mutex.withLock {
        val normalized = value.trim()
        if (normalized.isNotEmpty() && CloudEndpoint.chatUrl(normalized) == null) return@withLock false
        if (normalized != settings.endpoint.value) { store.clear(); _signedIn.value = false; _error.value = null }
        settings.save(normalized)
    } }
    suspend fun signOut() = withContext(Dispatchers.IO) { mutex.withLock {
        store.clear(); _signedIn.value = false; _error.value = null
    } }
    fun reportFailure() { _error.value = "Sign-in failed or was cancelled. Try again." }

    suspend fun begin(): Intent = withContext(Dispatchers.IO) { mutex.withLock {
        _error.value = null
        val server = settings.endpoint.value
        require(CloudEndpoint.chatUrl(server) != null) { "Set a server URL first" }
        val metadataUrl = server.toHttpUrl().newBuilder().addPathSegments(".well-known/simple-ai").build()
        val metadata = http.newCall(Request.Builder().url(metadataUrl).build()).withResponse {
            check(it.isSuccessful) { "Server does not provide sign-in configuration" }
            JSONObject(it.body?.string() ?: error("Empty server configuration"))
        }
        val issuer = metadata.getString("issuer")
        require(CloudEndpoint.chatUrl(issuer) != null)
        val clientId = metadata.optString("client_id").takeUnless { it.isBlank() || it == "null" }
            ?: error("Server administrator must configure Android sign-in")
        val configuration = suspendCancellableCoroutine<AuthorizationServiceConfiguration> { c ->
            AuthorizationServiceConfiguration.fetchFromIssuer(Uri.parse(issuer)) { result, exception ->
                if (c.isActive) { if (result != null) c.resume(result) else c.resumeWithException(exception ?: Exception("Discovery failed")) }
            }
        }
        require(configuration.discoveryDoc?.issuer?.toString() == issuer)
        require(configuration.authorizationEndpoint.scheme == "https" && configuration.tokenEndpoint.scheme == "https")
        val request = AuthorizationRequest.Builder(configuration, clientId, ResponseTypeValues.CODE, Uri.parse(REDIRECT))
            .setScope("openid profile email").build()
        check(server == settings.endpoint.value)
        store.write(JSONObject().put("server", server).put("pending", request.jsonSerialize()).toString())
        _signedIn.value = false
        service.getAuthorizationRequestIntent(request)
    } }

    suspend fun complete(intent: Intent) = withContext(Dispatchers.IO) { mutex.withLock {
        val saved = read() ?: error("Sign-in is no longer active")
        val expected = AuthorizationRequest.jsonDeserialize(saved.getJSONObject("pending"))
        val response = AuthorizationResponse.fromIntent(intent) ?: throw (AuthorizationException.fromIntent(intent) ?: Exception("Sign-in cancelled"))
        check(response.state == expected.state && response.request.jsonSerializeString() == expected.jsonSerializeString()) { "Unexpected sign-in response" }
        val state = AuthState(response, null)
        val token = exchange(response.createTokenExchangeRequest())
        state.update(token, null)
        check(!state.accessToken.isNullOrBlank())
        persist(saved.getString("server"), state)
        _error.value = null
    } }

    private suspend fun exchange(request: TokenRequest): TokenResponse = suspendCancellableCoroutine { c ->
        service.performTokenRequest(request) { response, exception ->
            if (c.isActive) { if (response != null) c.resume(response) else c.resumeWithException(exception ?: Exception("Token request failed")) }
        }
    }

    data class Session(val server: String, val token: String)
    suspend fun session(): Session = withContext(Dispatchers.IO) { mutex.withLock {
        val saved = read() ?: throw CloudAuthException("Open SimpleAI → Settings → Cloud AI and sign in")
        val state = saved.optJSONObject("auth")?.let(AuthState::jsonDeserialize)
            ?: throw CloudAuthException("Sign in to SimpleAI first")
        val server = saved.getString("server")
        if (state.needsTokenRefresh) {
            if (state.refreshToken.isNullOrBlank()) { store.clear(); _signedIn.value = false; throw CloudAuthException("Sign in to SimpleAI again") }
            try { withContext(NonCancellable) {
                state.update(exchange(state.createTokenRefreshRequest()), null)
                persist(server, state)
            } }
            catch (e: AuthorizationException) {
                if (e.error == "invalid_grant") { store.clear(); _signedIn.value = false }
                throw CloudAuthException("Could not renew SimpleAI sign-in. Open Settings → Cloud AI")
            }
        }
        check(server == settings.endpoint.value)
        Session(server, state.accessToken ?: throw CloudAuthException("Sign in to SimpleAI again"))
    } }
    companion object { const val REDIRECT = "com.lelloman.simpleai:/oauth2redirect" }
}
