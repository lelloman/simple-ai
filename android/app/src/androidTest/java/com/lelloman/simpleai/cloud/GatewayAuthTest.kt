package com.lelloman.simpleai.cloud

import android.content.Context
import android.net.Uri
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import kotlinx.coroutines.runBlocking
import net.openid.appauth.*
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.After
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class GatewayAuthTest {
    private val context = InstrumentationRegistry.getInstrumentation().targetContext
    private val settings = CloudSettings(context.getSharedPreferences("gateway_test_settings", Context.MODE_PRIVATE), "")
    private val store = GatewayCredentialStore(context, "gateway_test_credentials", "simpleai.gateway.test")
    @Before fun setup() { store.clear(); settings.save("https://gateway.example") }
    @After fun cleanup() { store.clear(); context.getSharedPreferences("gateway_test_settings", Context.MODE_PRIVATE).edit().clear().commit() }

    private fun signedInState(expired: Boolean = false): String {
        val config = AuthorizationServiceConfiguration(Uri.parse("https://issuer.example/authorize"), Uri.parse("https://issuer.example/token"))
        val request = AuthorizationRequest.Builder(config, "gateway", ResponseTypeValues.CODE, Uri.parse(GatewayAuth.REDIRECT)).build()
        val response = AuthorizationResponse.Builder(request).setAuthorizationCode("test-code").build()
        val auth = AuthState(response, null)
        auth.update(TokenResponse.Builder(response.createTokenExchangeRequest())
            .setAccessToken("gateway-token-never-caller-token")
            .setRefreshToken("gateway-refresh-token")
            .setTokenType("Bearer")
            .setAccessTokenExpirationTime(System.currentTimeMillis() + if (expired) -600000 else 600000).build(), null)
        return JSONObject().put("server", settings.endpoint.value).put("auth", auth.jsonSerialize()).toString()
    }

    @Test fun credentialsAreEncryptedAndSurviveRecreation() = runBlocking {
        store.write(signedInState())
        val disk = context.getSharedPreferences("gateway_test_credentials", Context.MODE_PRIVATE).getString("state", "")!!
        assertFalse(disk.contains("gateway-token"))
        val auth = GatewayAuth(context, settings, store)
        assertTrue(auth.signedIn.value)
        assertEquals("gateway-token-never-caller-token", auth.session().token)
        assertEquals("https://gateway.example", auth.session().server)
    }
    @Test fun serverSwitchClearsSessionAndCannotReuseIt() = runBlocking {
        store.write(signedInState())
        val auth = GatewayAuth(context, settings, store)
        assertTrue(auth.changeServer("https://different.example"))
        assertFalse(auth.signedIn.value)
        assertNull(store.read())
        try { auth.session(); fail("A changed server must require sign-in") } catch (_: CloudAuthException) { }
        assertTrue(auth.changeServer("https://gateway.example"))
        try { auth.session(); fail("Switching back must not restore credentials") } catch (_: CloudAuthException) { }
    }
    @Test fun invalidServerDoesNotDiscardSessionAndSignOutDoes() = runBlocking {
        store.write(signedInState())
        val auth = GatewayAuth(context, settings, store)
        assertFalse(auth.changeServer("http://insecure.example"))
        assertTrue(auth.signedIn.value)
        auth.signOut()
        assertFalse(auth.signedIn.value)
        assertNull(store.read())
        try { auth.session(); fail("Signed-out requests must fail") } catch (_: CloudAuthException) { }
    }
    @Test fun staleRedirectCannotRestoreSessionAfterSignOut() = runBlocking {
        val config = AuthorizationServiceConfiguration(Uri.parse("https://issuer.example/authorize"), Uri.parse("https://issuer.example/token"))
        val request = AuthorizationRequest.Builder(config, "gateway", ResponseTypeValues.CODE, Uri.parse(GatewayAuth.REDIRECT)).build()
        store.write(JSONObject().put("server", settings.endpoint.value).put("pending", request.jsonSerialize()).toString())
        val auth = GatewayAuth(context, settings, store)
        auth.signOut()
        val result = AuthorizationResponse.Builder(request).setAuthorizationCode("stale-code").build().toIntent()
        try { auth.complete(result); fail("Stale callback must not restore a session") } catch (_: IllegalStateException) { }
        assertNull(store.read())
    }
    @Test fun expiredGatewayTokenRefreshesAndPersistsRotation() = runBlocking {
        store.write(signedInState(expired = true))
        val body = java.io.ByteArrayOutputStream()
        val configuration = AppAuthConfiguration.Builder().setConnectionBuilder { uri ->
            object : java.net.HttpURLConnection(java.net.URL(uri.toString())) {
                override fun connect() { }
                override fun disconnect() { }
                override fun usingProxy() = false
                override fun getResponseCode() = 200
                override fun getOutputStream(): java.io.OutputStream = body
                override fun getInputStream(): java.io.InputStream = java.io.ByteArrayInputStream(
                    """{"access_token":"fresh-gateway-token","refresh_token":"rotated-refresh-token","token_type":"Bearer","expires_in":3600}""".toByteArray()
                )
            }
        }.build()
        val service = AuthorizationService(context, configuration)
        try {
            val auth = GatewayAuth(context, settings, store, service)
            assertEquals("fresh-gateway-token", auth.session().token)
            assertTrue(body.toString("UTF-8").contains("grant_type=refresh_token"))
            assertTrue(body.toString("UTF-8").contains("refresh_token=gateway-refresh-token"))
            assertTrue(store.read()!!.contains("rotated-refresh-token"))
            assertEquals("fresh-gateway-token", GatewayAuth(context, settings, store).session().token)
        } finally { service.dispose() }
    }

}
