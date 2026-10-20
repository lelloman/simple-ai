package com.lelloman.simpleai.cloud

import android.util.Base64
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.jsonObject

/** Account identity shown in Settings. A null username means the provider supplied no readable name. */
data class GatewayAccount(val server: String, val username: String?)

/**
 * Reads the account name from the OIDC ID token. These claims are display-only; authentication
 * still depends on the validated session. No name is invented when every supported claim is absent.
 */
object GatewayIdentity {
    private val usernameClaims = listOf("preferred_username", "nickname", "username", "name", "email")

    fun usernameFromIdToken(idToken: String?): String? {
        val payload = idToken?.split('.')?.getOrNull(1)?.takeIf { it.isNotEmpty() } ?: return null
        val normalized = payload.replace('-', '+').replace('_', '/')
        val padded = normalized + "=".repeat((4 - normalized.length % 4) % 4)
        val claims = runCatching {
            String(Base64.decode(padded, Base64.NO_WRAP), Charsets.UTF_8)
        }.getOrNull() ?: return null
        return usernameFromClaims(claims)
    }

    fun usernameFromClaims(claimsJson: String): String? = runCatching {
        val claims = Json.parseToJsonElement(claimsJson).jsonObject
        usernameClaims.firstNotNullOfOrNull { claim ->
            (claims[claim] as? JsonPrimitive)?.takeIf { it.isString }?.contentOrNull?.trim()?.takeIf { it.isNotEmpty() }
        }
    }.getOrNull()
}
