package com.lelloman.simpleai.cloud

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class GatewayIdentityTest {
    @Test fun prefersTheOidcUsernameClaim() {
        assertEquals("alice", GatewayIdentity.usernameFromClaims(
            """{"preferred_username":"alice","nickname":"ali","name":"Alice Smith","email":"alice@example.com"}"""
        ))
    }

    @Test fun fallsBackToNicknameThenNameThenEmail() {
        assertEquals("ali", GatewayIdentity.usernameFromClaims(
            """{"nickname":"ali","name":"Alice Smith","email":"alice@example.com"}"""
        ))
        assertEquals("Alice Smith", GatewayIdentity.usernameFromClaims(
            """{"name":"Alice Smith","email":"alice@example.com"}"""
        ))
        assertEquals("alice@example.com", GatewayIdentity.usernameFromClaims(
            """{"email":"alice@example.com"}"""
        ))
    }

    @Test fun ignoresBlankAndNonTextClaims() {
        assertEquals("Bob", GatewayIdentity.usernameFromClaims(
            """{"preferred_username":"  ","nickname":42,"name":"Bob","email":"bob@example.com"}"""
        ))
        assertNull(GatewayIdentity.usernameFromClaims(
            """{"preferred_username":"","nickname":null,"name":{},"email":"   "}"""
        ))
    }

    @Test fun returnsNullForMalformedClaims() {
        assertNull(GatewayIdentity.usernameFromClaims("not json"))
        assertNull(GatewayIdentity.usernameFromClaims("[]"))
    }
}
