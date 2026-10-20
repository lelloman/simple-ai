package com.lelloman.simpleai.ui

import com.lelloman.simpleai.access.ClientAccess
import org.junit.Assert.assertEquals
import org.junit.Test

class ConnectedAppsTest {
    @Test fun packageIdentifiersAreTrimmedDeduplicatedAndEmptySafe() {
        assertEquals(listOf("com.example.one", "com.example.two"), parsePackageNames(" com.example.one, com.example.two ,com.example.one,"))
        assertEquals(emptyList<String>(), parsePackageNames(" , ,"))
    }

    @Test fun pendingAppsSortBeforeApprovedAppsAndThenByPackage() {
        val clients = listOf(
            ClientAccess.Client("a", "com.example.z", true),
            ClientAccess.Client("b", "com.example.b", false),
            ClientAccess.Client("c", "com.example.a", false),
            ClientAccess.Client("d", "com.example.a", true),
        )
        assertEquals(listOf("c", "b", "d", "a"), orderConnectedApps(clients).map { it.identity })
    }
}
