package com.lelloman.simpleai.cloud

import com.lelloman.simpleai.capability.CapabilityStatus
import org.junit.Assert.*
import org.junit.Test

class CloudEndpointTest {
    @Test fun invalidConfigurationIsUnavailable() {
        for (endpoint in listOf("", "nonsense", "http://cloud.example", "https://user:pass@cloud.example", "https://cloud.example?token=secret")) {
            assertNull(CloudEndpoint.chatUrl(endpoint))
            val status = CloudEndpoint.status(endpoint) as CapabilityStatus.Error
            assertFalse(status.canRetry)
        }
    }

    @Test fun configuredEndpointNormalizesTrailingSlash() {
        assertEquals("https://cloud.example/api/v1/chat/completions", CloudEndpoint.chatUrl("https://cloud.example/api/").toString())
        assertEquals(CapabilityStatus.Ready, CloudEndpoint.status("https://cloud.example"))
    }
}
