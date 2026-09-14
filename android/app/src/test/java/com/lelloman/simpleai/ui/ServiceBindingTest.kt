package com.lelloman.simpleai.ui

import android.content.Context
import android.content.Intent
import io.mockk.*
import org.junit.Assert.*
import org.junit.Test

class ServiceBindingTest {
    @Test fun disconnectStillRequiresUnbindAndCleanupIsIdempotent() {
        val context = mockk<Context>(relaxed = true)
        every { context.bindService(any(), any(), any<Int>()) } returns true
        val binding = ServiceBinding(context, { mockk<Intent>() }) { _, _ -> }
        binding.connect()
        binding.onServiceDisconnected(null)
        binding.close()
        binding.close()
        verify(exactly = 1) { context.unbindService(binding) }
    }

    @Test fun failedBindReportsErrorAndReleasesRegistration() {
        val context = mockk<Context>(relaxed = true)
        every { context.bindService(any(), any(), any<Int>()) } returns false
        var error: String? = null
        val binding = ServiceBinding(context, { mockk<Intent>() }) { _, message -> error = message }
        binding.connect()
        assertNotNull(error)
        verify(exactly = 1) { context.unbindService(binding) }
    }

    @Test fun deadAndNullBindingsCanBeReconnectedWithoutLeaking() {
        val context = mockk<Context>(relaxed = true)
        every { context.bindService(any(), any(), any<Int>()) } returns true
        val binding = ServiceBinding(context, { mockk<Intent>() }) { _, _ -> }
        binding.connect()
        binding.onBindingDied(null)
        binding.connect()
        binding.onNullBinding(null)
        binding.close()
        verify(exactly = 2) { context.unbindService(binding) }
        verify(exactly = 2) { context.bindService(any(), binding, Context.BIND_AUTO_CREATE) }
    }
}
