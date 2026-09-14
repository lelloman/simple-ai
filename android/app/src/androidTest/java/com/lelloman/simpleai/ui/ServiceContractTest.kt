package com.lelloman.simpleai.ui

import android.content.*
import android.os.IBinder
import android.os.IInterface
import androidx.test.platform.app.InstrumentationRegistry
import com.lelloman.simpleai.BuildConfig
import com.lelloman.simpleai.ISimpleAI
import com.lelloman.simpleai.service.SimpleAIService
import kotlinx.serialization.json.*
import org.junit.Assert.*
import org.junit.Test
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit

class ServiceContractTest {
    @Test fun boundServiceParcelsDiscoveryAndRejectsInvalidGenerationWithoutModelWork() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val connected = CountDownLatch(1)
        var api: ISimpleAI? = null
        val connection = object : ServiceConnection {
            override fun onServiceConnected(name: ComponentName?, service: IBinder) {
                // Force generated AIDL Proxy to exercise parcel/unparcel rather than the local fast path.
                val proxyBinder = object : IBinder by service {
                    override fun queryLocalInterface(descriptor: String): IInterface? = null
                }
                api = ISimpleAI.Stub.asInterface(proxyBinder)
                connected.countDown()
            }
            override fun onServiceDisconnected(name: ComponentName?) { api = null }
        }
        try {
            assertTrue(context.bindService(Intent(context, SimpleAIService::class.java), connection, Context.BIND_AUTO_CREATE))
            assertTrue(connected.await(10, TimeUnit.SECONDS))
            val service = requireNotNull(api)
            val info = Json.parseToJsonElement(service.getServiceInfo(BuildConfig.MAX_PROTOCOL_VERSION)).jsonObject
            assertEquals("success", info["status"]?.jsonPrimitive?.content)
            assertNotNull(info["data"]?.jsonObject?.get("capabilities"))
            val rejected = Json.parseToJsonElement(service.localGenerate(BuildConfig.MAX_PROTOCOL_VERSION, "hello", 0, Float.NaN)).jsonObject
            assertEquals("INVALID_REQUEST", rejected["error"]?.jsonObject?.get("code")?.jsonPrimitive?.content)
            val old = Json.parseToJsonElement(service.getServiceInfo(1)).jsonObject
            assertEquals("error", old["status"]?.jsonPrimitive?.content)
        } finally { context.unbindService(connection) }
    }
}
