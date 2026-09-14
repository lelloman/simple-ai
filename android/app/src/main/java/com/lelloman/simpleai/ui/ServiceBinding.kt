package com.lelloman.simpleai.ui

import android.content.*
import android.os.IBinder
import com.lelloman.simpleai.ISimpleAI
import com.lelloman.simpleai.service.SimpleAIService

/** Binding registration survives a temporary service-process disconnect. */
internal class ServiceBinding(
    private val context: Context,
    private val intent: () -> Intent = { Intent(context, SimpleAIService::class.java) },
    private val changed: (ISimpleAI?, String?) -> Unit
) : ServiceConnection {
    private var registered = false

    fun connect() {
        close()
        try {
            registered = true
            if (!context.bindService(intent(), this, Context.BIND_AUTO_CREATE)) {
                // Android requires unbinding even when bindService returns false.
                close()
                changed(null, "Could not bind to SimpleAI. Retry connection.")
            }
        } catch (e: Exception) { close(); changed(null, "Could not connect: ${e.message}") }
    }

    override fun onServiceConnected(name: ComponentName?, service: IBinder?) {
        val api = ISimpleAI.Stub.asInterface(service)
        if (api == null) onNullBinding(name) else changed(api, null)
    }

    override fun onServiceDisconnected(name: ComponentName?) {
        changed(null, "SimpleAI disconnected. Waiting for reconnection; you can also retry.")
    }

    override fun onBindingDied(name: ComponentName?) {
        close()
        changed(null, "Service binding expired. Retry connection.")
    }

    override fun onNullBinding(name: ComponentName?) {
        close()
        changed(null, "Service did not provide a connection. Retry connection.")
    }

    fun close() {
        if (registered) {
            registered = false
            try { context.unbindService(this) } catch (_: IllegalArgumentException) { /* already removed by framework */ }
        }
    }
}
