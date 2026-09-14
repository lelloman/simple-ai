package com.lelloman.simpleai.ui

import com.lelloman.simpleai.R

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
                changed(null, context.getString(R.string.ui_could_not_bind))
            }
        } catch (e: Exception) { close(); changed(null, context.getString(R.string.ui_connect_error, e.message.orEmpty())) }
    }

    override fun onServiceConnected(name: ComponentName?, service: IBinder?) {
        val api = ISimpleAI.Stub.asInterface(service)
        if (api == null) onNullBinding(name) else changed(api, null)
    }

    override fun onServiceDisconnected(name: ComponentName?) {
        changed(null, context.getString(R.string.ui_waiting_reconnection))
    }

    override fun onBindingDied(name: ComponentName?) {
        close()
        changed(null, context.getString(R.string.ui_binding_expired))
    }

    override fun onNullBinding(name: ComponentName?) {
        close()
        changed(null, context.getString(R.string.ui_null_binding))
    }

    fun close() {
        if (registered) {
            registered = false
            try { context.unbindService(this) } catch (_: IllegalArgumentException) { /* already removed by framework */ }
        }
    }
}
