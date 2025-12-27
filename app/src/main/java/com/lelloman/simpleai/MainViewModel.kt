package com.lelloman.simpleai

import android.app.Application
import android.content.BroadcastReceiver
import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.content.ServiceConnection
import android.os.Build
import android.os.IBinder
import androidx.core.content.ContextCompat
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.lelloman.simpleai.service.LLMService
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

sealed class AppState {
    data object Starting : AppState()
    data class Downloading(val progress: Float) : AppState()
    data object Loading : AppState()
    data object Ready : AppState()
    data class Error(val message: String) : AppState()
}

class MainViewModel(application: Application) : AndroidViewModel(application) {

    private val _state = MutableStateFlow<AppState>(AppState.Starting)
    val state: StateFlow<AppState> = _state.asStateFlow()

    private var llmService: ILLMService? = null
    private var isBound = false

    private val serviceConnection = object : ServiceConnection {
        override fun onServiceConnected(name: ComponentName?, service: IBinder?) {
            llmService = ILLMService.Stub.asInterface(service)
            updateStateFromService()
        }

        override fun onServiceDisconnected(name: ComponentName?) {
            llmService = null
            isBound = false
        }
    }

    private val statusReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context?, intent: Intent?) {
            if (intent?.action == LLMService.ACTION_STATUS_UPDATE) {
                updateStateFromService()
            }
        }
    }

    init {
        startAndBindService()
        registerStatusReceiver()
    }

    private fun startAndBindService() {
        val context = getApplication<Application>()
        val serviceIntent = Intent(context, LLMService::class.java)

        // Start as foreground service
        ContextCompat.startForegroundService(context, serviceIntent)

        // Bind to it
        context.bindService(serviceIntent, serviceConnection, Context.BIND_AUTO_CREATE)
        isBound = true
    }

    private fun registerStatusReceiver() {
        val context = getApplication<Application>()
        val filter = IntentFilter(LLMService.ACTION_STATUS_UPDATE)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            context.registerReceiver(statusReceiver, filter, Context.RECEIVER_NOT_EXPORTED)
        } else {
            context.registerReceiver(statusReceiver, filter)
        }
    }

    private fun updateStateFromService() {
        viewModelScope.launch {
            val service = llmService ?: return@launch
            val status = service.status

            _state.value = when {
                status == "ready" -> AppState.Ready
                status == "loading" -> AppState.Loading
                status == "downloading" -> AppState.Downloading(0f)
                status == "initializing" -> AppState.Starting
                status.startsWith("error:") -> AppState.Error(status.removePrefix("error: "))
                else -> AppState.Starting
            }
        }
    }

    fun testGenerate(prompt: String, onResult: (String) -> Unit) {
        viewModelScope.launch {
            val service = llmService
            if (service == null) {
                onResult("Error: Service not connected")
                return@launch
            }
            try {
                val result = service.generate(prompt)
                onResult(result)
            } catch (e: Exception) {
                onResult("Error: ${e.message}")
            }
        }
    }

    override fun onCleared() {
        super.onCleared()
        val context = getApplication<Application>()
        if (isBound) {
            context.unbindService(serviceConnection)
            isBound = false
        }
        try {
            context.unregisterReceiver(statusReceiver)
        } catch (_: Exception) {}
    }
}
