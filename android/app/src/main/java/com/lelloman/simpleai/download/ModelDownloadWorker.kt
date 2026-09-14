package com.lelloman.simpleai.download

import android.app.NotificationChannel
import android.app.NotificationManager
import android.content.Context
import android.content.pm.ServiceInfo
import android.os.Build
import androidx.core.app.NotificationCompat
import androidx.work.*
import com.lelloman.simpleai.R
import com.lelloman.simpleai.capability.CapabilityStatus
import com.lelloman.simpleai.model.ModelRepository
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.launch

class ModelDownloadWorker(context: Context, params: WorkerParameters) : CoroutineWorker(context, params) {
    companion object {
        const val VOICE = "voice"
        const val LOCAL = "local"
        fun name(model: String) = "model-download-$model"
        fun enqueue(context: Context, model: String) {
            require(model == VOICE || model == LOCAL)
            val request = OneTimeWorkRequestBuilder<ModelDownloadWorker>()
                .setInputData(workDataOf("model" to model))
                .setConstraints(Constraints.Builder().setRequiredNetworkType(NetworkType.CONNECTED).build())
                .build()
            WorkManager.getInstance(context).enqueueUniqueWork(name(model), ExistingWorkPolicy.KEEP, request)
        }
    }

    override suspend fun doWork(): Result = coroutineScope {
        val model = inputData.getString("model") ?: return@coroutineScope Result.failure()
        if (model != VOICE && model != LOCAL) return@coroutineScope Result.failure()
        val repository = ModelRepository.get(applicationContext)
        val status = if (model == VOICE) repository.capabilities.voiceCommandsStatus else repository.capabilities.localAiStatus
        setForeground(notification(model, "Starting download", null))
        val progress = launch {
            var lastUpdate = 0L
            status.collect { state ->
                val now = System.currentTimeMillis()
                if (state !is CapabilityStatus.Downloading || now - lastUpdate > 1000) {
                    val percent = (state as? CapabilityStatus.Downloading)?.takeIf { it.totalBytes > 0 }?.let { (it.progress * 100).toInt() }
                    setProgress(workDataOf("percent" to (percent ?: -1)))
                    setForeground(notification(model, if (state == CapabilityStatus.Loading) "Loading model" else "Downloading model", percent))
                    lastUpdate = now
                }
            }
        }
        try {
            if (model == VOICE) repository.voice.downloadAndActivate() else repository.local.downloadAndActivate()
            if (status.value == CapabilityStatus.Ready) Result.success() else Result.failure()
        } finally { progress.cancel() }
    }

    private fun notification(model: String, text: String, percent: Int?): ForegroundInfo {
        val manager = applicationContext.getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
        val channel = "model_downloads"
        if (Build.VERSION.SDK_INT >= 26) manager.createNotificationChannel(NotificationChannel(channel, "Model downloads", NotificationManager.IMPORTANCE_LOW))
        val notification = NotificationCompat.Builder(applicationContext, channel)
            .setSmallIcon(R.drawable.ic_notification)
            .setContentTitle(if (model == VOICE) "Voice Commands" else "Local AI")
            .setContentText(text)
            .setProgress(100, percent ?: 0, percent == null)
            .setOngoing(true)
            .addAction(android.R.drawable.ic_media_pause, "Pause", WorkManager.getInstance(applicationContext).createCancelPendingIntent(id))
            .build()
        val notificationId = if (model == VOICE) 201 else 202
        return if (Build.VERSION.SDK_INT >= 29) ForegroundInfo(notificationId, notification, ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC)
            else ForegroundInfo(notificationId, notification)
    }
}
