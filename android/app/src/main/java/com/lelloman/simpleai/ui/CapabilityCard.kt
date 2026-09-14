package com.lelloman.simpleai.ui

import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Check
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.semantics.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.lelloman.simpleai.capability.CapabilityStatus

/**
 * Card displaying a capability and its status.
 */
@OptIn(androidx.compose.foundation.layout.ExperimentalLayoutApi::class)
@Composable
fun CapabilityCard(
    title: String,
    icon: String,
    description: String,
    status: CapabilityStatus,
    onDownload: (() -> Unit)? = null,
    downloadJob: String? = null,
    onPause: (() -> Unit)? = null,
    onRetry: (() -> Unit)? = null,
    onDelete: (() -> Unit)? = null,
    onConfigure: (() -> Unit)? = null,
    onTest: (() -> Unit)? = null,
    extraContent: @Composable (() -> Unit)? = null,
    modifier: Modifier = Modifier
) {
    Card(modifier = modifier.fillMaxWidth()) {
        Column(Modifier.fillMaxWidth().padding(16.dp), verticalArrangement = androidx.compose.foundation.layout.Arrangement.spacedBy(8.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text(icon, modifier = Modifier.clearAndSetSemantics {}, style = MaterialTheme.typography.headlineSmall)
                Spacer(Modifier.width(12.dp))
                Column(Modifier.weight(1f)) {
                    Text(title, style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)
                    Text(description, style = MaterialTheme.typography.bodySmall)
                }
            }
            if (downloadJob in listOf("ENQUEUED", "BLOCKED", "RUNNING")) {
                Text(if (downloadJob == "RUNNING") "Download in progress" else "Download queued — waiting for network or scheduler")
                onPause?.let { TextButton(onClick = it) { Text("Pause $title download") } }
            } else if (downloadJob == "CANCELLED") {
                Text("Download paused. Retry or Download resumes saved progress.")
            } else if (downloadJob == "FAILED") {
                Text("Download interrupted. Retry to resume.")
            }
            if (downloadJob in listOf("CANCELLED", "FAILED") && status !is CapabilityStatus.Ready && status != CapabilityStatus.Downloaded) {
                onDelete?.let { TextButton(onClick = it) { Text("Remove partial $title download") } }
            }
            when (status) {
                CapabilityStatus.Checking, CapabilityStatus.Loading -> {
                    LinearProgressIndicator(Modifier.fillMaxWidth())
                    Text(if (status == CapabilityStatus.Checking) "Checking availability…" else "Loading model…")
                }
                is CapabilityStatus.NotDownloaded -> {
                    Text("Not downloaded" + if (status.totalBytes > 0) " • ${formatSize(status.totalBytes)}" else "")
                    onDownload?.let { Button(onClick = it) { Text("Download $title") } }
                }
                is CapabilityStatus.Downloading -> {
                    LinearProgressIndicator(progress = { status.progress }, modifier = Modifier.fillMaxWidth().semantics { contentDescription = "$title download progress" })
                    Text("Downloading… ${(status.progress * 100).toInt()}%")
                    if (status.totalBytes > 0) Text("${formatSize(status.downloadedBytes)} / ${formatSize(status.totalBytes)}")
                }
                CapabilityStatus.Downloaded, CapabilityStatus.Ready -> {
                    Text(if (status == CapabilityStatus.Downloaded) "Downloaded • loads when needed" else "Ready")
                    androidx.compose.foundation.layout.FlowRow(horizontalArrangement = androidx.compose.foundation.layout.Arrangement.spacedBy(8.dp)) {
                        onDelete?.let { TextButton(onClick = it) { Text("Delete $title") } }
                        onTest?.let { OutlinedButton(onClick = it) { Text("Test $title") } }
                    }
                }
                is CapabilityStatus.Error -> {
                    Text(status.message, color = MaterialTheme.colorScheme.error, modifier = Modifier.semantics { liveRegion = LiveRegionMode.Polite })
                    if (status.canRetry) onRetry?.let { TextButton(onClick = it) { Text("Retry $title") } }
                }
            }
            onConfigure?.let { OutlinedButton(onClick = it) { Text(if (status is CapabilityStatus.NotDownloaded) "Download languages" else "Manage languages") } }
            extraContent?.invoke()
        }
    }
}

private fun formatSize(bytes: Long): String {
    return when {
        bytes >= 1024 * 1024 * 1024 -> String.format("%.1f GB", bytes / (1024.0 * 1024.0 * 1024.0))
        bytes >= 1024 * 1024 -> String.format("%.0f MB", bytes / (1024.0 * 1024.0))
        bytes >= 1024 -> String.format("%.0f KB", bytes / 1024.0)
        else -> "$bytes B"
    }
}
