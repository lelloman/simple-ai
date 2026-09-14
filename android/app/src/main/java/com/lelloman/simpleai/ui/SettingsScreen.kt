package com.lelloman.simpleai.ui

import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.res.stringResource
import com.lelloman.simpleai.R
import com.lelloman.simpleai.capability.CapabilityStatus

@Composable
fun SettingsScreen(viewModel: CapabilitiesViewModel, onAbout: () -> Unit) {
    val strings = androidx.compose.ui.platform.LocalContext.current
    val state by viewModel.state.collectAsState()
    var cloudInfo by remember { mutableStateOf(false) }
    SimplePage(stringResource(R.string.nav_settings)) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Column(Modifier.weight(1f)) {
                Text(stringResource(R.string.settings_mobile), style = MaterialTheme.typography.titleMedium)
                Text(stringResource(R.string.settings_mobile_hint), style = MaterialTheme.typography.bodySmall)
            }
            Switch(state.allowMeteredDownloads, viewModel::setAllowMeteredDownloads, modifier = Modifier.semantics { contentDescription = strings.getString(R.string.settings_mobile) })
        }
        HorizontalDivider()
        state.storage?.let {
            Text(stringResource(R.string.settings_storage), style = MaterialTheme.typography.titleMedium)
            Text(stringResource(R.string.settings_storage_values, formatSize(it.usedBytes), formatSize(it.availableBytes)), color = MaterialTheme.colorScheme.onSurfaceVariant)
        }
        HorizontalDivider()
        ModelRow(stringResource(R.string.ui_cloud_ai), stringResource(if (state.cloudAiStatus == CapabilityStatus.Ready) R.string.settings_configured else R.string.settings_not_configured), { cloudInfo = true })
        state.serviceError?.let {
            Text(it, color = MaterialTheme.colorScheme.error)
            TextButton(onClick = viewModel::refreshCapabilities) { Text(stringResource(R.string.ui_retry_connection)) }
        }
        ModelRow(stringResource(R.string.settings_about), stringResource(R.string.settings_about_hint), onAbout)
    }
    if (cloudInfo) AlertDialog(onDismissRequest = { cloudInfo = false }, title = { Text(stringResource(R.string.ui_cloud_ai)) }, text = {
        Text(stringResource(if (state.cloudAiStatus == CapabilityStatus.Ready) R.string.settings_cloud_ready else R.string.settings_cloud_unavailable))
    }, confirmButton = { TextButton(onClick = { cloudInfo = false }) { Text(stringResource(R.string.ui_close)) } })
}
