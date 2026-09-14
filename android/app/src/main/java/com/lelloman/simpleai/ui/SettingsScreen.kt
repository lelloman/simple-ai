package com.lelloman.simpleai.ui

import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.ui.text.input.KeyboardType
import com.lelloman.simpleai.cloud.CloudEndpoint
import kotlinx.coroutines.launch
import androidx.compose.ui.unit.dp
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.res.stringResource
import com.lelloman.simpleai.R

@Composable
fun SettingsScreen(viewModel: CapabilitiesViewModel, onAbout: () -> Unit) {
    val strings = androidx.compose.ui.platform.LocalContext.current
    val state by viewModel.state.collectAsState()
    val endpoint by viewModel.cloudEndpoint.collectAsState()
    var cloudInfo by rememberSaveable { mutableStateOf(false) }
    var draft by rememberSaveable { mutableStateOf("") }
    var saveFailed by remember { mutableStateOf(false) }
    var saving by remember { mutableStateOf(false) }
    val scope = rememberCoroutineScope()
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
        ModelRow(stringResource(R.string.ui_cloud_ai), endpoint.ifBlank { stringResource(R.string.settings_not_configured) }, { draft = endpoint; saveFailed = false; cloudInfo = true })
        state.serviceError?.let {
            Text(it, color = MaterialTheme.colorScheme.error)
            TextButton(onClick = viewModel::refreshCapabilities) { Text(stringResource(R.string.ui_retry_connection)) }
        }
        ModelRow(stringResource(R.string.settings_about), stringResource(R.string.settings_about_hint), onAbout)
    }
    if (cloudInfo) {
        val valid = draft.isBlank() || CloudEndpoint.chatUrl(draft) != null
        AlertDialog(
            onDismissRequest = { if (!saving) cloudInfo = false },
            title = { Text(stringResource(R.string.cloud_server_title)) },
            text = {
                Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
                    OutlinedTextField(
                        value = draft, onValueChange = { draft = it; saveFailed = false },
                        label = { Text(stringResource(R.string.cloud_server_url)) },
                        placeholder = { Text("https://ai.example.com") },
                        singleLine = true, enabled = !saving, isError = !valid,
                        keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Uri),
                        supportingText = { Text(stringResource(if (valid) R.string.cloud_server_hint else R.string.cloud_server_invalid)) }
                    )
                    Text(stringResource(R.string.cloud_server_auth), style = MaterialTheme.typography.bodySmall)
                    if (saveFailed) Text(stringResource(R.string.cloud_server_save_failed), color = MaterialTheme.colorScheme.error)
                }
            },
            confirmButton = {
                TextButton(enabled = valid && !saving, onClick = {
                    saving = true
                    scope.launch {
                        try {
                            if (viewModel.saveCloudEndpoint(draft)) cloudInfo = false else saveFailed = true
                        } finally { saving = false }
                    }
                }) { Text(stringResource(R.string.cloud_server_save)) }
            },
            dismissButton = { TextButton(enabled = !saving, onClick = { cloudInfo = false }) { Text(stringResource(R.string.cloud_server_cancel)) } }
        )
    }
}
