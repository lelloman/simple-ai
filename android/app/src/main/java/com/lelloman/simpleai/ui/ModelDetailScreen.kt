package com.lelloman.simpleai.ui

import com.lelloman.lellodesign.LelloButton
import com.lelloman.lellodesign.LelloOutlinedButton
import com.lelloman.lellodesign.LelloTextButton

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.unit.dp
import com.lelloman.simpleai.R
import com.lelloman.simpleai.capability.CapabilityStatus
import com.lelloman.simpleai.model.LocalAIModel
import com.lelloman.simpleai.model.NluModel

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ModelDetailScreen(model: String, viewModel: CapabilitiesViewModel, onBack: () -> Unit) {
    val state by viewModel.state.collectAsState()
    val voice = model == "voice"
    val title = stringResource(if (voice) R.string.ui_voice_commands else R.string.ui_local_ai)
    val size = if (voice) NluModel.SIZE_BYTES else LocalAIModel.SIZE_BYTES
    val status = if (voice) state.voiceCommandsStatus else state.localAiStatus
    val job = state.downloadJobs[model]
    val downloaded = status == CapabilityStatus.Ready || status == CapabilityStatus.Downloaded
    val running = job in listOf("ENQUEUED", "BLOCKED", "RUNNING")
    var confirmDelete by remember { mutableStateOf(false) }
    if (confirmDelete) AlertDialog(
        onDismissRequest = { confirmDelete = false },
        title = { Text(stringResource(R.string.ui_delete, title)) },
        text = { Text(stringResource(R.string.model_delete_body)) },
        confirmButton = { LelloTextButton(onClick = {
            if (voice) viewModel.deleteVoiceCommands() else viewModel.deleteLocalAi()
            confirmDelete = false
        }) { Text(stringResource(R.string.ui_delete_2)) } },
        dismissButton = { LelloTextButton(onClick = { confirmDelete = false }) { Text(stringResource(R.string.ui_cancel)) } }
    )
    SimplePage(title, onBack) {
        Text(stringResource(if (voice) R.string.model_voice_purpose else R.string.model_local_purpose), style = MaterialTheme.typography.bodyLarge)
        Text(if (voice) "XLM-RoBERTa int8" else LocalAIModel.NAME, style = MaterialTheme.typography.titleMedium)
        Text(stringResource(R.string.model_download_size, formatSize(size)), color = MaterialTheme.colorScheme.onSurfaceVariant)
        if (voice && !downloaded) Text(stringResource(R.string.model_voice_space, formatSize(size * 2)), style = MaterialTheme.typography.bodySmall)
        when {
            downloaded -> {
                Text(stringResource(R.string.model_ready), color = MaterialTheme.colorScheme.primary)
                LelloOutlinedButton(onClick = { confirmDelete = true }) { Text(stringResource(R.string.model_remove)) }
            }
            running || status is CapabilityStatus.Downloading -> {
                if (status is CapabilityStatus.Downloading) {
                    LinearProgressIndicator(progress = { status.progress }, modifier = Modifier.fillMaxWidth())
                    Text(stringResource(R.string.ui_value_2, formatSize(status.downloadedBytes), formatSize(status.totalBytes)))
                } else {
                    LinearProgressIndicator(Modifier.fillMaxWidth())
                    Text(stringResource(R.string.model_queued))
                }
                LelloOutlinedButton(onClick = { viewModel.pauseDownload(model) }) { Text(stringResource(R.string.model_pause)) }
            }
            status == CapabilityStatus.Checking || status == CapabilityStatus.Loading -> {
                LinearProgressIndicator(Modifier.fillMaxWidth())
                Text(stringResource(R.string.model_checking))
            }
            else -> {
                if (status is CapabilityStatus.Error && job != "CANCELLED") Text(status.message, color = MaterialTheme.colorScheme.error)
                val partial = job in listOf("CANCELLED", "FAILED") && status is CapabilityStatus.Error
                LelloButton(onClick = { if (voice) viewModel.downloadVoiceCommands() else viewModel.downloadLocalAi() }, enabled = status !is CapabilityStatus.Error || status.canRetry) {
                    Text(stringResource(if (partial) R.string.model_resume else R.string.model_download))
                }
                if (partial) LelloTextButton(onClick = { confirmDelete = true }) { Text(stringResource(R.string.model_remove_partial)) }
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
internal fun SimplePage(title: String, onBack: (() -> Unit)? = null, content: @Composable ColumnScope.() -> Unit) {
    com.lelloman.lellodesign.LelloWorkspace(
        Modifier.widthIn(max = 720.dp).fillMaxSize().verticalScroll(androidx.compose.foundation.rememberScrollState()), content = content
    )
}
