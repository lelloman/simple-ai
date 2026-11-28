package com.lelloman.simpleai.ui

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.unit.dp
import com.lelloman.lellodesign.LelloAlert
import com.lelloman.lellodesign.LelloButton
import com.lelloman.lellodesign.LelloDimensions
import com.lelloman.lellodesign.LelloOutlinedButton
import com.lelloman.lellodesign.LelloSection
import com.lelloman.lellodesign.LelloTextStyles
import com.lelloman.lellodesign.LelloTone
import com.lelloman.lellodesign.LelloWorkspace
import com.lelloman.lellodesign.LocalLelloPalette
import com.lelloman.simpleai.R
import com.lelloman.simpleai.capability.CapabilityStatus
import com.lelloman.simpleai.model.LocalAIModel
import com.lelloman.simpleai.model.NluModel

@Composable
fun ModelDetailScreen(model: String, viewModel: CapabilitiesViewModel, @Suppress("UNUSED_PARAMETER") onBack: () -> Unit) {
    val state by viewModel.state.collectAsState()
    val voice = model == "voice"
    ModelDetailContent(
        voice = voice,
        status = if (voice) state.voiceCommandsStatus else state.localAiStatus,
        job = state.downloadJobs[model],
        onDownload = { if (voice) viewModel.downloadVoiceCommands() else viewModel.downloadLocalAi() },
        onPause = { viewModel.pauseDownload(model) },
        onDelete = { if (voice) viewModel.deleteVoiceCommands() else viewModel.deleteLocalAi() },
    )
}

@Composable
internal fun ModelDetailContent(
    voice: Boolean,
    status: CapabilityStatus,
    job: String?,
    onDownload: () -> Unit,
    onPause: () -> Unit,
    onDelete: () -> Unit,
) {
    val title = stringResource(if (voice) R.string.ui_voice_commands else R.string.ui_local_ai)
    val size = if (voice) NluModel.SIZE_BYTES else LocalAIModel.SIZE_BYTES
    val downloaded = status == CapabilityStatus.Ready || status == CapabilityStatus.Downloaded
    val running = job in listOf("ENQUEUED", "BLOCKED", "RUNNING")
    val partial = job in listOf("CANCELLED", "FAILED") && status is CapabilityStatus.Error
    val secondary = LocalLelloPalette.current["text-secondary"]
    val track = LocalLelloPalette.current["neutral"]
    var confirmDelete by remember { mutableStateOf(false) }
    if (confirmDelete) CalmDialog(
        title = stringResource(R.string.ui_delete, title),
        onDismiss = { confirmDelete = false },
        confirmLabel = stringResource(R.string.ui_delete_2),
        onConfirm = { onDelete(); confirmDelete = false },
        dismissLabel = stringResource(R.string.ui_cancel),
        danger = true,
    ) { Text(stringResource(R.string.model_delete_body)) }
    SimplePage {
        Text(stringResource(if (voice) R.string.model_voice_purpose else R.string.model_local_purpose),
            style = MaterialTheme.typography.bodyLarge)
        LelloSection(stringResource(R.string.model_details)) {
            Column {
                DetailRow(stringResource(R.string.model_detail_model)) {
                    Text(if (voice) "XLM-RoBERTa int8" else LocalAIModel.NAME, style = MaterialTheme.typography.bodyMedium)
                }
                RowDivider()
                DetailRow(stringResource(R.string.model_detail_size)) { Text(formatSize(size), style = LelloTextStyles.numeral) }
                RowDivider()
                DetailRow(stringResource(R.string.model_detail_status)) {
                    val summary = modelStatus(status, job)
                    if (summary != null) StatusPill(summary.label, summary.tone)
                    else StatusPill(stringResource(R.string.model_not_downloaded), PillTone.Neutral)
                }
                RowDivider()
            }
            if (voice && !downloaded) Text(stringResource(R.string.model_voice_space, formatSize(size * 2)),
                style = LelloTextStyles.caption)
        }
        Column(verticalArrangement = Arrangement.spacedBy(LelloDimensions.space3)) {
            when {
                downloaded -> {
                    Text(stringResource(R.string.model_ready), style = MaterialTheme.typography.bodyMedium, color = secondary)
                    QuietDangerButton(stringResource(R.string.model_remove), { confirmDelete = true }, icon = R.drawable.ic_delete)
                }
                running || status is CapabilityStatus.Downloading -> {
                    if (status is CapabilityStatus.Downloading) {
                        LinearProgressIndicator(progress = { status.progress }, modifier = Modifier.fillMaxWidth(),
                            trackColor = track, drawStopIndicator = {})
                        Text(stringResource(R.string.ui_value_2, formatSize(status.downloadedBytes), formatSize(status.totalBytes)),
                            style = LelloTextStyles.timestamp, color = secondary)
                    } else {
                        LinearProgressIndicator(Modifier.fillMaxWidth(), trackColor = track)
                        Text(stringResource(R.string.model_queued), style = MaterialTheme.typography.bodyMedium, color = secondary)
                    }
                    LelloOutlinedButton(onClick = onPause) { Text(stringResource(R.string.model_pause)) }
                }
                status == CapabilityStatus.Checking || status == CapabilityStatus.Loading -> {
                    LinearProgressIndicator(Modifier.fillMaxWidth(), trackColor = track)
                    Text(stringResource(R.string.model_checking), style = MaterialTheme.typography.bodyMedium, color = secondary)
                }
                else -> {
                    if (status is CapabilityStatus.Error && job != "CANCELLED") {
                        LelloAlert(stringResource(R.string.model_needs_attention), tone = LelloTone.Error) { Text(status.message) }
                    }
                    LelloButton(onClick = onDownload, enabled = status !is CapabilityStatus.Error || status.canRetry) {
                        Icon(painterResource(R.drawable.ic_download), null, Modifier.size(18.dp))
                        Spacer(Modifier.width(8.dp))
                        Text(stringResource(if (partial) R.string.model_resume else R.string.model_download))
                    }
                    if (partial) QuietDangerButton(stringResource(R.string.model_remove_partial), { confirmDelete = true })
                }
            }
        }
    }
}

/** Scrolling single-column page constrained to a readable width inside the workspace. */
@Composable
internal fun SimplePage(content: @Composable ColumnScope.() -> Unit) {
    LelloWorkspace(
        Modifier.widthIn(max = 720.dp).fillMaxSize().verticalScroll(androidx.compose.foundation.rememberScrollState()), content = content
    )
}
